#!/usr/bin/env python3
"""
generate_qa.py
════════════════════════════════════════════════════════════════════════════════
Generate Q&A pairs from the evolving-concept threads.

Input : concept_threads/cluster_*.json with num_member_objects > 1
Output: concept_qa/<same name>_QA.json  — one file per source thread

Six question types are produced (each Q&A carries a "type" key):
  1 scenario_short_answer     — real-world scenario question + short answer
  2 temporal_mcq              — temporal multiple-choice (when/what-changed)
  3 event_sorting             — order shuffled events by time
  4 open_ended                — analytical open question + answer
  5 perturbation_detection    — a fact is perturbed; detect & correct it
  6 context_misalignment      — misaligned context given; resolve the conflict

One LLM call per (thread, type) so each type is generated in depth; results are
aggregated into a single QA file per thread. Grounded in the thread's summary,
evolution, timeline and member clauses (full original text + entities + concepts).

Keys/model: free keys only (all_api.txt) + gemini-3.1-flash-lite (inherited).

Usage:
    python generate_qa.py
    python generate_qa.py --limit 3          # smoke test
    python generate_qa.py --only cluster_00326_t0
    python generate_qa.py --force            # regenerate existing QA files
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.genai import types
from tqdm import tqdm

from break_subentities import GEMINI_MODEL, _call_with_watchdog, setup_logging
from cluster_entities import ApiKeyRotator, KeyPool, _build_judge_rotator, _classify_err

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
THREADS_DIR = SCRIPT_DIR / "concept_threads"
OUT_DIR = SCRIPT_DIR / "concept_qa"

CONCURRENCY = 16
WATCHDOG = 120
MAX_CLAUSES = 22                 # member clauses shown (sampled across time if more)
CLAUSE_CHARS = 700               # chars of each clause's resolved_text
GEN_CFG = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.5,
                                      max_output_tokens=8192)


# ════════════════════════════════════════════════════════════════════════════════
# Context building
# ════════════════════════════════════════════════════════════════════════════════

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _sample(objs: list[dict], k: int) -> list[dict]:
    """Keep all if <= k, else evenly sample across the (date-sorted) list, always
    keeping first and last so the temporal span is preserved."""
    if len(objs) <= k:
        return objs
    idxs = sorted({round(i * (len(objs) - 1) / (k - 1)) for i in range(k)})
    return [objs[i] for i in idxs]


def build_context(thread: dict) -> str:
    parts = [f'THREAD TITLE: {thread.get("title","")}',
             f'CONCEPT (entity cluster): {thread.get("cluster_label","")}',
             f'SUMMARY: {_clean(thread.get("summary",""))}', ""]
    evo = thread.get("evolution") or {}
    if evo:
        parts.append("EVOLUTION DIMENSIONS:")
        for dim, desc in evo.items():
            parts.append(f"  - {dim}: {_clean(desc)}")
        parts.append("")
    tl = thread.get("timeline") or []
    if tl:
        parts.append("TIMELINE:")
        for t in tl:
            parts.append(f'  {t.get("date","?")} | {t.get("instrument","?")} | {_clean(t.get("change",""))}')
        parts.append("")
    members = thread.get("member_objects") or []
    members = sorted(members, key=lambda o: (o.get("clause_date", "undated"), o.get("text_id", "")))
    shown = _sample(members, MAX_CLAUSES)
    parts.append(f"CLAUSES (chronological, {len(shown)} of {len(members)} shown):")
    for o in shown:
        ents = ", ".join((o.get("entity_concepts") or {}).keys())
        concepts = []
        for cons in (o.get("entity_concepts") or {}).values():
            concepts.extend(cons or [])
        subs = sorted({s for v in (o.get("entity_sub_entities") or {}).values() for s in (v or [])})
        parts.append(
            f'  [{o.get("clause_date","?")}] src={o.get("source_file","").split("/")[-1][:40]}\n'
            f'     entities: {ents}\n'
            f'     concepts: {" | ".join(_clean(c) for c in concepts[:4])}\n'
            f'     sub-entities: {", ".join(subs[:10])}\n'
            f'     text: {_clean(o.get("resolved_text",""))[:CLAUSE_CHARS]}')
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════════
# Per-type prompts
# ════════════════════════════════════════════════════════════════════════════════

_BASE = """You are creating a Q&A dataset from the evolution of a Bangladeshi customs/tax
policy concept over time. Use ONLY the facts in the CONTEXT. Every question and
answer must be grounded in it; never invent dates, rates, instruments or entities.
Generate AS MANY high-quality, DISTINCT pairs as the content genuinely supports
(aim for {target} where possible; fewer if the content is thin). Each answer must
be self-contained and cite the relevant date/instrument when applicable.

CONTEXT:
{context}

"""

TYPE_SPECS = {
    "scenario_short_answer": (6, """Produce TYPE-1 questions: realistic REAL-WORLD SCENARIOS (an importer, manufacturer,
customs officer, or passenger in a specific year/situation) with a SHORT factual answer
about what rule/rate/requirement applied at that time.
Return STRICT JSON:
{{"qa_pairs": [{{"type": "scenario_short_answer", "scenario": "<the situation>",
  "question": "<question>", "answer": "<short factual answer>",
  "date_or_instrument": "<relevant date/SRO>"}}]}}"""),

    "temporal_mcq": (6, """Produce TYPE-2 questions: TEMPORAL MULTIPLE-CHOICE questions about WHEN something
changed, WHAT was true at a given time, or WHICH instrument introduced a change.
Exactly 4 options, one correct, plausible distractors.
Return STRICT JSON:
{{"qa_pairs": [{{"type": "temporal_mcq", "question": "<question>",
  "options": ["<A>","<B>","<C>","<D>"], "answer_index": <0-3>,
  "answer": "<correct option text>", "explanation": "<why, with date/instrument>"}}]}}"""),

    "event_sorting": (4, """Produce TYPE-3 questions: EVENT-SORTING. Give a SHUFFLED set of real events/changes
from the timeline and ask to order them chronologically (earliest first).
Each event must be a real change from the context. Use 3-6 events per question.
Return STRICT JSON:
{{"qa_pairs": [{{"type": "event_sorting",
  "question": "Arrange these events in chronological order (earliest first).",
  "events_shuffled": ["<event a>","<event b>","<event c>"],
  "answer_ordered": [{{"event": "<event>", "date": "<YYYY-MM-DD or year>"}}]}}]}}"""),

    "open_ended": (6, """Produce TYPE-4 questions: OPEN-ENDED analytical questions about HOW or WHY the concept
evolved, comparisons across periods, drivers, or implications. Answers are a few
grounded sentences.
Return STRICT JSON:
{{"qa_pairs": [{{"type": "open_ended", "question": "<analytical question>",
  "answer": "<few-sentence grounded answer>"}}]}}"""),

    "perturbation_detection": (6, """Produce TYPE-5 questions: PERTURBATION DETECTION. Write a statement about the thread
in which exactly ONE fact is subtly WRONG (a changed date, rate, category, instrument,
or scope). Ask the reader to judge accuracy and correct it. Most statements should be
perturbed; you may include some that are fully correct.
Return STRICT JSON:
{{"qa_pairs": [{{"type": "perturbation_detection", "statement": "<statement, usually with one wrong fact>",
  "question": "Is this statement accurate? If not, identify and correct the error.",
  "is_perturbed": <true|false>, "perturbed_fact": "<the wrong part, or null>",
  "correct_fact": "<the correct fact>", "answer": "<verdict + correction>"}}]}}"""),

    "context_misalignment": (5, """Produce TYPE-6 questions: CONTEXT-MISALIGNMENT CONFLICT RESOLUTION. Provide a short
MISALIGNED context passage that mixes in a detail from the WRONG time period or that
contradicts the true evolution, then ask a question whose correct answer requires
RESOLVING the conflict using the true chronology/facts.
Return STRICT JSON:
{{"qa_pairs": [{{"type": "context_misalignment", "misaligned_context": "<passage with a temporal/factual conflict>",
  "question": "<question requiring conflict resolution>",
  "conflict": "<what is misaligned and why>",
  "answer": "<resolution citing the correct date/instrument/fact>"}}]}}"""),
}


# ════════════════════════════════════════════════════════════════════════════════
# LLM call + parse
# ════════════════════════════════════════════════════════════════════════════════

def _call(pool: KeyPool, prompt: str) -> str | None:
    for _ in range(6):
        k = pool.acquire()
        if k is None:
            return None
        try:
            resp = _call_with_watchdog(
                lambda: pool.client(k).models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=GEN_CFG), WATCHDOG)
            pool.release(k)
            return resp.text
        except Exception as e:  # noqa: BLE001
            etype = _classify_err(e)
            if etype == "permanent":
                pool.release(k, dead=True)
            elif etype == "rate_limit":
                pool.release(k, cooldown=max(ApiKeyRotator._extract_retry_delay(e), 30))
            else:
                pool.release(k, cooldown=3)
    return None


def _salvage_objects(text: str) -> list[dict]:
    """Recover complete top-level {...} objects from a (possibly truncated) JSON
    string by scanning balanced braces with string/escape awareness. Lets us keep
    every fully-formed qa_pair even when the response was cut off mid-array."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        out.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return out


def _parse_pairs(text: str, qtype: str) -> list[dict]:
    if not text:
        return []
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

    items = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("qa_pairs"), list):
            items = data["qa_pairs"]
        elif isinstance(data, list):
            items = data
    except json.JSONDecodeError:
        items = []
    if not items:
        # truncated / malformed → salvage every complete object, drop the outer wrapper
        salvaged = _salvage_objects(raw)
        items = [o for o in salvaged if "qa_pairs" not in o]

    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not (item.get("question") or item.get("statement") or item.get("misaligned_context")):
            continue
        item["type"] = qtype           # enforce the type key
        out.append(item)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Per-thread generation
# ════════════════════════════════════════════════════════════════════════════════

def generate_for_thread(pool: KeyPool, thread: dict) -> dict:
    context = build_context(thread)
    qa: list[dict] = []
    per_type: dict[str, int] = {}
    for qtype, (target, spec) in TYPE_SPECS.items():
        prompt = _BASE.format(target=target, context=context) + spec
        pairs = _parse_pairs(_call(pool, prompt), qtype)
        per_type[qtype] = len(pairs)
        qa.extend(pairs)
    return {
        "thread_id": thread.get("thread_id", ""),
        "cluster_id": thread.get("cluster_id"),
        "cluster_label": thread.get("cluster_label", ""),
        "title": thread.get("title", ""),
        "source_num_member_objects": thread.get("num_member_objects"),
        "num_qa_pairs": len(qa),
        "qa_counts_by_type": per_type,
        "qa_pairs": qa,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default=None, help="thread_id substring filter")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY,
                    help="parallel workers (use a low value to repair large stragglers)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging(False)

    files = sorted(f for f in THREADS_DIR.glob("cluster_*.json"))
    work = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("num_member_objects", 0) <= 1:
            continue
        if args.only and args.only not in f.stem:
            continue
        work.append((f, d))
    if args.limit:
        work = work[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def out_path(f: Path) -> Path:
        return OUT_DIR / f"{f.stem}_QA.json"

    todo = [(f, d) for f, d in work if args.force or not out_path(f).exists()]
    logger.info(f"{len(work)} qualifying threads; {len(todo)} to generate "
                f"({len(work)-len(todo)} already done)")

    pool = KeyPool(_build_judge_rotator().keys)
    lock = threading.Lock()
    totals = {"files": 0, "pairs": 0}

    def task(item):
        f, d = item
        res = generate_for_thread(pool, d)
        out_path(f).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return res["num_qa_pairs"]

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(task, it) for it in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="qa", unit="thread"):
                npairs = fut.result()
                with lock:
                    totals["files"] += 1
                    totals["pairs"] += npairs

    logger.info(f"Wrote {totals['files']} QA files, {totals['pairs']} Q&A pairs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
