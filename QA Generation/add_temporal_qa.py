#!/usr/bin/env python3
"""
add_temporal_qa.py
════════════════════════════════════════════════════════════════════════════════
Add a new Q&A type — "temporal_scenario_short_answer" — to the existing
concept_qa/*_QA.json files.

Unlike the plain scenario type (one clause, one point in time), this type requires:
  • a SCENARIO that depends on MULTIPLE related clauses at DIFFERENT times, and
  • a QUESTION that needs TEMPORAL reasoning (what changed, before-vs-after,
    what applies now given an earlier rule, sequence/consequence of instruments).

Not forced: the model only emits pairs that are factually sound and genuinely need
≥2 related clauses across time. Threads that can't support it get none.

The new pairs are APPENDED to each file's qa_pairs (existing pairs untouched);
num_qa_pairs and qa_counts_by_type are updated. Idempotent (existing temporal pairs
are replaced, not duplicated) and resumable via a sidecar done-list.

Grounding comes from the matching concept_threads/<stem>.json (timeline, summary,
evolution, full member clauses). Keys/model: free keys (all_api.txt) + gemini-3.1-flash-lite.

Usage:
    python add_temporal_qa.py
    python add_temporal_qa.py --limit 3
    python add_temporal_qa.py --only cluster_00326_t0
    python add_temporal_qa.py --concurrency 5
    python add_temporal_qa.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from google.genai import types

from break_subentities import GEMINI_MODEL, _call_with_watchdog, setup_logging
from cluster_entities import ApiKeyRotator, KeyPool, _build_judge_rotator, _classify_err
# reuse the robust parser from the QA generator
from generate_qa import _parse_pairs, _clean

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa"
THREADS_DIR = SCRIPT_DIR / "concept_threads"
DONE_FILE = SCRIPT_DIR / "temporal_qa_done.json"

QTYPE = "temporal_scenario_short_answer"
TARGET = 4
CONCURRENCY = 2                  # low: large prompts hit per-minute TOKEN (TPM) limits

# Lean generation config — temporal answers are short, so a small output cap keeps
# the per-call token footprint (and thus TPM pressure) low.
LEAN_CFG = types.GenerateContentConfig(response_mime_type="application/json",
                                       temperature=0.6, max_output_tokens=2048)
LEAN_CLAUSES = 10                # clauses shown in the lean context
LEAN_CLAUSE_CHARS = 320

PROMPT = """You are creating a TEMPORAL scenario Q&A set from how a Bangladeshi customs/tax
concept evolves over time. Use ONLY facts in the CONTEXT — never invent dates,
rates, instruments, categories or entities.

Produce questions of type "{qtype}" where BOTH hold:
  • the SCENARIO is grounded in and DEPENDS ON MULTIPLE related clauses at DIFFERENT
    points in time (e.g. an importer/manufacturer/officer whose situation is affected
    by an earlier rule and a later change to it), and
  • the QUESTION requires TEMPORAL UNDERSTANDING across those clauses — comparing
    before vs after, what changed, what now applies to someone who started under an
    older rule, or the sequence/consequence of successive instruments.
The ANSWER is short and factual, grounded in the evolution across the referenced clauses.

DO NOT FORCE IT. Only emit a pair if it is factually sound and genuinely needs
reasoning over >=2 related clauses at different times. If the thread lacks enough
distinct temporal steps to support this honestly, return an EMPTY list. A short list
of strong pairs is better than a padded one. Do not reveal the answer inside the
scenario, and do not invent specific figures beyond what the context states.

CONTEXT:
{context}

Return STRICT JSON (aim for up to {target} strong pairs, or fewer / none):
{{"qa_pairs": [{{"type": "{qtype}",
  "scenario": "<real-world story spanning multiple time points>",
  "question": "<question requiring temporal reasoning across the clauses>",
  "answer": "<short factual answer grounded in the evolution>",
  "clauses_involved": ["<date/instrument 1>", "<date/instrument 2>", "..."]}}]}}
Return only the JSON object."""


def build_lean_context(thread: dict) -> str:
    """Compact, high-signal context for temporal reasoning: summary + evolution +
    full timeline (the core temporal signal) + a few short clause excerpts. Much
    smaller than the full QA context, which keeps per-call token use under the TPM cap."""
    parts = [f'THREAD: {thread.get("title","")}  (concept: {thread.get("cluster_label","")})',
             f'SUMMARY: {_clean(thread.get("summary",""))}', ""]
    evo = thread.get("evolution") or {}
    if evo:
        parts.append("EVOLUTION:")
        for dim, desc in evo.items():
            parts.append(f"  - {dim}: {_clean(desc)}")
        parts.append("")
    tl = thread.get("timeline") or []
    if tl:
        parts.append("TIMELINE:")
        for t in tl:
            parts.append(f'  {t.get("date","?")} | {t.get("instrument","?")} | {_clean(t.get("change",""))}')
        parts.append("")
    members = sorted(thread.get("member_objects") or [],
                     key=lambda o: (o.get("clause_date", "undated"), o.get("text_id", "")))
    # evenly sample up to LEAN_CLAUSES across the timeline to preserve the span
    if len(members) > LEAN_CLAUSES:
        idxs = sorted({round(i * (len(members) - 1) / (LEAN_CLAUSES - 1)) for i in range(LEAN_CLAUSES)})
        members = [members[i] for i in idxs]
    parts.append("KEY CLAUSES (chronological):")
    for o in members:
        concepts = [c for cons in (o.get("entity_concepts") or {}).values() for c in (cons or [])]
        parts.append(f'  [{o.get("clause_date","?")}] {_clean(" | ".join(concepts[:2]))[:LEAN_CLAUSE_CHARS]}'
                     f' :: {_clean(o.get("resolved_text",""))[:LEAN_CLAUSE_CHARS]}')
    return "\n".join(parts)


def _patient_call(pool: KeyPool, prompt: str, retries: int = 20) -> str | None:
    """Very patient call — KeyPool.acquire() blocks on cooldowns, so extra retries
    ride out the free-tier per-minute (RPM/TPM) throttling that failed whole files."""
    for _ in range(retries):
        k = pool.acquire()
        if k is None:
            return None
        try:
            resp = _call_with_watchdog(
                lambda: pool.client(k).models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=LEAN_CFG), 90)
            pool.release(k)
            return resp.text
        except Exception as e:  # noqa: BLE001
            etype = _classify_err(e)
            if etype == "permanent":
                pool.release(k, dead=True)
            elif etype == "rate_limit":
                pool.release(k, cooldown=max(ApiKeyRotator._extract_retry_delay(e), 5))
            else:
                pool.release(k, cooldown=2)
    return None


def process_file(pool: KeyPool, qa_path: Path) -> int:
    stem = qa_path.name[:-len("_QA.json")]
    thread_path = THREADS_DIR / f"{stem}.json"
    if not thread_path.exists():
        return 0
    thread = json.loads(thread_path.read_text(encoding="utf-8"))
    context = build_lean_context(thread)
    prompt = PROMPT.format(qtype=QTYPE, target=TARGET, context=context)
    raw = _patient_call(pool, prompt)
    if raw is None:
        return -1                      # call failed (throttled/dead) → retry later, don't mark done
    pairs = _parse_pairs(raw, QTYPE)

    d = json.loads(qa_path.read_text(encoding="utf-8"))
    # idempotent: drop any previously-added temporal pairs, then append fresh ones
    base = [q for q in d.get("qa_pairs", []) if q.get("type") != QTYPE]
    d["qa_pairs"] = base + pairs
    d["num_qa_pairs"] = len(d["qa_pairs"])
    counts = d.get("qa_counts_by_type", {})
    counts[QTYPE] = len(pairs)
    d["qa_counts_by_type"] = counts
    qa_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--paid", action="store_true",
                    help="use the paid key (gemini_api.txt) instead of the free keys")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging(False)

    done = set()
    if DONE_FILE.exists() and not args.force:
        done = set(json.loads(DONE_FILE.read_text()))

    files = sorted(QA_DIR.glob("*_QA.json"))
    todo = [f for f in files if f.name not in done and (not args.only or args.only in f.name)]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"{len(files)} QA files; {len(todo)} to process ({len(files)-len(todo)} done)")

    if args.paid:
        paid = [k.strip() for k in re.split(r"[,\n]+",
                (SCRIPT_DIR / "gemini_api.txt").read_text()) if k.strip()]
        pool = KeyPool(paid)
        logger.info(f"using PAID key(s): {len(paid)}")
    else:
        pool = KeyPool(_build_judge_rotator().keys)
    lock = threading.Lock()
    total = 0
    made = 0
    failed = 0

    def task(f: Path):
        return f.name, process_file(pool, f)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(task, f) for f in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="temporal", unit="file"):
                name, n = fut.result()
                with lock:
                    if n < 0:              # call failed → leave out of done-list for retry
                        failed += 1
                        continue
                    total += n
                    made += 1 if n else 0
                    done.add(name)
                    if len(done) % 20 == 0:
                        DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    logger.info(f"added {total} temporal pairs across {made} files; "
                f"{failed} call-failures (will retry on re-run); "
                f"rest had none — not warranted")


if __name__ == "__main__":
    main()
