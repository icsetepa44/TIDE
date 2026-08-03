#!/usr/bin/env python3
"""
regenerate_qa.py
════════════════════════════════════════════════════════════════════════════════
Regenerate the ENTIRE Q&A dataset (all 9 types) from concept_threads/, with:
  • a better model (default gemini-3.5-flash) on the PAID key,
  • improved, quality-focused prompts,
  • per-pair clause GROUNDING cited by the model → each pair carries the EXACT
    `clause_text_ids`, and TWO source-file fields:
       - `thread_source_file`  — the concept_threads/ file the Q&A was generated from,
       - `clause_source_files` — the real legal source file(s) each cited clause comes
         from, MAPPED from merged_ec_output_canon_subentities.json (the authoritative
         text_id → document.source_file index, disambiguated by the member's own
         source_file for the few non-unique `metadata_N` ids).

Types (8 per-thread + 1 cross-thread):
  scenario_short_answer (story-style), temporal_mcq, event_sorting, open_ended,
  perturbation_detection, context_misalignment, temporal_scenario_short_answer,
  relative_time_qa,  and  multi_entity_concept_qa (across clusters).

Per-thread files → concept_qa/ ; cross-thread files → concept_qa_multi/. The
previous dataset is backed up to concept_qa_v1/ and concept_qa_multi_v1/ once.

Efficiency / no-loop: one paid client, bounded exponential backoff on 503/429
(capped retries — will not loop forever), a hard per-call watchdog, resumable via
per-file presence so re-runs only do what's missing.

Usage:
    python regenerate_qa.py --model gemini-3.5-flash            # full run
    python regenerate_qa.py --only cluster_00326_t0            # one thread
    python regenerate_qa.py --limit 3
    python regenerate_qa.py --multi-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from break_subentities import _call_with_watchdog, setup_logging

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
THREADS_DIR = SCRIPT_DIR / "concept_threads"
QA_DIR = SCRIPT_DIR / "concept_qa"
MULTI_DIR = SCRIPT_DIR / "concept_qa_multi"
PAID_KEYS = SCRIPT_DIR / "gemini_api.txt"
MAIN_FILE = SCRIPT_DIR / "merged_ec_output_canon_subentities.json"

DEFAULT_MODEL = "gemini-3.5-flash"
CONCURRENCY = 5
# 503s are billed as NOTHING (paid credit = generated tokens only), so we retry them
# patiently — riding through "high demand" spikes on the flash tier until served.
# A generous cap keeps a truly-dead model from spinning forever.
RETRY_503_CAP = 400
MAX_OTHER_ERRORS = 5
WATCHDOG = 120
CLAUSE_CHARS = 650
MAX_CLAUSES = 24
CFG = types.GenerateContentConfig(response_mime_type="application/json",
                                  temperature=0.6, max_output_tokens=8192)

# ════════════════════════════════════════════════════════════════════════════════
# Paid caller (single key, concurrent, bounded backoff)
# ════════════════════════════════════════════════════════════════════════════════

_CLIENT = None
_MODEL = DEFAULT_MODEL


def _client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        key = [k.strip() for k in re.split(r"[,\n]+", PAID_KEYS.read_text()) if k.strip()][0]
        _CLIENT = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=WATCHDOG * 1000))
    return _CLIENT


def _call(prompt: str) -> str | None:
    """Patient caller: transient 503/429 are retried up to RETRY_503_CAP with capped
    backoff (they cost no credit); other errors give up quickly."""
    delay, transient_tries, other = 2, 0, 0
    while True:
        try:
            resp = _call_with_watchdog(
                lambda: _client().models.generate_content(model=_MODEL, contents=prompt, config=CFG),
                WATCHDOG)
            return resp.text
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            transient = ("503" in msg or "unavailable" in msg or "429" in msg
                         or "resource" in msg or "exhaust" in msg or "overloaded" in msg
                         or "high demand" in msg or "timeout" in msg)
            if transient:
                transient_tries += 1
                if transient_tries >= RETRY_503_CAP:
                    logger.warning(f"giving up after {transient_tries} transient retries on {_MODEL}")
                    return None
                if transient_tries % 25 == 0:
                    logger.info(f"waiting on {_MODEL}: {transient_tries} transient retries (503/high-demand)")
                time.sleep(delay)
                delay = min(delay * 2, 20)
                continue
            if "permission" in msg or "denied" in msg or "invalid" in msg or "not found" in msg:
                logger.warning(f"fatal call error on {_MODEL}: {str(e)[:160]}")
                return None
            other += 1
            if other >= MAX_OTHER_ERRORS:
                return None
            time.sleep(min(2 ** other, 15))


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
    except json.JSONDecodeError:
        items = _salvage(raw)
    if not items:
        items = [o for o in _salvage(raw) if "qa_pairs" not in o]
    out = []
    for it in items:
        if isinstance(it, dict) and (it.get("question") or it.get("statement")
                                     or it.get("misaligned_context")):
            it["type"] = qtype
            out.append(it)
    return out


def _salvage(text: str) -> list[dict]:
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = (ch == "\\" and not esc)
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    out.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1
    return out


def _clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


# ════════════════════════════════════════════════════════════════════════════════
# Context (numbered clauses carry text_id + source_file so the model can cite them)
# ════════════════════════════════════════════════════════════════════════════════

def _sample(objs, k):
    if len(objs) <= k:
        return objs
    idx = sorted({round(i * (len(objs) - 1) / (k - 1)) for i in range(k)})
    return [objs[i] for i in idx]


def _thread_clauses(thread):
    members = sorted(thread.get("member_objects") or [],
                     key=lambda o: (o.get("clause_date", "undated"), o.get("text_id", "")))
    return _sample(members, MAX_CLAUSES)


def _clause_lines(clauses):
    lines = []
    for i, o in enumerate(clauses):
        ents = ", ".join((o.get("entity_concepts") or {}).keys())
        concepts = [c for cons in (o.get("entity_concepts") or {}).values() for c in (cons or [])]
        src = (o.get("source_file", "") or "").split("/")[-1]
        lines.append(f'[{i}] date={o.get("clause_date","?")} source={src}\n'
                     f'    entities: {ents}\n'
                     f'    concepts: {_clean(" | ".join(concepts[:3]))[:300]}\n'
                     f'    text: {_clean(o.get("resolved_text",""))[:CLAUSE_CHARS]}')
    return "\n".join(lines)


def build_context(thread):
    parts = [f'THREAD: {thread.get("title","")}  (concept: {thread.get("cluster_label","")})',
             f'SUMMARY: {_clean(thread.get("summary",""))}', ""]
    evo = thread.get("evolution") or {}
    if evo:
        parts.append("EVOLUTION:")
        parts += [f"  - {d}: {_clean(v)}" for d, v in evo.items()]
        parts.append("")
    tl = thread.get("timeline") or []
    if tl:
        parts.append("TIMELINE:")
        parts += [f'  {t.get("date","?")} | {t.get("instrument","?")} | {_clean(t.get("change",""))}'
                  for t in tl]
        parts.append("")
    parts.append("NUMBERED CLAUSES (each has a source file; cite the ones you use):")
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════════
# Prompts (improved) — every type must return clause_indices it is grounded in
# ════════════════════════════════════════════════════════════════════════════════

_HEAD = """You build a HIGH-QUALITY, exam-grade Q&A dataset from how a Bangladeshi customs/tax
concept evolves over time. Use ONLY the facts in the context below — never invent
dates, rates, instruments, categories, or entities.

Quality bar for EVERY pair:
  • SPECIFIC & grounded — name the exact date/year, SRO/instrument number, rate,
    category/HS heading, or entity involved; never vague or generic.
  • SELF-CONTAINED — fully answerable from the cited clauses, no outside knowledge.
  • NON-TRIVIAL & DISTINCT — no two pairs test the same fact; favour pairs that
    require reading the evolution/timeline over a single surface fact.
  • FAITHFUL — the answer must be exactly supported by the cited clauses.
Generate AS MANY strong, distinct pairs as the content genuinely supports ({target}
is a guide; return fewer if the material is thin — never pad with weak pairs).

CRITICAL: for every pair include "clause_indices" — the list of clause [index]
numbers (from the NUMBERED CLAUSES) whose content the question/answer is grounded in.
Cite the exact clauses used; never leave it empty.

{context}
{clauses}

"""

SPECS = {
    "scenario_short_answer": (6, """Produce TYPE scenario_short_answer: a realistic, STORY-LIKE real-world SCENARIO
(3-6 sentences: a named persona/firm/officer in a concrete situation and year) that
naturally leads to a question, with a SHORT factual answer about the rule/rate/
requirement that applied. Do not reveal the answer inside the scenario.
JSON: {{"qa_pairs":[{{"type":"scenario_short_answer","scenario":"<story>","question":"<q>",
"answer":"<short factual answer>","date_or_instrument":"<relevant date/SRO>","clause_indices":[<int>]}}]}}"""),

    "temporal_mcq": (6, """Produce TYPE temporal_mcq: TEMPORAL multiple-choice (when did X change / what held at
time T / which instrument introduced Y). 4 options, one correct, plausible distractors.
JSON: {{"qa_pairs":[{{"type":"temporal_mcq","question":"<q>","options":["A","B","C","D"],
"answer_index":<0-3>,"answer":"<correct option>","explanation":"<why, with date/instrument>","clause_indices":[<int>]}}]}}"""),

    "event_sorting": (4, """Produce TYPE event_sorting: give a SHUFFLED set of 3-6 real events/changes and ask to
order them chronologically (earliest first).
JSON: {{"qa_pairs":[{{"type":"event_sorting","question":"Arrange these events in chronological order (earliest first).",
"events_shuffled":["e1","e2","e3"],"answer_ordered":[{{"event":"e","date":"YYYY-MM-DD"}}],"clause_indices":[<int>]}}]}}"""),

    "open_ended": (6, """Produce TYPE open_ended: analytical questions on HOW/WHY the concept evolved, cross-
period comparisons, drivers, or implications; answers a few grounded sentences.
JSON: {{"qa_pairs":[{{"type":"open_ended","question":"<analytical q>","answer":"<grounded answer>","clause_indices":[<int>]}}]}}"""),

    "perturbation_detection": (6, """Produce TYPE perturbation_detection: a statement in which exactly ONE fact is subtly
WRONG (changed date/rate/category/instrument/scope); ask the reader to judge accuracy
and correct it. Most perturbed; a few may be fully correct.
JSON: {{"qa_pairs":[{{"type":"perturbation_detection","statement":"<usually one wrong fact>",
"question":"Is this statement accurate? If not, identify and correct the error.","is_perturbed":<true|false>,
"perturbed_fact":"<wrong part or null>","correct_fact":"<correct fact>","answer":"<verdict + correction>","clause_indices":[<int>]}}]}}"""),

    "context_misalignment": (5, """Produce TYPE context_misalignment: give a short MISALIGNED context passage that mixes in
a detail from the WRONG time period or contradicts the true evolution, then a question
whose correct answer requires RESOLVING the conflict using the true chronology.
JSON: {{"qa_pairs":[{{"type":"context_misalignment","misaligned_context":"<passage with a conflict>",
"question":"<q needing conflict resolution>","conflict":"<what is misaligned>","answer":"<resolution citing correct date/instrument>","clause_indices":[<int>]}}]}}"""),

    "temporal_scenario_short_answer": (5, """Produce TYPE temporal_scenario_short_answer: a scenario that DEPENDS ON MULTIPLE related
clauses at DIFFERENT times, with a question requiring TEMPORAL reasoning (before vs after,
what now applies to someone who started under an older rule, sequence/consequence). Only
if it genuinely needs >=2 clauses across time; else fewer/none.
JSON: {{"qa_pairs":[{{"type":"temporal_scenario_short_answer","scenario":"<multi-time story>",
"question":"<temporal-reasoning q>","answer":"<short factual answer>","clauses_involved":["<date/instrument>"],"clause_indices":[<int>]}}]}}"""),

    "relative_time_qa": (4, """Produce TYPE relative_time_qa: state an explicit SEED time, then ask about a different
point via a RELATIVE offset ("six years later","two years earlier") WITHOUT naming the
target year in the question; the target must be a real timeline event; the answer resolves
it to the actual time/instrument.
JSON: {{"qa_pairs":[{{"type":"relative_time_qa","seed_time":"<seed>","relative_offset":"<e.g. 'six years later'>",
"question":"<uses seed + relative offset>","answer":"<resolves to actual time + fact>","clause_indices":[<int>]}}]}}"""),
}


# ── authoritative clause → source-file map from the main merged file ──────────────
# In the main file the real source lives at DOCUMENT level (clause-level source_file is
# null), and a few `metadata_N` text_ids are reused across documents. So we index
# text_id → {document.source_file}; when that set is ambiguous we disambiguate with the
# thread member's OWN source_file (verified to always be a member of the set).
_CLAUSE_SRC_IDX = None


def _clause_src_index() -> dict:
    global _CLAUSE_SRC_IDX
    if _CLAUSE_SRC_IDX is None:
        idx = defaultdict(set)
        d = json.loads(MAIN_FILE.read_text(encoding="utf-8"))
        for doc in d.get("merged_documents", []):
            src = doc.get("source_file")
            if not src:
                continue
            for c in doc.get("processed_clauses", []):
                tid = c.get("text_id")
                if tid:
                    idx[tid].add(src)
        _CLAUSE_SRC_IDX = idx
    return _CLAUSE_SRC_IDX


def _clause_source(text_id: str, own_src: str) -> str:
    """Map a clause text_id to its legal source file via the main file, disambiguating
    the rare non-unique ids with the member's own source_file."""
    srcs = _clause_src_index().get(text_id) or set()
    if len(srcs) == 1:
        return next(iter(srcs))
    if own_src and own_src in srcs:      # ambiguous → member's own source resolves it
        return own_src
    if srcs:
        return sorted(srcs)[0]
    return own_src or ""                 # not in main (shouldn't happen) → member's own


def _resolve(pairs, clauses, thread_src):
    """clause_indices -> exact clause_text_ids (tracked from the cited clauses), then the
    clause source file(s) MAPPED from the main file. Also stamps thread_source_file."""
    for p in pairs:
        idxs = p.pop("clause_indices", []) if isinstance(p.get("clause_indices"), list) else []
        chosen = []
        for x in idxs:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= xi < len(clauses) and clauses[xi] not in chosen:
                chosen.append(clauses[xi])
        if not chosen:                     # grounded in the whole thread
            chosen = list(clauses)
        tids, csrcs = [], []
        for o in chosen:
            tid = o.get("text_id", "")
            if tid and tid not in tids:
                tids.append(tid)
                cs = _clause_source(tid, o.get("source_file", ""))
                if cs:
                    csrcs.append(cs)
        p["clause_text_ids"] = tids
        p["clause_source_files"] = sorted(set(csrcs))
        p["thread_source_file"] = thread_src
    return pairs


def gen_thread(name, thread) -> dict:
    clauses = _thread_clauses(thread)
    ctx = build_context(thread)
    cl = _clause_lines(clauses)
    qa_all, counts = [], {}
    for qtype, (target, spec) in SPECS.items():
        prompt = _HEAD.format(target=target, context=ctx, clauses=cl) + spec
        pairs = _resolve(_parse_pairs(_call(prompt), qtype), clauses, name)
        counts[qtype] = len(pairs)
        qa_all.extend(pairs)
    return {
        "thread_id": thread.get("thread_id", ""), "cluster_id": thread.get("cluster_id"),
        "cluster_label": thread.get("cluster_label", ""), "title": thread.get("title", ""),
        "thread_source_file": name,
        "source_num_member_objects": thread.get("num_member_objects"),
        "num_qa_pairs": len(qa_all), "qa_counts_by_type": counts, "qa_pairs": qa_all,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Multi-entity (cross-thread)
# ════════════════════════════════════════════════════════════════════════════════

MULTI_PROMPT = """You build CROSS-CONCEPT Q&A from Bangladeshi customs/tax law. Two different concepts
share the clauses below (the common ground). Use ONLY these facts.

Produce questions of type "multi_entity_concept_qa" that meaningfully involve BOTH
Concept A and Concept B together on the common ground; answer grounded strictly in the
shared clauses. Only emit sound pairs. Cite the shared-clause [index]es used.

Concept A: "{a}" — {sa}
Concept B: "{b}" — {sb}

SHARED CLAUSES (each has a source file):
{clauses}

JSON (up to {target}, or fewer/none):
{{"qa_pairs":[{{"type":"multi_entity_concept_qa","question":"<needs both concepts>",
"answer":"<grounded answer>","entities_involved":["{a}","{b}"],"clause_indices":[<int>]}}]}}
Return only the JSON object."""


def gen_multi(a_name, b_name, a_meta, b_meta, shared_objs) -> list[dict]:
    shared = sorted(shared_objs, key=lambda o: o.get("text_id", ""))[:12]
    lines = []
    for i, o in enumerate(shared):
        src = (o.get("source_file", "") or "").split("/")[-1]
        lines.append(f'[{i}] date={o.get("clause_date","?")} source={src}\n'
                     f'    {_clean(o.get("resolved_text",""))[:CLAUSE_CHARS]}')
    prompt = MULTI_PROMPT.format(a=a_meta["label"], b=b_meta["label"],
                                 sa=a_meta["summary"][:400], sb=b_meta["summary"][:400],
                                 clauses="\n".join(lines), target=4)
    pairs = _resolve(_parse_pairs(_call(prompt), "multi_entity_concept_qa"), shared, None)
    for p in pairs:                        # cross-thread → two originating threads
        p.pop("thread_source_file", None)
        p["thread_source_files"] = [a_name, b_name]
    return pairs


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def _load_threads():
    out = {}
    for f in THREADS_DIR.glob("cluster_*.json"):
        if f.name.endswith("_index.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("num_member_objects", 0) > 1:
            out[f.name] = d
    return out


def _multi_groups(threads):
    tid2 = defaultdict(set)
    for name, d in threads.items():
        for m in d.get("member_objects", []):
            if m.get("text_id"):
                tid2[m["text_id"]].add(name)
    pair = defaultdict(list)
    for tid, names in tid2.items():
        if len(names) < 2:
            continue
        for a, b in combinations(sorted(names), 2):
            if threads[a]["cluster_id"] != threads[b]["cluster_id"]:
                pair[(a, b)].append(tid)
    cand = sorted(((a, b, ids) for (a, b), ids in pair.items() if len(ids) >= 3),
                  key=lambda x: -len(x[2]))
    used = defaultdict(int)
    groups = []
    for a, b, ids in cand:
        if used[a] >= 3 or used[b] >= 3:
            continue
        groups.append((a, b, ids))
        used[a] += 1
        used[b] += 1
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--multi-only", action="store_true")
    ap.add_argument("--threads-only", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    setup_logging(False)
    global _MODEL
    _MODEL = args.model
    logger.info(f"regenerating with model={_MODEL} on PAID key, concurrency={args.concurrency}")

    threads = _load_threads()

    # one-time backup of the v1 dataset
    if not args.no_backup and not args.only and not args.limit:
        for src, dst in [(QA_DIR, SCRIPT_DIR / "concept_qa_v1"),
                         (MULTI_DIR, SCRIPT_DIR / "concept_qa_multi_v1")]:
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
                logger.info(f"backed up {src.name} → {dst.name}")

    QA_DIR.mkdir(exist_ok=True)
    MULTI_DIR.mkdir(exist_ok=True)
    lock = threading.Lock()

    # ---- per-thread types ----
    if not args.multi_only:
        names = sorted(threads)
        if args.only:
            names = [n for n in names if args.only in n]
        todo = [n for n in names
                if args.only or args.limit or not (QA_DIR / f"{n[:-5]}_QA.json").exists()]
        if args.limit:
            todo = todo[:args.limit]
        logger.info(f"per-thread: {len(todo)} threads to generate")
        done = {"n": 0, "pairs": 0}

        def ttask(name):
            res = gen_thread(name, threads[name])
            # 0 pairs almost always means every call failed (503/high-demand) — don't
            # write an empty file (it would be marked done and skipped on resume).
            if res["num_qa_pairs"] == 0:
                return 0
            (QA_DIR / f"{name[:-5]}_QA.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            return res["num_qa_pairs"]

        if todo:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = [ex.submit(ttask, n) for n in todo]
                empties = 0
                for fut in tqdm(as_completed(futs), total=len(futs), desc="threads", unit="thr"):
                    n_pairs = fut.result()
                    with lock:
                        done["pairs"] += n_pairs
                        done["n"] += 1
                        empties += 1 if n_pairs == 0 else 0
                if empties:
                    logger.info(f"{empties} threads produced 0 pairs (deferred — re-run to retry)")
        logger.info(f"per-thread done: {done['n']} threads, {done['pairs']} pairs")

    # ---- cross-thread multi-entity ----
    if not args.threads_only:
        groups = _multi_groups(threads)
        if args.limit:
            groups = groups[:args.limit]
        meta = {n: {"label": d["cluster_label"], "summary": _clean(d.get("summary", "")),
                    "cluster_id": d["cluster_id"],
                    "tid": re.match(r"(cluster_\d+_t\d+)", n).group(1),
                    "members": {m["text_id"]: m for m in d.get("member_objects", []) if m.get("text_id")}}
                for n, d in threads.items()}
        todo = [g for g in groups
                if args.limit or not (MULTI_DIR / f'{meta[g[0]]["tid"]}__X__{meta[g[1]]["tid"]}.json').exists()]
        logger.info(f"multi: {len(todo)} groups to generate")
        mdone = {"n": 0, "pairs": 0}

        def mtask(g):
            a, b, ids = g
            am, bm = meta[a], meta[b]
            shared = [am["members"].get(t) or bm["members"].get(t) for t in ids]
            shared = [o for o in shared if o]
            pairs = gen_multi(a, b, am, bm, shared)
            if not pairs:                 # all-failed / none warranted — defer, don't mark done
                return 0
            gid = f'{am["tid"]}__X__{bm["tid"]}'
            (MULTI_DIR / f"{gid}.json").write_text(json.dumps({
                "group_id": gid, "entities": [am["label"], bm["label"]],
                "cluster_ids": [am["cluster_id"], bm["cluster_id"]],
                "thread_ids": [am["tid"], bm["tid"]], "num_shared_clauses": len(ids),
                "num_qa_pairs": len(pairs), "qa_pairs": pairs}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            return len(pairs)

        if todo:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futs = [ex.submit(mtask, g) for g in todo]
                for fut in tqdm(as_completed(futs), total=len(futs), desc="multi", unit="grp"):
                    with lock:
                        mdone["pairs"] += fut.result()
                        mdone["n"] += 1
        logger.info(f"multi done: {mdone['n']} groups, {mdone['pairs']} pairs")


if __name__ == "__main__":
    main()
