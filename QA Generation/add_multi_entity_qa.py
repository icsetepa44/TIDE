#!/usr/bin/env python3
"""
add_multi_entity_qa.py
════════════════════════════════════════════════════════════════════════════════
Generate a new Q&A type — "multi_entity_concept_qa" — that spans DIFFERENT
entity-clusters (concept_threads). Two threads from different clusters that SHARE
clauses provide the "common ground"; the question must meaningfully involve BOTH
concepts/entities and be answerable from those shared clauses.

Output → concept_qa_multi/<groupid>.json, following the concept_qa pair structure
(each pair has type/question/answer + clause_text_ids). Clause ids come straight
from the shared-clause indices the model cites, so they are exact.

Free keys only. Resumable + idempotent (per group).

Usage:
    python add_multi_entity_qa.py [--limit N] [--concurrency K] [--min-shared M] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from break_subentities import setup_logging
from cluster_entities import KeyPool, _build_judge_rotator
from generate_qa import _parse_pairs, _clean
from add_temporal_qa import _patient_call

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
THREADS_DIR = SCRIPT_DIR / "concept_threads"
OUT_DIR = SCRIPT_DIR / "concept_qa_multi"
DONE_FILE = SCRIPT_DIR / "multi_entity_qa_done.json"

QTYPE = "multi_entity_concept_qa"
TARGET = 4
CONCURRENCY = 4
MIN_SHARED = 3                   # min clauses two threads must share to be paired
MAX_GROUPS_PER_THREAD = 3        # spread coverage across entity pairs
CLAUSE_CHARS = 400
MAX_SHARED_SHOWN = 12

PROMPT = """You are creating CROSS-CONCEPT Q&A from Bangladeshi customs/tax law. You are given
TWO different concepts/entities and a set of SHARED clauses (the common ground —
clauses that touch BOTH concepts). Use ONLY facts in the context.

Produce questions of type "{qtype}" where:
  • the question meaningfully involves BOTH Concept A and Concept B together (their
    relationship, interaction, or joint treatment) on the COMMON GROUND, and
  • the answer is grounded strictly in the shared clauses.
Only emit a pair if BOTH concepts are genuinely needed and it is factually sound;
otherwise return fewer or an empty list. Reference the shared clauses by their [index].

Concept A: "{label_a}" — {summary_a}
Concept B: "{label_b}" — {summary_b}

COMMON GROUND (shared clauses):
{clauses}

Return STRICT JSON (up to {target} strong pairs, or fewer / none):
{{"qa_pairs": [{{"type": "{qtype}",
  "question": "<question needing BOTH concepts>",
  "answer": "<answer grounded in the shared clauses>",
  "entities_involved": ["{label_a}", "{label_b}"],
  "clause_indices": [<indices of the shared clauses used>]}}]}}
Return only the JSON object."""


def _load_threads() -> dict:
    threads = {}
    for f in THREADS_DIR.glob("cluster_*.json"):
        if f.name.endswith("_index.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("num_member_objects", 0) <= 1:
            continue
        members = {m.get("text_id"): m for m in d.get("member_objects", []) if m.get("text_id")}
        threads[f.name] = {
            "cluster_id": d.get("cluster_id"), "label": d.get("cluster_label", ""),
            "title": d.get("title", ""), "summary": _clean(d.get("summary", "")),
            "members": members,
        }
    return threads


def _build_groups(threads: dict, min_shared: int) -> list[tuple[str, str, list[str]]]:
    """Pairs of threads (different clusters) sharing >= min_shared clauses, greedily
    selected with a per-thread cap so coverage spreads across entity pairs."""
    tid2th = defaultdict(set)
    for name, t in threads.items():
        for tid in t["members"]:
            tid2th[tid].add(name)
    pair_shared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tid, names in tid2th.items():
        if len(names) < 2:
            continue
        for a, b in combinations(sorted(names), 2):
            if threads[a]["cluster_id"] != threads[b]["cluster_id"]:
                pair_shared[(a, b)].append(tid)
    cand = sorted(((a, b, ids) for (a, b), ids in pair_shared.items() if len(ids) >= min_shared),
                  key=lambda x: -len(x[2]))
    used = defaultdict(int)
    groups = []
    for a, b, ids in cand:
        if used[a] >= MAX_GROUPS_PER_THREAD or used[b] >= MAX_GROUPS_PER_THREAD:
            continue
        groups.append((a, b, ids))
        used[a] += 1
        used[b] += 1
    return groups


def _group_id(a: str, b: str) -> str:
    ca = re.match(r"(cluster_\d+_t\d+)", a).group(1)
    cb = re.match(r"(cluster_\d+_t\d+)", b).group(1)
    return f"{ca}__X__{cb}"


def process_group(pool: KeyPool, threads: dict, a: str, b: str, shared_ids: list[str]) -> dict:
    ta, tb = threads[a], threads[b]
    shared = sorted(shared_ids, key=lambda t: t)[:MAX_SHARED_SHOWN]
    lines = []
    for i, tid in enumerate(shared):
        m = ta["members"].get(tid) or tb["members"].get(tid) or {}
        lines.append(f'[{i}] {tid}  ({m.get("clause_date","?")})\n'
                     f'    {_clean(m.get("resolved_text",""))[:CLAUSE_CHARS]}')
    prompt = PROMPT.format(qtype=QTYPE, target=TARGET, label_a=ta["label"], label_b=tb["label"],
                           summary_a=ta["summary"][:400], summary_b=tb["summary"][:400],
                           clauses="\n".join(lines))
    raw = _patient_call(pool, prompt)
    if raw is None:
        return {"_failed": True}
    pairs = _parse_pairs(raw, QTYPE)
    # resolve clause_indices -> exact text_ids
    for p in pairs:
        idxs = p.pop("clause_indices", []) if isinstance(p.get("clause_indices"), list) else []
        ids = []
        for x in idxs:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= xi < len(shared) and shared[xi] not in ids:
                ids.append(shared[xi])
        p["clause_text_ids"] = ids or shared[:]   # fall back to all shared clauses
    gid = _group_id(a, b)
    return {
        "group_id": gid,
        "entities": [ta["label"], tb["label"]],
        "cluster_ids": [ta["cluster_id"], tb["cluster_id"]],
        "thread_ids": [re.match(r"(cluster_\d+_t\d+)", a).group(1),
                       re.match(r"(cluster_\d+_t\d+)", b).group(1)],
        "num_shared_clauses": len(shared_ids),
        "num_qa_pairs": len(pairs),
        "qa_pairs": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--min-shared", type=int, default=MIN_SHARED)
    ap.add_argument("--paid", action="store_true", help="use paid key (gemini_api.txt)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging(False)

    threads = _load_threads()
    groups = _build_groups(threads, args.min_shared)
    logger.info(f"{len(threads)} threads; {len(groups)} cross-concept groups "
                f"(>= {args.min_shared} shared clauses)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if DONE_FILE.exists() and not args.force:
        done = set(json.loads(DONE_FILE.read_text()))
    todo = [g for g in groups if _group_id(g[0], g[1]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"{len(todo)} groups to process ({len(groups)-len(todo)} done)")

    if args.paid:
        paid = [k.strip() for k in re.split(r"[,\n]+", (SCRIPT_DIR / "gemini_api.txt").read_text()) if k.strip()]
        pool = KeyPool(paid)
        logger.info(f"using PAID key(s): {len(paid)}")
    else:
        pool = KeyPool(_build_judge_rotator().keys)
    lock = threading.Lock()
    total = made = failed = 0

    def task(g):
        a, b, ids = g
        return _group_id(a, b), process_group(pool, threads, a, b, ids)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(task, g) for g in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="multi", unit="group"):
                gid, res = fut.result()
                with lock:
                    if res.get("_failed"):
                        failed += 1
                        continue
                    (OUT_DIR / f"{gid}.json").write_text(
                        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                    total += res["num_qa_pairs"]
                    made += 1 if res["num_qa_pairs"] else 0
                    done.add(gid)
                    if len(done) % 20 == 0:
                        DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    logger.info(f"added {total} multi-entity pairs across {made} groups; "
                f"{failed} call-failures (retry on re-run)")


if __name__ == "__main__":
    main()
