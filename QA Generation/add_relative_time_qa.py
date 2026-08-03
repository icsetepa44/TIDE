#!/usr/bin/env python3
"""
add_relative_time_qa.py
════════════════════════════════════════════════════════════════════════════════
Add a new Q&A type — "relative_time_qa" — to concept_qa/*_QA.json.

The question gives an explicit SEED time but asks about a point some years EARLIER
or LATER using RELATIVE language (e.g. "six years later", "two years before"),
never naming the target year in the question. The answer resolves the relative
reference to the actual time/instrument and states the fact.

Not forced: only emitted when the thread has >=2 time points a meaningful gap
apart. Appended to each file's qa_pairs (existing pairs untouched); num_qa_pairs and
qa_counts_by_type updated. Idempotent + resumable. Free keys only.

`clause_text_ids` is added afterwards by re-running map_qa_clauses.py.

Usage:
    python add_relative_time_qa.py [--limit N] [--only ID] [--concurrency K] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from break_subentities import setup_logging
from cluster_entities import KeyPool, _build_judge_rotator
from generate_qa import _parse_pairs
from add_temporal_qa import build_lean_context, _patient_call

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa"
THREADS_DIR = SCRIPT_DIR / "concept_threads"
DONE_FILE = SCRIPT_DIR / "relative_time_qa_done.json"

QTYPE = "relative_time_qa"
TARGET = 4
CONCURRENCY = 4

PROMPT = """You are creating RELATIVE-TIME Q&A from how a Bangladeshi customs/tax concept
evolves over time. Use ONLY facts in the CONTEXT — never invent dates, rates,
instruments, categories or entities.

Produce questions of type "{qtype}" where:
  • the question states an explicit SEED time (a year/date that appears in the
    timeline), then asks about a DIFFERENT point expressed as a RELATIVE offset —
    e.g. "X years later", "X years earlier", "the following amendment", "a few
    years after that" — WITHOUT naming the target year in the question;
  • the target point must correspond to a REAL event in the timeline (seed +/- the
    offset lands on an actual instrument/change);
  • the ANSWER resolves the relative reference to the ACTUAL time/instrument and
    states the factual change, so the reader must reason across the gap.

DO NOT FORCE IT. Only emit a pair when the thread has >=2 real time points a
meaningful gap apart. If it lacks distinct temporal steps, return an EMPTY list.
Do not name the target year inside the question (keep it relative); do not invent
figures beyond the context.

CONTEXT:
{context}

Return STRICT JSON (up to {target} strong pairs, or fewer / none):
{{"qa_pairs": [{{"type": "{qtype}",
  "seed_time": "<the explicit seed year/date>",
  "relative_offset": "<e.g. 'six years later' / 'two years earlier'>",
  "question": "<question using the seed + relative offset, target year NOT named>",
  "answer": "<resolves to the actual time/instrument + the factual change>"}}]}}
Return only the JSON object."""


def process_file(pool: KeyPool, qa_path: Path) -> int:
    stem = qa_path.name[:-len("_QA.json")]
    thread_path = THREADS_DIR / f"{stem}.json"
    if not thread_path.exists():
        return 0
    thread = json.loads(thread_path.read_text(encoding="utf-8"))
    prompt = PROMPT.format(qtype=QTYPE, target=TARGET, context=build_lean_context(thread))
    raw = _patient_call(pool, prompt)
    if raw is None:
        return -1
    pairs = _parse_pairs(raw, QTYPE)

    d = json.loads(qa_path.read_text(encoding="utf-8"))
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
    ap.add_argument("--paid", action="store_true", help="use paid key (gemini_api.txt)")
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
        import re as _re
        paid = [k.strip() for k in _re.split(r"[,\n]+", (SCRIPT_DIR / "gemini_api.txt").read_text()) if k.strip()]
        pool = KeyPool(paid)
        logger.info(f"using PAID key(s): {len(paid)}")
    else:
        pool = KeyPool(_build_judge_rotator().keys)
    lock = threading.Lock()
    total = made = failed = 0

    def task(f: Path):
        return f.name, process_file(pool, f)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(task, f) for f in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="relative", unit="file"):
                name, n = fut.result()
                with lock:
                    if n < 0:
                        failed += 1
                        continue
                    total += n
                    made += 1 if n else 0
                    done.add(name)
                    if len(done) % 20 == 0:
                        DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    logger.info(f"added {total} relative_time pairs across {made} files; "
                f"{failed} call-failures (retry on re-run); rest not warranted")


if __name__ == "__main__":
    main()
