#!/usr/bin/env python3
"""
regen_event_sorting.py
════════════════════════════════════════════════════════════════════════════════
Regenerate ONLY the "event_sorting" Q&A type in concept_qa/*_QA.json, on the PAID
key with a better model (default gemini-3.5-flash).

WHY: in the v1 dataset 61% of event_sorting pairs printed the date straight into the
shuffled events — e.g. "SRO 151-172 repeals older notifications ... (2011-06-09)" —
so "arrange chronologically" was solvable by reading the parenthetical, not by
reasoning. Here the events must be DATE-FREE: ordering has to follow from the
SUBSTANCE of each change.

The date-free rule is enforced twice:
  1. in the prompt (no years/dates/months/Bengali date digits, no year-bearing SRO or
     act names, no ordering giveaways like "initial"/"subsequent"), and
  2. programmatically — any pair whose events still carry time information is dropped
     and the call retried, so a leak cannot reach the dataset.

Dates live ONLY in answer_ordered[].date (the answer key). answer_ordered is sorted
chronologically here (deterministic), and events_shuffled is verified to be genuinely
non-chronological.

Each pair keeps the full metadata: clause_text_ids, clause_source_files (mapped from
merged_ec_output_canon_subentities.json) and thread_source_file.

Idempotent: replaces this type only, leaving every other pair untouched. Resumable.

Usage:
    python regen_event_sorting.py                     # all threads, gemini-3.5-flash
    python regen_event_sorting.py --limit 2           # smoke test
    python regen_event_sorting.py --only cluster_00001_t0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import regenerate_qa as rq
from break_subentities import setup_logging

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa"
THREADS_DIR = SCRIPT_DIR / "concept_threads"
DONE_FILE = SCRIPT_DIR / "event_sorting_regen_done.json"

QTYPE = "event_sorting"
TARGET = 4
CONCURRENCY = 5
CLEAN_ATTEMPTS = 3          # regenerate this many times to get date-free events

# ── date/time detectors ──────────────────────────────────────────────────────────
_YEAR = re.compile(r"(?<!\d)(1[89]|20)\d{2}(?!\d)")
_ISO = re.compile(r"\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}")
_DMY = re.compile(r"\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}")
_MONTH = re.compile(r"\b(january|february|march|april|june|july|august|september|"
                    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b", re.I)
_MAY = re.compile(r"\bmay\s*\d|\d\s*may\b", re.I)          # "May 2015" / "29 May", not modal "may"
_BN_DIGITS = re.compile(r"[০-৯]{2,}")
_BN_MONTH = re.compile(r"(জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর)")
# a year-looking number followed by a unit is an amount ("2000 taka"), not a date
_UNIT_AFTER = re.compile(r"^\s*(taka|tk\b|bdt|kg|kilogram|ton|tonne|mt\b|litre|liter|ml\b|"
                         r"%|percent|crore|lakh|piece|pcs|unit|metric)", re.I)


def _has_time_info(s) -> bool:
    """True if the string carries any date/time signal that would give the order away."""
    t = str(s or "")
    if _ISO.search(t) or _DMY.search(t) or _MONTH.search(t) or _MAY.search(t) \
            or _BN_DIGITS.search(t) or _BN_MONTH.search(t):
        return True
    for m in _YEAR.finditer(t):
        if not _UNIT_AFTER.match(t[m.end():m.end() + 14]):   # a real year, not an amount
            return True
    return False


# Words that betray an event's position in the sequence without naming a date. Not a date,
# so this is a SOFT rule: we retry for wording free of them, but accept a date-free pair on
# the last attempt rather than lose the coverage.
_GIVEAWAY = re.compile(r"\b(initial(ly)?|first|original(ly)?|baseline|earlier|later|"
                       r"subsequent(ly)?|thereafter|afterwards?|final(ly)?|last|"
                       r"most recent(ly)?|newly|previous(ly)?|prior|then existing)\b", re.I)


def _has_order_giveaway(p: dict) -> bool:
    return any(_GIVEAWAY.search(str(e)) for e in p.get("events_shuffled", []))


PROMPT = """You build a high-quality EVENT SORTING task from how a Bangladeshi customs/tax concept
evolves over time. Use ONLY the facts in the context below — never invent events.

Produce pairs of type "event_sorting": take 3-6 REAL, distinct events/changes from this
concept's timeline, present them SHUFFLED (NOT in chronological order), and ask the reader
to arrange them chronologically.

════════ ABSOLUTE RULE — THE EVENTS MUST CONTAIN NO TIME INFORMATION ════════
The whole point is that the reader must infer the order from WHAT CHANGED, not read it off
a date. In every string of "events_shuffled" (and the matching "answer_ordered[].event"):
  • NO years — not 2015, not 1969, not in any form.
  • NO dates (2015-11-29, 29/11/2015) and NO month names, English or Bengali.
  • NO Bengali date digits (২০১৫, ২৯).
  • NO instrument/SRO numbers that embed a year — never "S.R.O. 353-AIN/2015", "AIN/2022".
    (A bare number like "SRO 353" is only acceptable if it reveals no year.)
  • NO act names carrying a year — write "the Customs Act", NEVER "the Customs Act 1969".
  • NO ordering giveaways — avoid "initial", "first", "original", "baseline", "earlier",
    "later", "subsequent", "thereafter", "final", "most recent", "newly", "now".
Describe each event PURELY by its SUBSTANCE: the specific rule, rate, scope, category,
procedure or exemption that changed, and how. Make each event self-contained and concrete
enough that its place in the sequence follows from the content itself.

The DATES belong ONLY in "answer_ordered[].date" — that is the answer key.
"answer_ordered[].event" must repeat the SAME date-free wording used in "events_shuffled".

{context}
{clauses}

Return STRICT JSON (up to {target} strong pairs, or fewer / none):
{{"qa_pairs":[{{"type":"event_sorting",
"question":"Arrange these events in chronological order (earliest first).",
"events_shuffled":["<date-free event>","<date-free event>","<date-free event>"],
"answer_ordered":[{{"event":"<same date-free wording>","date":"YYYY-MM-DD"}}],
"clause_indices":[<indices of the clauses used>]}}]}}

Only emit a pair when the thread genuinely has >=3 distinct dated events; otherwise return
an empty list. Include "clause_indices" for every pair. Return only the JSON object."""

_RETRY_NUDGE = """

REMINDER: your previous attempt gave the order away. In "events_shuffled" and
"answer_ordered[].event" remove EVERY year, date, month name and year-bearing SRO/act
reference, AND every positional word ("initial", "first", "original", "baseline", "later",
"subsequent", "final", "previously", ...). Describe each event ONLY by what substantively
changed, so the order must be inferred from the content."""


def _valid_pair(p: dict) -> bool:
    """Keep only well-formed pairs whose events are completely date-free."""
    evs = p.get("events_shuffled")
    ans = p.get("answer_ordered")
    if not isinstance(evs, list) or not isinstance(ans, list) or len(evs) < 3 or len(ans) < 3:
        return False
    if any(_has_time_info(e) for e in evs):
        return False
    if any(_has_time_info(a.get("event")) for a in ans if isinstance(a, dict)):
        return False
    if not all(isinstance(a, dict) and a.get("date") and a.get("event") for a in ans):
        return False
    return True


def _tidy(p: dict, seed: str) -> dict:
    """Sort the answer chronologically and guarantee the prompt list is really shuffled."""
    ans = sorted(p["answer_ordered"], key=lambda a: str(a.get("date", "")))
    p["answer_ordered"] = ans
    ordered = [a["event"] for a in ans]
    evs = list(p["events_shuffled"])
    if len(evs) > 1 and evs == ordered:          # not actually shuffled → deterministic shuffle
        rnd = random.Random(seed)
        while evs == ordered:
            rnd.shuffle(evs)
        p["events_shuffled"] = evs
    return p


def gen_events(name: str, thread: dict) -> list[dict] | None:
    clauses = rq._thread_clauses(thread)
    base = PROMPT.format(context=rq.build_context(thread), clauses=rq._clause_lines(clauses),
                         target=TARGET)
    for attempt in range(CLEAN_ATTEMPTS):
        last = attempt == CLEAN_ATTEMPTS - 1
        raw = rq._call(base if attempt == 0 else base + _RETRY_NUDGE)
        if raw is None:
            return None                       # call failed (503 cap) → defer, don't overwrite
        pairs = rq._parse_pairs(raw, QTYPE)
        clean = [p for p in pairs if _valid_pair(p)]          # hard rule: date-free
        strict = [p for p in clean if not _has_order_giveaway(p)]   # soft rule: no giveaways
        use = strict or (clean if last else [])
        if use:
            use = [_tidy(p, f"{name}:{i}") for i, p in enumerate(use)]
            return rq._resolve(use, clauses, name)
        if not pairs:                         # model judged the thread unsuitable
            return []
    return []                                 # never produced date-free events


def process(name: str) -> int:
    qa_path = QA_DIR / f"{name[:-5]}_QA.json"
    tpath = THREADS_DIR / name
    if not qa_path.exists() or not tpath.exists():
        return -1
    pairs = gen_events(name, json.loads(tpath.read_text(encoding="utf-8")))
    if pairs is None:
        return -1                             # API failure → retry on re-run
    d = json.loads(qa_path.read_text(encoding="utf-8"))
    kept = [q for q in d.get("qa_pairs", []) if q.get("type") != QTYPE]   # drop-and-replace
    d["qa_pairs"] = kept + pairs
    d["num_qa_pairs"] = len(d["qa_pairs"])
    counts = d.get("qa_counts_by_type", {})
    counts[QTYPE] = len(pairs)
    d["qa_counts_by_type"] = counts
    qa_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging(False)
    rq._MODEL = args.model
    logger.info(f"event_sorting regen on PAID key, model={rq._MODEL}, concurrency={args.concurrency}")

    names = sorted(f.name for f in THREADS_DIR.glob("cluster_*.json")
                   if not f.name.endswith("_index.json")
                   and (QA_DIR / f"{f.name[:-5]}_QA.json").exists())
    done = set()
    if DONE_FILE.exists() and not args.force:
        done = set(json.loads(DONE_FILE.read_text()))
    todo = [n for n in names if n not in done and (not args.only or args.only in n)]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"{len(names)} threads; {len(todo)} to regenerate ({len(done)} already done)")

    lock = threading.Lock()
    total = failed = empty = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(process, n): n for n in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="event_sorting", unit="thr"):
                n = futs[fut]
                r = fut.result()
                with lock:
                    if r < 0:
                        failed += 1
                    else:
                        total += r
                        empty += 1 if r == 0 else 0
                        done.add(n)
                        if len(done) % 20 == 0:
                            DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    logger.info(f"event_sorting: {total} date-free pairs written; {empty} threads yielded none; "
                f"{failed} call-failures (re-run to retry)")


if __name__ == "__main__":
    main()
