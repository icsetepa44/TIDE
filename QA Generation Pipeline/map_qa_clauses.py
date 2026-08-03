#!/usr/bin/env python3
"""
map_qa_clauses.py
════════════════════════════════════════════════════════════════════════════════
Add a `clause_text_ids` key to EVERY Q&A pair (all 7 types) in concept_qa/*_QA.json,
listing the exact clause id(s) — the `text_id` values as they appear in the main
merged_ec_output_canon_subentities.json — of the source clause(s) the pair maps to.

How the mapping is derived (deterministic, no LLM):
  Each pair references its source clauses by DATE (YYYY-MM-DD) and/or SRO number
  (e.g. "SRO 139-AIN/2022"). Each of the thread's member clauses carries the exact
  text_id plus its clause_date, and the text_id itself encodes the date + SRO. We
  match a pair to every member clause whose date / SRO / year it references. If a
  pair references nothing resolvable, it maps to all of its thread's member clauses
  (it is grounded in the whole thread).

The text_ids are validated against the main subentities file. Existing keys/pairs
are untouched; only the new key is added. A one-time backup is made.

Usage:
    python map_qa_clauses.py
    python map_qa_clauses.py --only cluster_00326_t0     # inspect one
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa"
THREADS_DIR = SCRIPT_DIR / "concept_threads"
MAIN_FILE = SCRIPT_DIR / "merged_ec_output_canon_subentities.json"
BACKUP_DIR = SCRIPT_DIR / "concept_qa_backup_premap"
KEY = "clause_text_ids"


def _blob(pair: dict) -> str:
    return json.dumps(pair, ensure_ascii=False)


def _refs(pair: dict) -> tuple[set[str], set[str], set[str]]:
    """Return (full YYYY-MM-DD dates, 4-digit years, SRO numbers) referenced anywhere
    in the pair's fields."""
    b = _blob(pair)
    full = set(re.findall(r"\d{4}-\d{2}-\d{2}", b))
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", b))
    sros = set(re.findall(r"(\d{2,4})\s*[-/]?\s*AIN", b, re.I))
    sros |= set(re.findall(r"S\.?\s*R\.?\s*O\.?\s*(?:No\.?)?\s*(\d{2,4})", b, re.I))
    return full, years, sros


def _member_sro(text_id: str) -> str:
    """SRO number from a text_id like '20170601_135-AIN_2016_17-Customs_...' → '135'."""
    m = re.match(r"\d{8}_(\d{1,4})", text_id)
    return m.group(1) if m else ""


def map_pair(pair: dict, members: list[dict]) -> list[str]:
    full, years, sros = _refs(pair)
    matched: list[str] = []
    seen = set()
    for m in members:
        tid = m.get("text_id", "")
        if not tid or tid in seen:
            continue
        cdate = m.get("clause_date", "") or ""
        hit = (cdate in full) or (_member_sro(tid) in sros and _member_sro(tid)) \
            or (cdate[:4] in years and cdate)
        if hit:
            seen.add(tid)
            matched.append(tid)
    if not matched:                      # grounded in the whole thread
        for m in members:
            tid = m.get("text_id", "")
            if tid and tid not in seen:
                seen.add(tid)
                matched.append(tid)
    return matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    valid_ids = set()
    if MAIN_FILE.exists():
        main = json.loads(MAIN_FILE.read_text(encoding="utf-8"))
        for doc in main.get("merged_documents", []):
            for e in doc.get("processed_clauses", []):
                valid_ids.add(e.get("text_id"))

    if not args.dry_run and not args.only and not BACKUP_DIR.exists():
        shutil.copytree(QA_DIR, BACKUP_DIR)
        print(f"backed up concept_qa → {BACKUP_DIR.name}")

    files = sorted(QA_DIR.glob("*_QA.json"))
    if args.only:
        files = [f for f in files if args.only in f.name]

    total_pairs = 0
    total_links = 0
    unknown = 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        stem = f.name[:-len("_QA.json")]
        tp = THREADS_DIR / f"{stem}.json"
        members = json.loads(tp.read_text(encoding="utf-8")).get("member_objects", []) if tp.exists() else []
        for q in d.get("qa_pairs", []):
            ids = map_pair(q, members)
            if valid_ids:
                for i in ids:
                    if i not in valid_ids:
                        unknown += 1
            q[KEY] = ids
            total_pairs += 1
            total_links += len(ids)
        if not args.dry_run:
            f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'(dry-run) ' if args.dry_run else ''}mapped {total_pairs} pairs across {len(files)} files; "
          f"avg {total_links/max(1,total_pairs):.1f} clause ids/pair; "
          f"{unknown} ids not found in main file")


if __name__ == "__main__":
    main()
