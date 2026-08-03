#!/usr/bin/env python3
"""
split_qa_by_type.py
════════════════════════════════════════════════════════════════════════════════
Aggregate all Q&A pairs from concept_qa/*_QA.json into ONE JSON file per question
type, written to concept_qa_by_type/.

Seven type files are produced (one per type). Each pooled pair keeps its original
fields and gains provenance (thread_id, cluster_id, cluster_label, thread_title) so
it can be traced back to its source thread.

    concept_qa_by_type/<type>.json:
      {"type": "...", "count": N, "qa_pairs": [ {<pair> + provenance}, ... ]}

Usage:
    python split_qa_by_type.py
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa"
MULTI_DIR = SCRIPT_DIR / "concept_qa_multi"
OUT_DIR = SCRIPT_DIR / "concept_qa_by_type"

TYPES = [
    "scenario_short_answer",
    "temporal_mcq",
    "event_sorting",
    "open_ended",
    "perturbation_detection",
    "context_misalignment",
    "temporal_scenario_short_answer",
    "relative_time_qa",
    "multi_entity_concept_qa",
]


def main() -> None:
    buckets: dict[str, list[dict]] = defaultdict(list)
    files = sorted(QA_DIR.glob("*_QA.json"))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        prov = {
            "thread_id": d.get("thread_id", ""),
            "cluster_id": d.get("cluster_id"),
            "cluster_label": d.get("cluster_label", ""),
            "thread_title": d.get("title", ""),
            "source_file": f.name,
        }
        for q in d.get("qa_pairs", []):
            t = q.get("type")
            if not t:
                continue
            item = dict(q)          # keep all original pair fields
            item.update(prov)       # add provenance
            buckets[t].append(item)

    # cross-concept (multi-entity) pairs live in their own directory, per group
    multi_files = sorted(MULTI_DIR.glob("*.json")) if MULTI_DIR.exists() else []
    for f in multi_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        prov = {
            "group_id": d.get("group_id", ""),
            "entities": d.get("entities", []),
            "cluster_ids": d.get("cluster_ids", []),
            "thread_ids": d.get("thread_ids", []),
            "source_file": f.name,
        }
        for q in d.get("qa_pairs", []):
            t = q.get("type")
            if not t:
                continue
            item = dict(q)
            item.update(prov)
            buckets[t].append(item)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # write the seven known types (plus any unexpected type, defensively)
    all_types = TYPES + [t for t in buckets if t not in TYPES]
    total = 0
    print(f"Read {len(files)} QA files.")
    for t in all_types:
        pairs = buckets.get(t, [])
        (OUT_DIR / f"{t}.json").write_text(
            json.dumps({"type": t, "count": len(pairs), "qa_pairs": pairs},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        total += len(pairs)
        print(f"  {t}.json: {len(pairs)} pairs")
    print(f"Wrote {len(all_types)} type files ({total} pairs total) → {OUT_DIR}/")


if __name__ == "__main__":
    main()
