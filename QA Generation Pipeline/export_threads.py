#!/usr/bin/env python3
"""
export_threads.py
════════════════════════════════════════════════════════════════════════════════
Write every evolving concept thread (from concept_results.json) to its own JSON
file under `concept_threads/`, with the FULL original clause objects embedded —
not just clause ids, but each clause's resolved text, entities, sub-entities and
concepts (the complete object as it appears in the clustered dataset).

Why indices, not text_ids: metadata text_ids (e.g. "metadata_2") are NOT unique
across documents, so threads are resolved back to their exact objects by
reproducing each cluster's date-sorted order (the same order the analysis used)
and indexing into it — collision-proof.

Each file (concept_threads/cluster_<cid>_t<n>__<slug>.json):
    {
      "cluster_id", "cluster_label", "thread_index", "thread_id",
      "title", "summary", "evolution", "timeline",
      "num_member_objects", "member_objects":            [ <full clause object>, ... ],
      "num_static_replicated_objects", "static_replicated_objects": [ ... ]
    }

Each embedded object additionally carries "source_file" and "clause_date".

Usage:
    python export_threads.py
    python export_threads.py --outdir concept_threads
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from concept_cluster import _date_of

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTERED_FILE = SCRIPT_DIR / "merged_ec_output_canon_clustered.json"
RESULTS_FILE = SCRIPT_DIR / "concept_results.json"
OUTDIR = SCRIPT_DIR / "concept_threads"


def _slug(text: str, n: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:n].rstrip("-")) or "thread"


def _ordered_objects(objs: list[dict], source_of: dict[int, str]) -> list[dict]:
    """Reproduce concept_cluster._build_items ordering: sort by (date, text_id).
    Returns the ORIGINAL objects in that exact order so a stored index maps back
    to its object unambiguously."""
    decorated = []
    for e in objs:
        decorated.append(((_date_of(e) or "undated", e.get("text_id", "")), e))
    decorated.sort(key=lambda t: t[0])
    return [e for _, e in decorated]


def _enrich(obj: dict, source_file: str) -> dict:
    """Full original object + provenance/date; existing keys untouched."""
    out = dict(obj)
    out.setdefault("source_file", source_file)
    out.setdefault("clause_date", _date_of(obj) or "undated")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--clean", action="store_true", default=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)

    print("Loading clustered data + concept results ...")
    clustered = json.loads(CLUSTERED_FILE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

    # rebuild per-cluster object lists (+ each object's source_file), exactly as the
    # concept stage saw them
    objs: dict[int, list[dict]] = defaultdict(list)
    src_for: dict[int, str] = {}
    for doc in clustered["merged_documents"]:
        sf = doc.get("source_file", "")
        for e in doc["processed_clauses"]:
            for cid in (e.get("clause_clusters") or []):
                objs[cid].append(e)
                src_for[id(e)] = sf

    if args.clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    index = []
    n_files = 0
    n_member = 0
    for cid_str, res in results.items():
        cid = int(cid_str)
        threads = res.get("concept_clusters") or []
        if not threads:
            continue
        ordered = _ordered_objects(objs.get(cid, []), src_for)
        n = len(ordered)

        def obj_at(i: int) -> dict | None:
            if 0 <= i < n:
                e = ordered[i]
                return _enrich(e, src_for.get(id(e), ""))
            return None

        for ti, cc in enumerate(threads):
            members = [o for o in (obj_at(i) for i in cc.get("member_clause_indices", [])) if o]
            statics = [o for o in (obj_at(i) for i in cc.get("static_replicated_clause_indices", [])) if o]
            thread_id = f"cluster_{cid:05d}_t{ti}"
            fname = f"{thread_id}__{_slug(cc.get('title',''))}.json"
            payload = {
                "cluster_id": cid,
                "cluster_label": res.get("label", ""),
                "thread_index": ti,
                "thread_id": thread_id,
                "title": cc.get("title", ""),
                "summary": cc.get("summary", ""),
                "evolution": cc.get("evolution", {}),
                "timeline": [{k: t.get(k) for k in ("date", "instrument", "change", "clause_ids")}
                             for t in (cc.get("timeline") or [])],
                "num_member_objects": len(members),
                "member_objects": members,
                "num_static_replicated_objects": len(statics),
                "static_replicated_objects": statics,
            }
            (outdir / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
            index.append({"thread_id": thread_id, "file": fname,
                          "cluster_id": cid, "cluster_label": res.get("label", ""),
                          "title": cc.get("title", ""),
                          "num_member_objects": len(members),
                          "num_static_replicated_objects": len(statics)})
            n_files += 1
            n_member += len(members)

    index.sort(key=lambda r: (-r["num_member_objects"], r["cluster_id"]))
    (outdir / "_index.json").write_text(
        json.dumps({"total_threads": n_files, "total_member_objects": n_member,
                    "threads": index}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {n_files} thread files ({n_member} member objects) → {outdir}/")
    print(f"Index: {outdir}/_index.json")


if __name__ == "__main__":
    main()
