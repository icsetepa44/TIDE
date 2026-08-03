#!/usr/bin/env python3
"""
split_by_cluster.py
════════════════════════════════════════════════════════════════════════════════
Filter `merged_ec_output_canon_clustered.json` by the unique values of each
element's `clause_clusters` field and write ONE JSON file per cluster number into
an output directory. Each per-cluster file contains every clause/metadata object
that belongs to that cluster.

Because a clause can sit in multiple clusters (its `clause_clusters` is a list),
the same object is written into each of its clusters' files.

Each output file (clusters_by_id/cluster_<id>.json) looks like:
    {
      "cluster_id": 85,
      "label": "effective date",          # canonical entity label (if available)
      "num_objects": 84,
      "objects": [ <full clause/metadata object>, ... ]
    }

Each object additionally carries a "source_file" key so the file is self-contained.

Usage:
    python split_by_cluster.py
    python split_by_cluster.py --min-objects 2        # skip singleton clusters
    python split_by_cluster.py --outdir clusters_by_id
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "merged_ec_output_canon_clustered.json"
CLUSTERS_FILE = SCRIPT_DIR / "entity_clusters.json"   # for human-readable labels
OUTDIR = SCRIPT_DIR / "clusters_by_id"
CLUSTER_KEY = "clause_clusters"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(INPUT_FILE))
    ap.add_argument("--clusters", default=str(CLUSTERS_FILE))
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--min-objects", type=int, default=1,
                    help="only write clusters with at least this many objects (default 1 = all)")
    ap.add_argument("--clean", action="store_true",
                    help="wipe the output directory before writing")
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    print(f"Reading {input_path.name} ...")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # optional cluster labels
    labels: dict[int, str] = {}
    clusters_path = Path(args.clusters)
    if clusters_path.exists():
        ecj = json.loads(clusters_path.read_text(encoding="utf-8"))
        labels = {c["cluster_id"]: c.get("label", "") for c in ecj.get("clusters", [])}

    # bucket every object into each of its clusters
    buckets: dict[int, list[dict]] = defaultdict(list)
    n_objects = 0
    for doc in data.get("merged_documents", []):
        source_file = doc.get("source_file", "")
        for elem in doc.get("processed_clauses", []):
            cids = elem.get(CLUSTER_KEY) or []
            if not cids:
                continue
            n_objects += 1
            # self-contained copy with provenance
            obj = dict(elem)
            obj.setdefault("source_file", source_file)
            for cid in cids:
                buckets[int(cid)].append(obj)

    # prepare output directory
    if args.clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pad = max(5, len(str(max(buckets) if buckets else 0)))
    written = 0
    skipped = 0
    index = []
    for cid in sorted(buckets):
        objs = buckets[cid]
        if len(objs) < args.min_objects:
            skipped += 1
            continue
        fname = f"cluster_{cid:0{pad}d}.json"
        payload = {
            "cluster_id": cid,
            "label": labels.get(cid, ""),
            "num_objects": len(objs),
            "objects": objs,
        }
        (outdir / fname).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({"cluster_id": cid, "label": labels.get(cid, ""),
                      "num_objects": len(objs), "file": fname})
        written += 1

    # an index for convenience
    index.sort(key=lambda r: (-r["num_objects"], r["cluster_id"]))
    (outdir / "_index.json").write_text(
        json.dumps({"total_clusters_written": written,
                    "total_distinct_objects": n_objects,
                    "clusters": index}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"Distinct objects with {CLUSTER_KEY}: {n_objects}")
    print(f"Clusters written: {written}  (skipped < {args.min_objects}: {skipped})")
    print(f"Output dir: {outdir}/  (+ _index.json)")


if __name__ == "__main__":
    main()
