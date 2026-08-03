#!/usr/bin/env python3
"""
Merge per-document EC outputs into one corpus file (for the clustering stage)
=============================================================================

Flattens every per-document entity-concept output under --root into a single
`merged_ec_output.json`-style file and stamps each element with a corpus-GLOBAL
id, which is the input format the embedding/clustering stage (`cluster_gemini.py`)
consumes.

By default it merges the FINALIZED, canonicalized outputs (`*_ec_output_canon.json`)
so clustering runs on the curated benchmark, not the raw extraction. Pass
`--source raw` to merge the untouched `*_ec_output.json` originals instead.

It is deterministic and dependency-free (stdlib only). No API calls.

Output schema (matches what cluster_gemini.py expects):
  {
    "source_files":     [ "<path to each merged source file>", ... ],
    "merged_documents": [ { "source_file": "<relative input path>",
                            "processed_clauses": [ ...clause/metadata entries... ] }, ... ]
  }

Each processed-clause entry keeps all its original fields and gains one id field:
  - type == "clause"   → "merged_clause_number":   "clause_{n}"
  - type == "metadata" → "merged_metadata_number": "metadata_{m}"
The two counters are GLOBAL (continue across documents) and SEPARATE (clauses vs
metadata), both 1-indexed. Source order is preserved; the original text_id is left
untouched.

Usage
-----
  ./venv/bin/python merge_ec_outputs.py                          # canon → merged_ec_output_canon.json
  ./venv/bin/python merge_ec_outputs.py --root ./Custom_Data/Data
  ./venv/bin/python merge_ec_outputs.py --source raw --out merged_ec_output_raw.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOT = "./Custom_Data/Data"
DEFAULT_OUT = "merged_ec_output_canon.json"


def find_input_files(root: Path, source: str) -> list[Path]:
    """Discover the per-document EC outputs to merge, in sorted (category) order.

    source == "canon" → *_ec_output_canon.json (the finalized, canonical outputs)
    source == "raw"   → *_ec_output.json originals (excluding the *_canon.json)
    """
    if source == "canon":
        return sorted(root.glob("**/*_ec_output_canon.json"))
    return sorted(
        f for f in root.glob("**/*_ec_output.json") if "_canon" not in f.name
    )


def merge(files: list[Path]) -> dict:
    """Build the {source_files, merged_documents} structure with global ids."""
    clause_n = 0
    meta_n = 0
    source_files: list[str] = []
    merged_documents: list[dict] = []

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARN: skipping unreadable {f}: {e}", file=sys.stderr)
            continue

        out_clauses: list[dict] = []
        for clause in data.get("processed_clauses", []):
            entry = dict(clause)  # copy; preserve every existing key
            etype = entry.get("type")
            if etype == "clause":
                clause_n += 1
                entry["merged_clause_number"] = f"clause_{clause_n}"
            elif etype == "metadata":
                meta_n += 1
                entry["merged_metadata_number"] = f"metadata_{meta_n}"
            # any other type: include unchanged, no merged id
            out_clauses.append(entry)

        merged_documents.append({
            "source_file": data.get("source_file", ""),
            "processed_clauses": out_clauses,
        })
        source_files.append(str(f))

    print(
        f"Merged {len(merged_documents)} document(s): "
        f"{clause_n} clause(s) + {meta_n} metadata element(s)."
    )
    return {"source_files": source_files, "merged_documents": merged_documents}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge per-document EC outputs into one corpus file for clustering."
    )
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"Corpus root to scan (default: {DEFAULT_ROOT})")
    ap.add_argument("--source", choices=["canon", "raw"], default="canon",
                    help="Merge the canonicalized outputs (*_ec_output_canon.json) or the "
                         "raw originals (*_ec_output.json). Default: canon")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output file path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    files = find_input_files(root, args.source)
    if not files:
        print(f"No {args.source} EC output files found under {root}. Nothing to do.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} {args.source} EC output file(s) under {root}.")

    merged = merge(files)

    out_path = Path(args.out)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Failed to write {out_path}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
