#!/usr/bin/env python3
"""
Cross-Document Entity Canonicalization
======================================

Post-hoc pass that clusters near-duplicate entity strings produced by the
extraction pipeline (across the Act, SROs, Rules, and GOs) into a single
canonical name per cluster. This complements the in-prompt "preferred names"
nudge: the pipeline keeps spellings *mostly* consistent at generation time;
this script enforces consistency *after the fact* so the benchmark links the
same legal subject across all four document levels.

It is deterministic and dependency-free (stdlib only — difflib for fuzzy
matching). No API calls.

What it does
------------
1. Collects the entity universe from every `entities.txt` under --root
   (customs_*_OCR/entities.txt), or from explicit --entities-files.
2. Counts how often each entity name occurs across all `*_ec_output.json`
   files (used to pick the canonical representative of each cluster).
3. Clusters names by:
     - exact match after normalization (case/space/punctuation folding),
     - acronym ↔ expansion (e.g., 'VAT' ↔ 'Value Added Tax'),
     - fuzzy string similarity (difflib ratio >= --similarity) within blocks.
4. Picks a canonical name per cluster: most frequent → most tokens →
   longest → alphabetical.
5. Writes:
     - <out>.json         : {canonical: [variants...]} + reverse {variant: canonical}
     - <out>.report.txt   : only the multi-member clusters (the actual merges),
                            for human review.
     - <out>.entities.txt : the single merged corpus-wide vocabulary — one
                            canonical name per cluster, sorted (all four
                            category entities.txt collapsed into one list).
6. Optionally (--apply) rewrites each `*_ec_output.json`, remapping
   entity_concepts keys to canonical names (merging concept lists on
   collision), into a sibling `*_canon.json` (originals untouched).

Usage
-----
  ./venv/bin/python canonicalize_entities.py                       # dry run → map + report
  ./venv/bin/python canonicalize_entities.py --root ./Custom_Data/Data --similarity 0.9
  ./venv/bin/python canonicalize_entities.py --apply               # cluster + rewrite outputs
  ./venv/bin/python canonicalize_entities.py \
      --entities-files a/entities.txt b/entities.txt --out my_canon

Curated apply (review-then-apply for a benchmark)
-------------------------------------------------
  1. Dry run to produce entity_canonical_map.json + .report.txt
  2. Eyeball the report; in entity_canonical_map.json, delete any WRONG variant
     from its "canonical_to_variants" cluster (split it back out)
  3. Apply exactly the edited map (no re-clustering):
       ./venv/bin/python canonicalize_entities.py --from-map entity_canonical_map.json
     This rewrites the *_canon.json outputs AND refreshes a sibling
     <map-stem>.entities.txt with the curated vocabulary (names you split out
     reappear on their own, since they map to themselves).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_ROOT = "./Custom_Data/Data"
DEFAULT_SIMILARITY = 0.90
DEFAULT_OUT = "entity_canonical_map"

# Generic tokens dropped before fuzzy comparison / token-count tie-breaks so
# that names differing only in boilerplate still cluster.
_STOPWORDS = {"the", "of", "and", "for", "a", "an", "to", "in", "on"}


# ════════════════════════════════════════════════════════════════════════════════
# Discovery
# ════════════════════════════════════════════════════════════════════════════════

def find_entities_files(root: Path) -> list[Path]:
    """All entities.txt files under the customs_*_OCR category folders (and root)."""
    found = sorted(root.glob("**/entities.txt"))
    return found


def find_ec_output_files(root: Path) -> list[Path]:
    """All *_ec_output.json files under root."""
    return sorted(root.glob("**/*_ec_output.json"))


def load_entities(files: list[Path]) -> set[str]:
    """Union of all non-empty lines across the given entities.txt files."""
    universe: set[str] = set()
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    name = line.strip()
                    if name:
                        universe.add(name)
        except OSError as e:
            print(f"  WARN: could not read {f}: {e}", file=sys.stderr)
    return universe


def count_frequencies(ec_files: list[Path]) -> Counter:
    """
    Count occurrences of each entity name across all *_ec_output.json files
    (one increment per processed clause that mentions the entity).
    """
    freq: Counter = Counter()
    for f in ec_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARN: could not read {f}: {e}", file=sys.stderr)
            continue
        for clause in data.get("processed_clauses", []):
            for name in (clause.get("entity_concepts") or {}):
                freq[name] += 1
    return freq


# ════════════════════════════════════════════════════════════════════════════════
# Normalization & similarity
# ════════════════════════════════════════════════════════════════════════════════

def normalize(name: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9ঀ-৿]+", " ", s)  # keep Bangla range too
    return re.sub(r"\s+", " ", s).strip()


def significant_tokens(name: str) -> list[str]:
    return [t for t in normalize(name).split() if t and t not in _STOPWORDS]


def acronym_of(name: str) -> str:
    """Initials of capitalized words, e.g. 'Value Added Tax' → 'VAT'."""
    words = [w for w in re.findall(r"[A-Za-z]+", name) if w[:1].isupper()]
    return "".join(w[0] for w in words).upper() if len(words) >= 2 else ""


def _number_tokens(name: str) -> set[str]:
    """Digit runs in a name (years, SRO numbers, section numbers, ...)."""
    return set(re.findall(r"\d+", name))


# Uppercase Roman-numeral token (I, II, III, IV, ... — as in TABLE-I vs TABLE-II).
_ROMAN_RE = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _roman_tokens(name: str) -> set[str]:
    """Standalone UPPERCASE Roman-numeral tokens in a name. Treated like the
    digit guard: differing Roman numerals mean different items (Table I ≠ II)."""
    return {
        t for t in re.findall(r"[A-Za-z]+", name)
        if t.isupper() and _ROMAN_RE.match(t)
    }


def is_acronym_pair(a: str, b: str) -> bool:
    """True if one name is the (uppercase) acronym of the other.
    A name carrying digits (a year or instrument number — e.g. 'Standard Rules of
    Origin 1977', 'S.R.O. 305/2011') is NEVER an acronym match: those are specific
    dated/numbered instruments, not the generic expansion of a bare acronym."""
    if _number_tokens(a) or _number_tokens(b):
        return False
    for short, long in ((a, b), (b, a)):
        s = re.sub(r"[^A-Za-z]", "", short).upper()
        if 2 <= len(s) <= 6 and s == acronym_of(long):
            return True
    return False


def similar(a: str, b: str, threshold: float) -> bool:
    """Fuzzy match on normalized strings, plus acronym detection."""
    # Domain guard: names carrying DIFFERENT numbers are different legal
    # instruments (e.g., Baggage Rules 2012 vs 2016, S.R.O. 169 vs 170, or
    # different section numbers). Never merge them — the year/number is
    # semantically load-bearing. (A name with no number may still merge into
    # a numbered one, treating it as an ambiguous generic reference.)
    na_nums, nb_nums = _number_tokens(a), _number_tokens(b)
    if na_nums and nb_nums and na_nums != nb_nums:
        return False

    # Same guard for Roman numerals (Table I / II / III are distinct).
    na_rom, nb_rom = _roman_tokens(a), _roman_tokens(b)
    if na_rom and nb_rom and na_rom != nb_rom:
        return False

    if is_acronym_pair(a, b):
        return True
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Quick token-set Jaccard gate before the more expensive ratio.
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb:
        jacc = len(ta & tb) / len(ta | tb)
        if jacc >= threshold:
            return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


# ════════════════════════════════════════════════════════════════════════════════
# Clustering (union-find with light blocking)
# ════════════════════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:        # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster(names: list[str], threshold: float) -> list[list[str]]:
    """Group names into near-duplicate clusters."""
    uf = UnionFind(names)

    # Blocking: group candidates by the first character of their first
    # significant token to avoid O(n^2) across the whole set. Acronyms are
    # handled by a separate global pass since their block differs.
    blocks: dict[str, list[str]] = {}
    for n in names:
        toks = significant_tokens(n)
        key = toks[0][0] if toks else (normalize(n)[:1] or "_")
        blocks.setdefault(key, []).append(n)

    for block in blocks.values():
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if similar(block[i], block[j], threshold):
                    uf.union(block[i], block[j])

    # Global acronym pass (acronym and expansion usually fall in different blocks)
    acro_names = [n for n in names if acronym_of(n) or re.fullmatch(r"[A-Za-z]{2,6}", n.strip())]
    for i in range(len(acro_names)):
        for j in range(i + 1, len(acro_names)):
            if is_acronym_pair(acro_names[i], acro_names[j]):
                uf.union(acro_names[i], acro_names[j])

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(uf.find(n), []).append(n)
    return list(groups.values())


def pick_canonical(variants: list[str], freq: Counter) -> str:
    """Most frequent → most significant tokens → longest → alphabetical first."""
    return sorted(
        variants,
        key=lambda v: (-freq.get(v, 0), -len(significant_tokens(v)), -len(v), v),
    )[0]


# ════════════════════════════════════════════════════════════════════════════════
# Apply remapping to EC outputs
# ════════════════════════════════════════════════════════════════════════════════

def apply_to_ec_outputs(ec_files: list[Path], variant_to_canonical: dict[str, str]) -> int:
    """Rewrite entity_concepts keys to canonical names → sibling *_canon.json."""
    written = 0
    for f in ec_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARN: skip {f}: {e}", file=sys.stderr)
            continue

        for clause in data.get("processed_clauses", []):
            ecs = clause.get("entity_concepts") or {}
            if not ecs:
                continue
            remapped: dict[str, list[str]] = {}
            for name, concepts in ecs.items():
                canon = variant_to_canonical.get(name, name)
                bucket = remapped.setdefault(canon, [])
                for c in concepts:
                    if c not in bucket:
                        bucket.append(c)
            clause["entity_concepts"] = remapped

        out = f.with_name(f.stem + "_canon.json")
        try:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            written += 1
        except OSError as e:
            print(f"  WARN: could not write {out}: {e}", file=sys.stderr)
    return written


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-document entity canonicalization")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"Root to scan for entities.txt / *_ec_output.json (default: {DEFAULT_ROOT})")
    ap.add_argument("--entities-files", nargs="*", default=None,
                    help="Explicit entities.txt paths (overrides --root discovery for the name universe)")
    ap.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY,
                    help=f"Fuzzy match threshold 0-1 (default: {DEFAULT_SIMILARITY})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output basename for the map/report (default: {DEFAULT_OUT})")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite *_ec_output.json with canonical names → *_canon.json")
    ap.add_argument("--from-map", default=None,
                    help="Skip clustering and apply a (hand-curated) "
                         "<out>.json map directly. Edit canonical_to_variants to "
                         "remove bad merges, then apply exactly what you approve.")
    args = ap.parse_args()

    root = Path(args.root)

    # ── Curated apply: bypass clustering, apply an edited map verbatim ──
    if args.from_map:
        map_path = Path(args.from_map)
        try:
            with open(map_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Could not read map file {map_path}: {e}", file=sys.stderr)
            sys.exit(1)
        # canonical_to_variants is the source of truth; rebuild the reverse map
        # from it so the user only needs to edit that readable section.
        c2v = saved.get("canonical_to_variants") or {}
        variant_to_canonical = {v: canon for canon, vs in c2v.items() for v in vs}
        if not variant_to_canonical:
            print(f"No usable 'canonical_to_variants' in {map_path}.", file=sys.stderr)
            sys.exit(1)
        ec_files = find_ec_output_files(root)
        merges = sum(1 for vs in c2v.values() if len(vs) > 1)
        print(f"Curated apply from {map_path}: {len(c2v)} canonical name(s), "
              f"{merges} multi-variant cluster(s), over {len(ec_files)} output file(s).")
        n = apply_to_ec_outputs(ec_files, variant_to_canonical)
        print(f"Applied curated map → wrote {n} *_canon.json file(s).")

        # Refresh the merged vocabulary to match the curated map: map EVERY known
        # entity name through the edited map and dedupe. This includes names a
        # user split back out (removed from a cluster → they map to themselves).
        ent_files = ([Path(p) for p in args.entities_files]
                     if args.entities_files else find_entities_files(root))
        universe = load_entities(ent_files)
        curated_vocab = sorted({variant_to_canonical.get(name, name) for name in universe})
        out_entities = map_path.with_name(map_path.stem + ".entities.txt")
        with open(out_entities, "w", encoding="utf-8") as fh:
            for name in curated_vocab:
                fh.write(name + "\n")
        print(f"Wrote curated merged vocabulary ({len(curated_vocab)}) → {out_entities}")
        return

    # ── Gather the entity universe ──
    if args.entities_files:
        ent_files = [Path(p) for p in args.entities_files]
    else:
        ent_files = find_entities_files(root)
    if not ent_files:
        print("No entities.txt files found. Nothing to do.", file=sys.stderr)
        sys.exit(1)
    print(f"Reading {len(ent_files)} entities.txt file(s):")
    for f in ent_files:
        print(f"  - {f}")
    universe = sorted(load_entities(ent_files))
    print(f"Entity universe: {len(universe)} unique name(s)")

    # ── Frequencies (for canonical selection) ──
    ec_files = find_ec_output_files(root)
    freq = count_frequencies(ec_files)
    print(f"Scanned {len(ec_files)} *_ec_output.json file(s) for frequencies")

    # ── Cluster ──
    clusters = cluster(universe, args.similarity)
    clusters.sort(key=len, reverse=True)
    merges = [c for c in clusters if len(c) > 1]
    print(f"Formed {len(clusters)} cluster(s); {len(merges)} contain >1 variant")

    canonical_map: dict[str, list[str]] = {}
    variant_to_canonical: dict[str, str] = {}
    for variants in clusters:
        canon = pick_canonical(variants, freq)
        canonical_map[canon] = sorted(variants)
        for v in variants:
            variant_to_canonical[v] = canon

    # ── Write map JSON ──
    out_json = Path(f"{args.out}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {"canonical_to_variants": canonical_map,
             "variant_to_canonical": variant_to_canonical},
            fh, ensure_ascii=False, indent=2,
        )
    print(f"Wrote canonical map → {out_json}")

    # ── Write human-readable report (merges only) ──
    out_report = Path(f"{args.out}.report.txt")
    with open(out_report, "w", encoding="utf-8") as fh:
        fh.write(f"Entity canonicalization report (similarity >= {args.similarity})\n")
        fh.write(f"{len(universe)} names → {len(clusters)} clusters "
                 f"({len(merges)} merges)\n")
        fh.write("=" * 70 + "\n\n")
        for variants in sorted(merges, key=lambda c: (-len(c), c[0].lower())):
            canon = pick_canonical(variants, freq)
            fh.write(f"CANONICAL: {canon}  (freq={freq.get(canon, 0)})\n")
            for v in sorted(variants):
                if v != canon:
                    fh.write(f"    ← {v}  (freq={freq.get(v, 0)})\n")
            fh.write("\n")
    print(f"Wrote merge report → {out_report}")

    # ── Write merged canonical entities.txt (one line per cluster, deduped) ──
    # The single corpus-wide entity vocabulary: every category's entities.txt
    # collapsed to one canonical name per cluster, sorted alphabetically.
    out_entities = Path(f"{args.out}.entities.txt")
    with open(out_entities, "w", encoding="utf-8") as fh:
        for name in sorted(canonical_map):
            fh.write(name + "\n")
    print(f"Wrote merged canonical entities ({len(canonical_map)}) → {out_entities}")

    # ── Optionally apply ──
    if args.apply:
        n = apply_to_ec_outputs(ec_files, variant_to_canonical)
        print(f"Applied canonical names → wrote {n} *_canon.json file(s)")
    else:
        print("Dry run (map/report only). Re-run with --apply to rewrite EC outputs.")


if __name__ == "__main__":
    main()
