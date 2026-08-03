"""
document_resolver.py — map QA `clause_text_ids` to OCR document JSONs.

Standalone module used by ICL.py to assemble grounded
prompt context. Two ID shapes are supported:

  Shape A — clause IDs: `<document_name>_<date>_clause_<n>` where <date> is
      inconsistent (Bengali script, ISO, English prose, sometimes empty).
      Resolved by boundary-guarded longest-prefix match of the ID against the
      index of filename stems under JSON/ocr_data/ — no date parsing.
  Shape B — metadata IDs: `metadata_<N>`. Resolved through
      JSON/merged_ec_output_canon_subentities.json: each merged_documents record's
      processed_clauses entries of type "metadata" carry a globally unique
      `merged_metadata_number` ("metadata_<N>"); the record's `source_file`
      (`<category>_OCR/<docname>/<docname>_output.json`) yields the document
      name (basename minus the `_output` suffix).

Unicode NFC normalization is applied to both stems and IDs (stems are ASCII
today, but the IDs carry Bengali date segments — cheap insurance against
byte-level mismatches).

Smoke test:
    python document_resolver.py           # fixed example IDs + sampled IDs
    python document_resolver.py --full    # resolve every ID in all 7 datasets
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("document_resolver")

SCRIPT_DIR = Path(__file__).resolve().parent
OCR_ROOT = SCRIPT_DIR / "JSON" / "ocr_data"
MERGED_FILE = SCRIPT_DIR / "JSON" / "merged_ec_output_canon_subentities.json"
QA_DIR = SCRIPT_DIR / "concept_qa_by_type"  # used by the smoke test only

# Corpus size at the time of writing; a mismatch is only warned about.
EXPECTED_DOC_COUNT = 644

_CLAUSE_SUFFIX_RE = re.compile(r"_clause_(\d+)$")
_METADATA_ID_RE = re.compile(r"^metadata_\d+$")
_OUTPUT_SUFFIX = "_output.json"


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@dataclass
class ResolvedDocument:
    """One loaded OCR document."""
    stem: str                    # document name == filename stem (unique)
    path: Path                   # absolute path to <stem>.json
    category: str                # customs_acts | customs_gos | customs_rules | customs_sros
    text: str                    # full flat OCR text ("" for empty-text docs)
    char_count: int
    document_date: str | None
    document_type: str | None


@dataclass
class ResolutionResult:
    """Resolution of one QA instance's clause_text_ids."""
    clause_text_ids: list[str]             # input IDs, as given
    resolved_docs: list[ResolvedDocument]  # deduped, first-occurrence order
    unresolved_ids: list[str]              # deduped, first-occurrence order
    # One entry per input ID, in input order. `clause_ref` is metadata only —
    # retained to enable a future clause-level ablation; NOT used for
    # resolution. None for metadata_<N> IDs.
    clause_refs: list[dict]
    context_status: str                    # "full" | "partial" | "none_resolved" | "no_ids"


@dataclass
class ResolverStats:
    """Cumulative resolution accounting for the current run."""
    total_ids: int = 0            # ID occurrences seen (not distinct)
    resolved_ids: int = 0
    unresolved_ids: int = 0
    metadata_ids: int = 0         # occurrences shaped metadata_<N>
    instances_total: int = 0
    instances_full: int = 0
    instances_partial: int = 0
    instances_none: int = 0
    instances_no_ids: int = 0
    unresolved_id_set: set[str] = field(default_factory=set)

    def as_dict(self) -> dict:
        return {
            "ids": {
                "total": self.total_ids,
                "resolved": self.resolved_ids,
                "unresolved": self.unresolved_ids,
                "metadata_shaped": self.metadata_ids,
            },
            "instances": {
                "total": self.instances_total,
                "full": self.instances_full,
                "partial": self.instances_partial,
                "none_resolved": self.instances_none,
                "no_ids": self.instances_no_ids,
            },
            "distinct_unresolved_ids": len(self.unresolved_id_set),
            "distinct_unresolved_sample": sorted(self.unresolved_id_set)[:50],
        }


class DocumentResolver:
    """
    Resolves clause_text_ids to OCR documents. The stem index is built
    eagerly (one cheap directory walk); the 19MB metadata lookup is parsed
    lazily on the first metadata_<N> ID; document texts are cached so no file
    is read from disk twice per run.
    """

    def __init__(self, ocr_root: Path = OCR_ROOT, merged_file: Path = MERGED_FILE):
        self.ocr_root = Path(ocr_root)
        self.merged_file = Path(merged_file)
        if not self.ocr_root.is_dir():
            raise FileNotFoundError(f"OCR corpus directory not found: {self.ocr_root}")

        # NFC stem -> (json path, category); layout <category>/<docname>/<docname>.json
        self._stem_index: dict[str, tuple[Path, str]] = {}
        for category_dir in sorted(p for p in self.ocr_root.iterdir() if p.is_dir()):
            for doc_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                doc_json = doc_dir / f"{doc_dir.name}.json"
                if not doc_json.is_file():
                    logger.warning("No %s.json inside %s — skipped", doc_dir.name, doc_dir)
                    continue
                stem = _nfc(doc_dir.name)
                if stem in self._stem_index:
                    logger.warning("Duplicate document stem %r (%s vs %s) — keeping first",
                                   stem, self._stem_index[stem][0], doc_json)
                    continue
                self._stem_index[stem] = (doc_json, category_dir.name)

        if len(self._stem_index) != EXPECTED_DOC_COUNT:
            logger.warning("Indexed %d documents (expected %d) under %s",
                           len(self._stem_index), EXPECTED_DOC_COUNT, self.ocr_root)
        else:
            logger.info("Indexed %d documents under %s", len(self._stem_index), self.ocr_root)

        # Longest-first so the first prefix hit is the longest one (ties in
        # length are impossible among prefixes of the same string; secondary
        # sort keeps the order deterministic anyway).
        self._stems_desc: list[str] = sorted(self._stem_index, key=lambda s: (-len(s), s))

        self._metadata_index: dict[str, str] | None = None
        self._doc_cache: dict[str, ResolvedDocument] = {}
        self._id_cache: dict[str, str | None] = {}
        self.stats = ResolverStats()

    # ── Shape B: metadata_<N> lookup ────────────────────────────────────────

    def _ensure_metadata_index(self) -> dict[str, str]:
        if self._metadata_index is not None:
            return self._metadata_index
        if not self.merged_file.is_file():
            raise FileNotFoundError(
                f"{self.merged_file} not found — required to resolve metadata_<N> "
                f"clause_text_ids (silently treating them as unresolved would "
                f"corrupt the grounded run).")
        logger.info("Loading metadata index from %s ...", self.merged_file.name)
        try:
            merged = json.loads(self.merged_file.read_text(encoding="utf-8"))
            records = merged["merged_documents"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Unexpected structure in {self.merged_file.name}: {e}") from e

        index: dict[str, str] = {}
        for rec in records:
            source_file = rec.get("source_file") or ""
            name = Path(source_file).name
            if name.endswith(_OUTPUT_SUFFIX):
                docname = name[: -len(_OUTPUT_SUFFIX)]
            else:  # fall back to the parent directory name
                docname = Path(source_file).parent.name or Path(name).stem
            docname = _nfc(docname)
            if docname not in self._stem_index:
                logger.warning("merged source_file %r derives to unknown stem %r — "
                               "its metadata IDs will be unresolvable", source_file, docname)
                continue
            for clause in rec.get("processed_clauses") or []:
                if clause.get("type") != "metadata":
                    continue
                num = clause.get("merged_metadata_number")
                if not num:
                    continue
                num = _nfc(str(num))
                prev = index.get(num)
                if prev is not None and prev != docname:
                    logger.warning("Duplicate merged_metadata_number %r (%r vs %r) — "
                                   "keeping first", num, prev, docname)
                    continue
                index[num] = docname
        self._metadata_index = index
        logger.info("Metadata index ready: %d merged_metadata_number entries", len(index))
        return index

    # ── ID resolution ────────────────────────────────────────────────────────

    def resolve_id(self, clause_text_id: str) -> str | None:
        """Return the document stem for one ID, or None (never raises for a
        well-formed corpus; missing merged file for a metadata ID does raise)."""
        if clause_text_id in self._id_cache:
            return self._id_cache[clause_text_id]
        cid = _nfc(clause_text_id.strip())
        stem: str | None = None
        if _METADATA_ID_RE.match(cid):
            stem = self._ensure_metadata_index().get(cid)
        else:
            # Boundary-guarded longest-prefix: the stem must be followed by "_"
            # (every clause ID continues `_<date>_clause_<n>`), which prevents
            # partial-token false positives; longest-first disambiguates
            # nested stems.
            for cand in self._stems_desc:
                if cid.startswith(cand + "_") or cid == cand:
                    stem = cand
                    break
        self._id_cache[clause_text_id] = stem
        return stem

    @staticmethod
    def extract_clause_ref(clause_text_id: str) -> str | None:
        """Trailing `clause_<n>` token, or None (metadata IDs). Metadata only —
        never used for resolution."""
        m = _CLAUSE_SUFFIX_RE.search(_nfc(clause_text_id.strip()))
        return f"clause_{m.group(1)}" if m else None

    # ── Document loading (cached) ────────────────────────────────────────────

    def load_document(self, stem: str) -> ResolvedDocument:
        doc = self._doc_cache.get(stem)
        if doc is not None:
            return doc
        path, category = self._stem_index[stem]
        data = json.loads(path.read_text(encoding="utf-8"))
        text = data.get("text") or ""
        if not isinstance(text, str):
            logger.warning("Document %s has non-string 'text' (%s) — joining defensively",
                           stem, type(text).__name__)
            text = "\n".join(str(t) for t in text) if isinstance(text, list) else str(text)
        if not text:
            logger.warning("Document %s has empty OCR text", stem)
        doc = ResolvedDocument(
            stem=stem, path=path, category=category, text=text,
            char_count=len(text),
            document_date=data.get("document_date"),
            document_type=data.get("document_type"),
        )
        self._doc_cache[stem] = doc
        return doc

    # ── Per-instance resolution ──────────────────────────────────────────────

    def resolve_instance(self, clause_text_ids: list[str]) -> ResolutionResult:
        ids = list(clause_text_ids or [])
        resolved_stems: dict[str, None] = {}   # insertion-ordered dedup
        unresolved: dict[str, None] = {}
        clause_refs: list[dict] = []
        n_resolved_occurrences = 0
        for cid in ids:
            stem = self.resolve_id(cid)
            clause_refs.append({
                "clause_text_id": cid,
                "doc": stem,
                "clause_ref": self.extract_clause_ref(cid),
            })
            if stem is None:
                unresolved[cid] = None
            else:
                resolved_stems[stem] = None
                n_resolved_occurrences += 1

        docs = [self.load_document(s) for s in resolved_stems]
        if not ids:
            status = "no_ids"
        elif not resolved_stems:
            status = "none_resolved"
        elif unresolved:
            status = "partial"
        else:
            status = "full"

        st = self.stats
        st.instances_total += 1
        st.total_ids += len(ids)
        st.resolved_ids += n_resolved_occurrences
        st.unresolved_ids += len(ids) - n_resolved_occurrences
        st.metadata_ids += sum(1 for cid in ids if _METADATA_ID_RE.match(_nfc(cid.strip())))
        st.unresolved_id_set.update(unresolved)
        if status == "full":
            st.instances_full += 1
        elif status == "partial":
            st.instances_partial += 1
        elif status == "none_resolved":
            st.instances_none += 1
        else:
            st.instances_no_ids += 1

        return ResolutionResult(
            clause_text_ids=ids,
            resolved_docs=docs,
            unresolved_ids=list(unresolved),
            clause_refs=clause_refs,
            context_status=status,
        )

    # ── Reporting ────────────────────────────────────────────────────────────

    def resolution_summary(self) -> dict:
        d = self.stats.as_dict()
        d["n_docs_indexed"] = len(self._stem_index)
        d["n_docs_loaded"] = len(self._doc_cache)
        d["n_empty_text_docs_loaded"] = sum(
            1 for doc in self._doc_cache.values() if not doc.text)
        return d

    def write_unresolved_log(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.stats
        lines = [
            "Clause-ID resolution report",
            f"generated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"ID occurrences: {s.total_ids} total, {s.resolved_ids} resolved, "
            f"{s.unresolved_ids} unresolved ({s.metadata_ids} metadata-shaped)",
            f"Instances: {s.instances_total} total — full={s.instances_full}, "
            f"partial={s.instances_partial}, none_resolved={s.instances_none}, "
            f"no_ids={s.instances_no_ids}",
            f"Documents: {len(self._doc_cache)} loaded of {len(self._stem_index)} indexed",
            "",
            f"Distinct unresolved IDs ({len(s.unresolved_id_set)}):",
        ]
        lines.extend(sorted(s.unresolved_id_set) or ["(none)"])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Resolution report written: %s", path)


# ════════════════════════════════════════════════════════════════════════════
# Standalone smoke test
# ════════════════════════════════════════════════════════════════════════════

FIXED_SMOKE_IDS = [
    "20221018_251-Act_2022_133_Customs_২০ জুলাই ২০২২_clause_1",
    "20221018_252-Act_2022_132_Customs_2022-07-21_clause_1",
    "20221018_259-Act_2022_134_Customs_August 4, 2022_clause_1",
]


def _collect_all_ids() -> list[tuple[str, str]]:
    """(dataset, clause_text_id) for every ID occurrence across the 7 QA files."""
    pairs: list[tuple[str, str]] = []
    for f in sorted(QA_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for rec in data.get("qa_pairs", []):
            for cid in rec.get("clause_text_ids") or []:
                pairs.append((f.stem, cid))
    return pairs


def _sample_smoke_ids(all_pairs: list[tuple[str, str]], n: int = 10,
                      seed: int = 42) -> list[tuple[str, str]]:
    """Seeded sample of n IDs across datasets, at least 2 metadata-shaped."""
    import random
    rng = random.Random(seed)
    meta = [p for p in all_pairs if _METADATA_ID_RE.match(p[1])]
    clause = [p for p in all_pairs if not _METADATA_ID_RE.match(p[1])]
    take_meta = min(2, len(meta), n)
    sample = rng.sample(meta, take_meta) + rng.sample(clause, min(n - take_meta, len(clause)))
    return sample


def _print_resolution_table(resolver: DocumentResolver,
                            rows: list[tuple[str, str]]) -> None:
    header = f"{'DATASET':<32} {'ID':<58} {'KIND':<8} {'STATUS':<10} {'DOC STEM':<42} {'CATEGORY':<14} {'CHARS':>7} {'CLAUSE_REF':<11}"
    print(header)
    print("-" * len(header))
    for dataset, cid in rows:
        kind = "metadata" if _METADATA_ID_RE.match(cid) else "clause"
        stem = resolver.resolve_id(cid)
        if stem is not None:
            doc = resolver.load_document(stem)
            status, category, chars = "resolved", doc.category, doc.char_count
        else:
            status, category, chars = "UNRESOLVED", "-", 0
        shown = cid if len(cid) <= 58 else cid[:55] + "..."
        print(f"{dataset:<32} {shown:<58} {kind:<8} {status:<10} {str(stem or '-'):<42} "
              f"{category:<14} {chars:>7} {DocumentResolver.extract_clause_ref(cid) or '-':<11}")


def _smoke_test(full: bool) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    resolver = DocumentResolver()

    if full:
        pairs = _collect_all_ids()
        print(f"\nFull audit: resolving all {len(pairs)} ID occurrences "
              f"across {len(set(d for d, _ in pairs))} datasets ...")
        unresolved: dict[str, list[str]] = {}
        n_ok = 0
        for dataset, cid in pairs:
            if resolver.resolve_id(cid) is None:
                unresolved.setdefault(cid, []).append(dataset)
            else:
                n_ok += 1
        print(f"Resolved {n_ok}/{len(pairs)} occurrences; "
              f"{len(unresolved)} distinct unresolved IDs.")
        for cid, datasets in sorted(unresolved.items()):
            print(f"  UNRESOLVED: {cid}  (in {', '.join(sorted(set(datasets)))})")
        if not unresolved:
            print("All IDs resolved — safe to launch the grounded evaluation.")
        return

    print("\n── Fixed example IDs ──")
    _print_resolution_table(resolver, [("(example)", cid) for cid in FIXED_SMOKE_IDS])

    print("\n── Sampled IDs across datasets (seed=42, ≥2 metadata-shaped) ──")
    all_pairs = _collect_all_ids()
    _print_resolution_table(resolver, _sample_smoke_ids(all_pairs))

    print("\n── resolve_instance() on the 3 fixed IDs (dedup/order demo) ──")
    res = resolver.resolve_instance(FIXED_SMOKE_IDS)
    print(f"context_status={res.context_status}; "
          f"documents (in order): {[d.stem for d in res.resolved_docs]}; "
          f"unresolved: {res.unresolved_ids}")
    print(f"clause_refs: {res.clause_refs}")

    print("\n── Resolution summary ──")
    print(json.dumps(resolver.resolution_summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke-test clause_text_id resolution.")
    ap.add_argument("--full", action="store_true",
                    help="Resolve every clause_text_id across all 7 QA datasets "
                         "(no document texts are loaded).")
    _smoke_test(ap.parse_args().full)
