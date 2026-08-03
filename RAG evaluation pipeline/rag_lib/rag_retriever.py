"""rag_retriever.py — query-driven context retrieval over the customs-law ChromaDB.

Instead of an oracle document lookup (mapping a question's gold `clause_text_ids`
to full OCR documents), we embed a query built from the QA record and pull the
nearest clauses out of the pre-built ChromaDB `customs_corpus` collection (Gemini
embeddings).

The returned `RetrievedContext` exposes the fields eval_lib's context-assembly
consumes (`resolved_docs`, `unresolved_ids`, `context_status`, `clause_refs`), so
build_context_block / build_capped_context / to_grounded_prompt / context_fields
work over retrieved clauses unchanged.

Configuration is via module-level globals (no CLI): TOP_K and the per-dataset
QUERY_FIELDS flags below.

Reuses the project-root RAG stack (config, embedding_utils, key_manager); the
root is this file's parent-of-parent, added to sys.path so `config.CHROMA_DIR`
etc. resolve no matter the working directory.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Sibling rag_lib modules (config, embedding_utils, key_manager) live in this
# file's own dir; shared modules (rubric_scoring, llm_council) live one up.
for _p in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from embedding_utils import embed_text, open_collection  # noqa: E402
from key_manager import AllKeysExhausted, KeyManager, TooLongError  # noqa: E402

logger = logging.getLogger("rag_retriever")

# ════════════════════════════════════════════════════════════════════════════
# Retrieval configuration (globals — no CLI)
# ════════════════════════════════════════════════════════════════════════════

TOP_K = 10  # clauses fed to the model per question (after dedup-by-parent)

# When a chunk of a long clause wins a slot, substitute the parent clause's
# FULL resolved_text (from JSON/simplified_corpus.json) instead of the ~1,200-
# char chunk. Ranking is untouched — the chunk's similarity still decides the
# slot; only the text fed to the model grows. The 82 giant clauses were never
# stored whole in ChromaDB, so the corpus JSON is the only full-text source.
EXPAND_CHUNKS_TO_PARENT = True

# Per-dataset-type flags: which record fields are concatenated into the
# retrieval query. Flip True/False to include/exclude a field for that type.
# List-valued fields (options, events_shuffled) are flattened to text.
QUERY_FIELDS: dict[str, dict[str, bool]] = {
    "open_ended":                     {"question": True},
    "relative_time_qa":               {"question": True},
    "scenario_short_answer":          {"scenario": True,  "question": True},
    "temporal_scenario_short_answer": {"scenario": True,  "question": True},
    "context_misalignment":           {"question": True,  "misaligned_context": False},
    "temporal_mcq":                   {"question": True,  "options": False},
    "event_sorting":                  {"events_shuffled": True, "question": False},
    "perturbation_detection":         {"statement": True},
}


# ════════════════════════════════════════════════════════════════════════════
# Retrieved-context shapes (duck-compatible with eval_lib's context-assembly)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedDoc:
    """One retrieved clause/chunk, carrying the attributes
    eval_lib.context_fields / build_context_block read."""
    stem: str            # human-readable id shown in the DOCUMENT header
    text: str            # the clause text (fed into the prompt)
    char_count: int
    source_file: str     # provenance (Chroma metadata `source_file`)
    category: str        # Chroma metadata `type` (clause | metadata)
    similarity: float    # cosine similarity (1 - distance)
    doc_id: str          # Chroma store id of the hit
    title: str = ""

    # `context_fields` accesses `.path`; expose the source_file as a Path so the
    # relative_to(...) try/except there resolves to the source_file string.
    @property
    def path(self) -> Path:
        return Path(self.source_file)


@dataclass
class RetrievedContext:
    """Result of retrieving for one QA instance; the field names are what
    eval_lib's context-assembly reads."""
    query: str
    resolved_docs: list[RetrievedDoc]
    context_status: str                     # "full" (>=1 hit) | "none_resolved"
    clause_refs: list[dict] = field(default_factory=list)  # per-hit retrieval metadata
    unresolved_ids: list[str] = field(default_factory=list)  # always [] for RAG


# ════════════════════════════════════════════════════════════════════════════
# Query construction
# ════════════════════════════════════════════════════════════════════════════

def _field_to_text(value) -> str:
    """Flatten a record field to a query string. Lists (options / events) are
    joined; dicts fall back to their string form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_field_to_text(v) for v in value if v).strip()
    return str(value).strip()


def build_query(dataset: str, rec: dict) -> str:
    """Concatenate the enabled fields (QUERY_FIELDS) for this dataset type.
    Unknown datasets fall back to the record's `question`."""
    flags = QUERY_FIELDS.get(dataset)
    if not flags:
        return _field_to_text(rec.get("question"))
    parts = [_field_to_text(rec.get(name)) for name, on in flags.items() if on]
    return "\n".join(p for p in parts if p).strip()


# ════════════════════════════════════════════════════════════════════════════
# Retriever
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class _RetrievalStats:
    """Lightweight run accounting, surfaced in the per-model summary."""
    queries: int = 0
    empty_queries: int = 0
    zero_hit_queries: int = 0
    total_hits: int = 0
    parent_expansions: int = 0
    top1_sims: list[float] = field(default_factory=list)


class RagRetriever:
    """Embeds a per-record query and returns the nearest customs clauses.

    Over-fetches then dedups by `parent_id` so a long clause stored as several
    chunks occupies at most one slot (same rule as build_vector_db.stage_search),
    keeping chunked long clauses on par with ordinary single-vector clauses.

    With EXPAND_CHUNKS_TO_PARENT on, a winning chunk's slot carries the parent
    clause's full resolved_text (small-to-big retrieval): the chunk still earns
    the slot on its own similarity, but the model sees the whole clause.
    """

    def __init__(self, top_k: int = TOP_K, key_manager=None,
                 over_fetch_mult: int = 4,
                 expand_chunks_to_parent: bool = EXPAND_CHUNKS_TO_PARENT):
        self.top_k = top_k
        self.over_fetch_mult = over_fetch_mult
        self.expand_chunks_to_parent = expand_chunks_to_parent
        # Reuse a shared Gemini key pool if given, else load the plaintext keys.
        self.km = key_manager or KeyManager.load(section=config.SECTION_GEMINI)
        self.collection = open_collection(create=False)
        count = self.collection.count()
        if count == 0:
            raise RuntimeError(
                f"ChromaDB collection '{config.COLLECTION_NAME}' is empty — "
                "build the embeddings first (build_vector_db.py).")
        self._count = count
        self._parent_texts = (self._load_parent_texts()
                              if expand_chunks_to_parent else {})
        self.stats = _RetrievalStats()

    @staticmethod
    def _load_parent_texts() -> dict[str, str]:
        """store_id -> full resolved_text for every corpus record, so chunk
        hits can be expanded to their parent clause. Chunks are the only rows
        with a parent_id, so lookups only ever target the 82 giants."""
        path = config.SIMPLIFIED_JSON
        if not path.exists():
            raise RuntimeError(
                f"EXPAND_CHUNKS_TO_PARENT is on but {path} is missing — "
                "run build_vector_db.py Stage 1 (RUN_SIMPLIFY) or set the "
                "toggle to False.")
        records = json.loads(path.read_text(encoding="utf-8"))
        texts = {str(r["store_id"]): r.get("resolved_text") or ""
                 for r in records if r.get("store_id")}
        logger.info("parent-expansion map loaded: %d corpus texts from %s",
                    len(texts), path)
        return texts

    def reset_stats(self) -> None:
        """Clear retrieval accounting (call between per-model runs so each
        summary.json reports only that model's retrieval stats)."""
        self.stats = _RetrievalStats()

    def retrieve(self, dataset: str, rec: dict) -> RetrievedContext:
        query = build_query(dataset, rec)
        self.stats.queries += 1
        if not query:
            self.stats.empty_queries += 1
            return RetrievedContext(query="", resolved_docs=[],
                                    context_status="none_resolved")

        qvec = embed_text(self.km, query, config.QUERY_TASK_TYPE)
        over_fetch = min(self.top_k * self.over_fetch_mult, self._count)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=over_fetch,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        seen: set[str] = set()
        hits: list[RetrievedDoc] = []
        refs: list[dict] = []
        for sid, doc, md, dist in zip(ids, docs, metas, dists):
            md = md or {}
            parent = md.get("parent_id") or sid  # normal clauses have no parent_id
            if parent in seen:
                continue
            seen.add(parent)
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            text = doc or ""
            stem = str(md.get("store_id") or md.get("title") or sid)
            expanded = False
            if md.get("parent_id") and self.expand_chunks_to_parent:
                parent_text = self._parent_texts.get(str(parent))
                if parent_text:
                    text = parent_text
                    stem = str(parent)  # header names the clause, not the chunk
                    expanded = True
                    self.stats.parent_expansions += 1
                else:
                    logger.warning("chunk %s hit but parent %s not in corpus "
                                   "map — falling back to chunk text", sid, parent)
            hits.append(RetrievedDoc(
                stem=stem,
                text=text,
                char_count=len(text),
                source_file=str(md.get("source_file") or ""),
                category=str(md.get("type") or ""),
                similarity=round(sim, 4),
                doc_id=str(sid),
                title=str(md.get("title") or ""),
            ))
            ref = {"id": str(sid), "similarity": round(sim, 4),
                   "title": str(md.get("title") or ""),
                   "source_file": str(md.get("source_file") or "")}
            if expanded:
                ref["expanded_to_parent"] = str(parent)
            refs.append(ref)
            if len(hits) >= self.top_k:
                break

        self.stats.total_hits += len(hits)
        if hits:
            self.stats.top1_sims.append(hits[0].similarity)
        else:
            self.stats.zero_hit_queries += 1

        status = "full" if hits else "none_resolved"
        return RetrievedContext(query=query, resolved_docs=hits,
                                context_status=status, clause_refs=refs)

    def resolution_summary(self) -> dict:
        """Retrieval stats for the per-model summary.json (parallels the oracle
        run's resolution stats block)."""
        s = self.stats
        mean_top1 = round(sum(s.top1_sims) / len(s.top1_sims), 4) if s.top1_sims else None
        return {
            "retrieval_mode": "rag_chromadb",
            "embed_model": config.EMBED_MODEL,
            "collection": config.COLLECTION_NAME,
            "top_k": self.top_k,
            "expand_chunks_to_parent": self.expand_chunks_to_parent,
            "parent_expansions": s.parent_expansions,
            "query_fields": QUERY_FIELDS,
            "queries": s.queries,
            "empty_queries": s.empty_queries,
            "zero_hit_queries": s.zero_hit_queries,
            "mean_hits_per_query": (round(s.total_hits / s.queries, 2)
                                    if s.queries else None),
            "mean_top1_similarity": mean_top1,
        }


# ════════════════════════════════════════════════════════════════════════════
# Smoke test (parameterized replacement for build_vector_db.stage_search's print)
# ════════════════════════════════════════════════════════════════════════════

_SAMPLE_QUERY_REC = {"question": "electronic bonded warehouse operation"}

if __name__ == "__main__":
    retriever = RagRetriever(top_k=TOP_K)
    ctx = retriever.retrieve("open_ended", _SAMPLE_QUERY_REC)
    print(f"\nQuery: {ctx.query!r}")
    print(f"Status: {ctx.context_status}  |  {len(ctx.resolved_docs)} hit(s)\n" + "=" * 70)
    for rank, d in enumerate(ctx.resolved_docs, 1):
        snippet = " ".join(d.text.split())[:220]
        print(f"\n#{rank}  sim={d.similarity:.3f}  [{d.category}]  "
              f"id={d.doc_id}  chars={d.char_count}"
              + ("  (expanded to parent)" if d.stem != d.doc_id and "#" in d.doc_id else ""))
        print(f"    title : {d.title}")
        print(f"    source: {d.source_file}")
        print(f"    text  : {snippet}...")
    print()
