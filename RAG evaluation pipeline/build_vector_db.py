"""build_vector_db.py — customs-law RAG vector-DB builder (all four stages in one
toggle-driven module): simplify the corpus, embed it into ChromaDB, chunk + embed
the long clauses, and search the store.

Run:  python build_vector_db.py

Flip the RUN SETTINGS below to choose which stages run and how. The stages run in order:

    1. RUN_SIMPLIFY  — slim the 19MB canonical corpus to simplified_corpus.json
    2. RUN_EMBED     — embed every resolved_text into ChromaDB (resumable)
    3. RUN_LONG      — chunk + embed the long clauses build skipped (resumable)
    4. RUN_SEARCH    — embed QUERY and print the nearest TOP_K clauses

Each stage writes the file the next one reads, so you can run any subset. The corpus and
its simplified form live in JSON/; resume bookmarks land in state/ and logs/ (see config.py);
the vector store stays in chroma_db/.

Shared helpers (JSON I/O, embedding call, Chroma factory, chunking) live in embedding_utils.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The helper/config modules live in OpenRouter/rag_lib/; the shared judge modules
# live in OpenRouter/ itself. Put both on the path before importing them.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "rag_lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config
from key_manager import KeyManager, TooLongError, AllKeysExhausted
from embedding_utils import (
    load_json, save_json, metadata_for, embed_text, open_collection, load_done_ids,
    chunk_text, chunk_metadata, embed_with_fallback,
)

# --------------------------------------------------------------------------- #
# RUN SETTINGS                                                                #
# --------------------------------------------------------------------------- #
# EMBEDDING MODEL: not set here. The model + output dimension live in
# rag_lib/config.py -> EMBED_MODEL and EMBED_DIM. They are deliberately shared:
# this builder embeds the documents and RAG.py embeds the queries with the SAME
# model, so keep them in one place (config.py). After changing EMBED_MODEL,
# delete chroma_db/ and re-run RUN_EMBED + RUN_LONG so the whole store is
# rebuilt with the new model (query and document vectors must match).

# Which stages to run, in order.
RUN_SIMPLIFY = False
RUN_EMBED    = False
RUN_LONG     = False
RUN_SEARCH   = True

# Only process the first N pending items (0 = process everything).
EMBED_LIMIT  = 0   # build stage: first N pending records (e.g. 20 for a quick dry run)
LONG_LIMIT   = 0   # long stage: first N pending parent clauses (e.g. 2 for a quick dry run)

# Search stage.
QUERY = "electronic bonded warehouse operation"  # the natural-language query to search for
TOP_K = 10                                         # number of results to return
# When a hit is a chunk of a long clause, return the parent clause's FULL
# resolved_text (from JSON/simplified_corpus.json) instead of the ~1,200-char
# chunk. Ranking is untouched — the chunk's similarity still earns the slot.
# Same rule as rag_lib/rag_retriever.py (EXPAND_CHUNKS_TO_PARENT there).
EXPAND_CHUNKS_TO_PARENT = True


# =========================================================================== #
# Stage 1 — Simplify the corpus                                               #
# =========================================================================== #
KEEP_ENTITIES = False  # flip to True to also carry entity_concepts / entity_sub_entities

CLAUSE_KEYS = [
    "type", "source_file", "text_id", "clause_number",
    "clause_title", "merged_clause_number", "resolved_text",
]
METADATA_KEYS = [
    "type", "source_file", "merged_metadata_number", "text_id",
    "metadata_category", "metadata_title", "resolved_text",
]


def store_id_for(item: dict) -> str | None:
    if item.get("type") == "clause":
        return item.get("text_id")
    if item.get("type") == "metadata":
        return item.get("merged_metadata_number")
    return None


def simplify(item: dict, source_file: str) -> dict | None:
    itype = item.get("type")
    keys = CLAUSE_KEYS if itype == "clause" else METADATA_KEYS if itype == "metadata" else None
    if keys is None:
        return None
    out = {"store_id": store_id_for(item)}
    for k in keys:
        out[k] = source_file if k == "source_file" else item.get(k)
    if KEEP_ENTITIES:
        out["entity_concepts"] = item.get("entity_concepts")
        out["entity_sub_entities"] = item.get("entity_sub_entities")
    return out


def stage_simplify() -> int:
    if not config.SOURCE_JSON.exists():
        print(f"ERROR: {config.SOURCE_JSON} not found.")
        return 1

    with open(config.SOURCE_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    records, skipped_ids = [], 0
    seen_ids: set[str] = set()
    for doc in data.get("merged_documents", []):
        source_file = doc.get("source_file", "")
        for item in doc.get("processed_clauses", []):
            rec = simplify(item, source_file)
            if rec is None:
                continue
            if not rec["store_id"]:
                skipped_ids += 1
                continue
            if rec["store_id"] in seen_ids:
                # Should not happen (IDs verified unique), but guard anyway.
                print(f"WARNING: duplicate store_id '{rec['store_id']}' — keeping first only.")
                continue
            seen_ids.add(rec["store_id"])
            records.append(rec)

    save_json(config.SIMPLIFIED_JSON, records)

    n_clause = sum(1 for r in records if r["type"] == "clause")
    n_meta = sum(1 for r in records if r["type"] == "metadata")
    src_mb = config.SOURCE_JSON.stat().st_size / 1e6
    out_mb = config.SIMPLIFIED_JSON.stat().st_size / 1e6
    print(f"Wrote {len(records)} records  ({n_clause} clause, {n_meta} metadata)")
    if skipped_ids:
        print(f"  ({skipped_ids} items skipped for missing store_id)")
    print(f"  {config.SOURCE_JSON.name}: {src_mb:.1f} MB  ->  "
          f"{config.SIMPLIFIED_JSON.name}: {out_mb:.1f} MB")
    return 0


# =========================================================================== #
# Stage 2 — Build the embeddings                                             #
# =========================================================================== #
def stage_embed() -> int:
    if not config.SIMPLIFIED_JSON.exists():
        print(f"ERROR: {config.SIMPLIFIED_JSON} not found. Enable RUN_SIMPLIFY first.")
        return 1

    records = load_json(config.SIMPLIFIED_JSON, [])
    print(f"Loaded {len(records)} records from {config.SIMPLIFIED_JSON.name}.")

    # --- Vector store --------------------------------------------------------
    collection = open_collection(create=True)

    # --- Resumability: figure out what's already done ------------------------
    done_ids = load_done_ids(collection, config.PROGRESS_FILE)

    skipped = load_json(config.SKIPPED_JSON, [])
    skipped_ids = {s["store_id"] for s in skipped}
    failed = load_json(config.FAILED_JSON, [])
    failed_ids = {f["store_id"] for f in failed}

    pending = [
        r for r in records
        if r["store_id"] not in done_ids
        and r["store_id"] not in skipped_ids
        and r["store_id"] not in failed_ids
    ]
    if EMBED_LIMIT:
        pending = pending[:EMBED_LIMIT]

    print(f"Already embedded: {len(done_ids)} | previously skipped(long): {len(skipped_ids)} "
          f"| previously failed: {len(failed_ids)}")
    print(f"To process this run: {len(pending)}")
    if not pending:
        print("Nothing to do. Collection count:", collection.count())
        return 0

    # --- Keys ----------------------------------------------------------------
    try:
        km = KeyManager.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Loaded {len(km.keys)} Gemini API key(s).\n")

    try:
        from tqdm import tqdm
        bar = tqdm(pending, unit="item")
    except ImportError:
        bar = pending

    embedded_this_run = 0
    for i, rec in enumerate(bar, 1):
        sid = rec["store_id"]
        text = (rec.get("resolved_text") or "").strip()

        if not text:
            failed.append({"store_id": sid, "type": rec.get("type"),
                           "source_file": rec.get("source_file"), "reason": "empty resolved_text"})
            failed_ids.add(sid)
            continue

        # Cheap guard: payloads this large are guaranteed over the token limit.
        if len(text) > config.PRESKIP_CHARS:
            skipped.append({"store_id": sid, "type": rec.get("type"),
                            "source_file": rec.get("source_file"), "char_len": len(text),
                            "reason": "pre-skip (char length over threshold)"})
            skipped_ids.add(sid)
            save_json(config.SKIPPED_JSON, skipped)
            continue

        try:
            vec = embed_text(km, text, config.DOC_TASK_TYPE)
        except TooLongError:
            skipped.append({"store_id": sid, "type": rec.get("type"),
                            "source_file": rec.get("source_file"), "char_len": len(text),
                            "reason": "over 2048-token model limit"})
            skipped_ids.add(sid)
            save_json(config.SKIPPED_JSON, skipped)
            continue
        except AllKeysExhausted as exc:
            print(f"\n{exc}\nStopping cleanly — re-run later to continue.")
            break
        except Exception as exc:  # noqa: BLE001
            failed.append({"store_id": sid, "type": rec.get("type"),
                           "source_file": rec.get("source_file"), "reason": str(exc)[:300]})
            failed_ids.add(sid)
            save_json(config.FAILED_JSON, failed)
            continue

        collection.upsert(
            ids=[sid],
            embeddings=[vec],
            documents=[text],
            metadatas=[metadata_for(rec)],
        )
        done_ids.add(sid)
        embedded_this_run += 1

        # Persist progress periodically so a crash never loses much.
        if embedded_this_run % 50 == 0:
            save_json(config.PROGRESS_FILE, sorted(done_ids))
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(km.status_line())

    # --- Final flush ---------------------------------------------------------
    save_json(config.PROGRESS_FILE, sorted(done_ids))
    save_json(config.SKIPPED_JSON, skipped)
    save_json(config.FAILED_JSON, failed)

    print(f"\nDone this run: embedded {embedded_this_run} item(s).")
    print(f"Collection '{config.COLLECTION_NAME}' now holds {collection.count()} vectors "
          f"({config.EMBED_DIM}-dim).")
    print(f"Skipped (long, embed later): {len(skipped)}  -> {config.SKIPPED_JSON.name}")
    if failed:
        print(f"Failed (other errors): {len(failed)}  -> {config.FAILED_JSON.name}")
    return 0


# =========================================================================== #
# Stage 3 — Chunk + embed the long clauses                                   #
# =========================================================================== #
def stage_embed_long() -> int:
    if not config.SIMPLIFIED_JSON.exists():
        print(f"ERROR: {config.SIMPLIFIED_JSON} not found. Enable RUN_SIMPLIFY first.")
        return 1

    skipped = load_json(config.SKIPPED_JSON, [])
    if not skipped:
        print(f"No entries in {config.SKIPPED_JSON.name} — nothing to chunk.")
        return 0

    records = load_json(config.SIMPLIFIED_JSON, [])
    by_id = {r["store_id"]: r for r in records}
    print(f"Loaded {len(records)} corpus records; {len(skipped)} skipped long item(s) to process.")

    # --- Vector store (SAME collection as the build stage) -------------------
    collection = open_collection(create=True)
    count_before = collection.count()

    # --- Resumability: chunk ids already done --------------------------------
    done = load_done_ids(collection, config.CHUNK_PROGRESS_FILE)

    parents = skipped[:LONG_LIMIT] if LONG_LIMIT else skipped

    # --- Keys ----------------------------------------------------------------
    try:
        km = KeyManager.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Loaded {len(km.keys)} Gemini API key(s).\n")

    try:
        from tqdm import tqdm
        bar = tqdm(parents, unit="clause")
    except ImportError:
        bar = parents

    embedded_this_run = 0
    misses = 0
    try:
        for parent_entry in bar:
            sid = parent_entry["store_id"]
            rec = by_id.get(sid)
            if rec is None:
                print(f"\n  WARN: {sid} not found in {config.SIMPLIFIED_JSON.name} — skipping.")
                misses += 1
                continue

            text = (rec.get("resolved_text") or "").strip()
            if not text:
                print(f"\n  WARN: {sid} has empty resolved_text — skipping.")
                misses += 1
                continue

            chunks = chunk_text(text)
            n = len(chunks)
            for i, chunk in enumerate(chunks):
                base = f"{sid}#{i}"
                if base in done:
                    continue  # already embedded on a previous run
                try:
                    for suffix, piece, vec in embed_with_fallback(
                            km, chunk, config.CHUNK_TARGET_CHARS):
                        cid = base + suffix
                        collection.upsert(
                            ids=[cid],
                            embeddings=[vec],
                            documents=[piece],
                            metadatas=[chunk_metadata(rec, i, n)],
                        )
                        done.add(cid)
                        embedded_this_run += 1
                        if embedded_this_run % 50 == 0:
                            save_json(config.CHUNK_PROGRESS_FILE, sorted(done))
                            if hasattr(bar, "set_postfix_str"):
                                bar.set_postfix_str(km.status_line())
                except AllKeysExhausted as exc:
                    print(f"\n{exc}\nStopping cleanly — re-run later to continue.")
                    return _finish_long(collection, done, count_before, embedded_this_run, misses)
    finally:
        save_json(config.CHUNK_PROGRESS_FILE, sorted(done))

    return _finish_long(collection, done, count_before, embedded_this_run, misses)


def _finish_long(collection, done, count_before, embedded_this_run, misses) -> int:
    save_json(config.CHUNK_PROGRESS_FILE, sorted(done))
    count_now = collection.count()
    print(f"\nDone this run: embedded {embedded_this_run} chunk(s).")
    print(f"Collection '{config.COLLECTION_NAME}' grew {count_before} -> {count_now} vectors "
          f"(+{count_now - count_before}).")
    if misses:
        print(f"Skipped {misses} entry(ies) that could not be joined / had no text.")
    return 0


# =========================================================================== #
# Stage 4 — Search                                                           #
# =========================================================================== #
def stage_search() -> int:
    try:
        collection = open_collection(create=False)
    except Exception:
        print(f"ERROR: collection '{config.COLLECTION_NAME}' not found. "
              "Enable RUN_EMBED first.")
        return 1
    if collection.count() == 0:
        print("Collection is empty — nothing to search yet.")
        return 1

    try:
        km = KeyManager.load()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    qvec = embed_text(km, QUERY, config.QUERY_TASK_TYPE)
    # Over-fetch, then keep only the best-ranked hit per parent clause. Long clauses are
    # stored as several chunks (id `<store_id>#<i>`, metadata `parent_id`); without this a
    # single long clause could occupy multiple slots and out-rank normal single-vector
    # clauses. Deduping by parent keeps every clause to at most one slot, so the chunked
    # long clauses retrieve on par with the rest of the corpus.
    over_fetch = min(TOP_K * 4, collection.count())
    res = collection.query(
        query_embeddings=[qvec],
        n_results=over_fetch,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    seen: set[str] = set()
    hits = []
    for sid, doc, md, dist in zip(ids, docs, metas, dists):
        parent = md.get("parent_id") or sid  # normal clauses have no parent_id
        if parent in seen:
            continue
        seen.add(parent)
        hits.append((sid, doc, md, dist))
        if len(hits) >= TOP_K:
            break

    # Parent expansion: a chunk earns its slot on its own similarity, but the
    # result carries the parent clause's full text. The giants were never
    # stored whole in ChromaDB, so the corpus JSON is the only full-text source.
    parent_texts: dict[str, str] = {}
    if EXPAND_CHUNKS_TO_PARENT and any(md.get("parent_id") for _, _, md, _ in hits):
        if config.SIMPLIFIED_JSON.exists():
            parent_texts = {r["store_id"]: r.get("resolved_text") or ""
                            for r in load_json(config.SIMPLIFIED_JSON, [])}
        else:
            print(f"WARN: {config.SIMPLIFIED_JSON.name} missing — "
                  "chunk hits shown as chunks, not full clauses.")

    print(f"\nTop {len(hits)} results for: {QUERY!r}\n" + "=" * 70)
    for rank, (sid, doc, md, dist) in enumerate(hits, 1):
        sim = 1 - dist  # cosine distance -> similarity
        chunk_tag = ""
        if md.get("parent_id"):  # this hit is a chunk of a long clause
            chunk_tag = f"  chunk {md.get('chunk_index')}/{md.get('n_chunks')}"
            full = parent_texts.get(md["parent_id"])
            if full:
                doc = full
                chunk_tag += f" -> full clause ({len(full):,} chars)"
        snippet = " ".join(doc.split())[:220]
        print(f"\n#{rank}  sim={sim:.3f}  [{md.get('type')}]  id={sid}{chunk_tag}")
        print(f"    title : {md.get('title')}")
        print(f"    source: {md.get('source_file')}")
        print(f"    text  : {snippet}...")
    print()
    return 0


# =========================================================================== #
# Driver                                                                     #
# =========================================================================== #
def main() -> int:
    rc = 0
    if RUN_SIMPLIFY:
        print("\n=== Stage 1: simplify ===")
        rc = stage_simplify() or rc
    if RUN_EMBED:
        print("\n=== Stage 2: build embeddings ===")
        rc = stage_embed() or rc
    if RUN_LONG:
        print("\n=== Stage 3: chunk + embed long items ===")
        rc = stage_embed_long() or rc
    if RUN_SEARCH:
        print("\n=== Stage 4: search ===")
        rc = stage_search() or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
