"""Shared, side-effect-free helpers for the RAG pipeline.

Everything that used to be duplicated across build_embeddings.py / embed_long_items.py /
search.py lives here now: JSON load/save, Chroma metadata building, the single embedding
call, the collection factory, the resume-set loader, and the long-item chunking logic.

Heavy SDKs (google.genai, chromadb) are imported lazily inside the functions that need
them so crypto-only / dry-run paths don't require them installed.
"""
from __future__ import annotations

import json
import sys as _sys
from pathlib import Path as _Path

for _p in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import config
from key_manager import KeyManager, TooLongError


# --------------------------------------------------------------------------- #
# Small JSON helpers                                                          #
# --------------------------------------------------------------------------- #
def load_json(path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)  # auto-create data/ state/ logs/
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic on the same filesystem


# --------------------------------------------------------------------------- #
# Chroma metadata                                                             #
# --------------------------------------------------------------------------- #
def title_of(rec: dict) -> str:
    return rec.get("clause_title") or rec.get("metadata_title") or ""


def metadata_for(rec: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool (no None)."""
    md = {
        "type": rec.get("type") or "",
        "source_file": rec.get("source_file") or "",
        "title": title_of(rec),
        "store_id": rec.get("store_id") or "",
    }
    if rec.get("type") == "clause":
        md["clause_number"] = rec.get("clause_number") or ""
        md["merged_clause_number"] = rec.get("merged_clause_number") or ""
    else:
        md["metadata_category"] = rec.get("metadata_category") or ""
    return md


# --------------------------------------------------------------------------- #
# Embedding                                                                   #
# --------------------------------------------------------------------------- #
def embed_text(km: KeyManager, text: str, task_type: str) -> list[float]:
    """Embed a single text, rotating keys under the hood. Raises TooLongError if over limit.

    `task_type` is config.DOC_TASK_TYPE for corpus documents / chunks and
    config.QUERY_TASK_TYPE for search queries.
    """
    from google.genai import types

    def _call(client):
        resp = client.models.embed_content(
            model=config.EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=config.EMBED_DIM,
            ),
        )
        return resp.embeddings[0].values

    return km.call(_call)


# --------------------------------------------------------------------------- #
# Vector store                                                                #
# --------------------------------------------------------------------------- #
def open_collection(create: bool = True):
    """Open the ChromaDB collection. create=True gets-or-creates it (for building);
    create=False opens it read-only (for search) and raises if it doesn't exist yet."""
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if create:
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL,
                      "dim": config.EMBED_DIM},
        )
    return client.get_collection(config.COLLECTION_NAME)


def load_done_ids(collection, progress_file) -> set[str]:
    """Ids already embedded: the progress file's set unioned with the ids already in Chroma."""
    done: set[str] = set(load_json(progress_file, []))
    try:
        existing = collection.get(include=[])  # ids only
        done.update(existing.get("ids", []))
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read existing ids from Chroma: {exc})")
    return done


# --------------------------------------------------------------------------- #
# Long-item chunking                                                          #
# --------------------------------------------------------------------------- #
_DELIMS = "।.!?\n"  # Bangla danda (U+0964) + Latin sentence enders + newline


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping the delimiter attached to each sentence."""
    out, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch in _DELIMS:
            seg = "".join(cur).strip()
            if seg:
                out.append(seg)
            cur = []
    seg = "".join(cur).strip()
    if seg:
        out.append(seg)
    return out


def hard_split(s: str, target: int) -> list[str]:
    """Last-resort split of an over-long span at whitespace boundaries (never mid-word
    unless the span has no whitespace at all)."""
    s = s.strip()
    pieces = []
    while len(s) > target:
        cut = s.rfind(" ", 0, target + 1)
        if cut <= 0:
            cut = target  # no usable whitespace — accept a hard cut
        pieces.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        pieces.append(s)
    return pieces


def tail_overlap(s: str, overlap: int) -> str:
    """Return roughly the last `overlap` chars of `s`, advanced to a whitespace boundary
    so the carried-over context never begins mid-word."""
    if overlap <= 0 or len(s) <= overlap:
        return s.strip()
    tail = s[-overlap:]
    sp = tail.find(" ")
    if sp != -1:
        tail = tail[sp + 1:]
    return tail.strip()


def chunk_text(text: str, target: int | None = None, overlap: int | None = None,
               min_tail: int | None = None) -> list[str]:
    """Split `text` into overlapping chunks of ~`target` chars on natural boundaries."""
    target = target if target is not None else config.CHUNK_TARGET_CHARS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP_CHARS
    min_tail = min_tail if min_tail is not None else config.CHUNK_MIN_TAIL_CHARS
    overlap = min(overlap, max(0, target - 1))  # overlap must be < target

    text = (text or "").strip()
    if not text:
        return []

    # 1. Break into atoms no larger than target: paragraph -> sentence -> hard split.
    atoms: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target:
            atoms.append(para)
            continue
        for sent in split_sentences(para):
            if len(sent) <= target:
                atoms.append(sent)
            else:
                atoms.extend(hard_split(sent, target))

    # 2. Greedily pack atoms up to target, seeding each new chunk with a word-aligned tail.
    chunks: list[str] = []
    cur = ""
    for a in atoms:
        if cur and len(cur) + 1 + len(a) > target:
            chunks.append(cur)
            seed = tail_overlap(cur, overlap)
            cur = (seed + "\n" + a) if seed else a
        else:
            cur = (cur + "\n" + a) if cur else a
    if cur:
        chunks.append(cur)

    # 3. Fold a tiny final remainder into the previous chunk (no stub chunks).
    if len(chunks) >= 2 and len(chunks[-1]) < min_tail:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + "\n" + tail

    return chunks


def chunk_metadata(parent: dict, i: int, n: int) -> dict:
    """Parent clause metadata (so a chunk hit looks like a normal clause) + chunk fields."""
    md = metadata_for(parent)  # already Chroma-safe (str/int/float/bool, no None)
    md["parent_id"] = parent.get("store_id") or ""
    md["chunk_index"] = int(i)
    md["n_chunks"] = int(n)
    return md


def embed_with_fallback(km: KeyManager, text: str, target: int, suffix: str = ""):
    """Yield (suffix, text, vector). Chunks are sized well under the limit, but if the API
    still reports a piece as too long, re-split it (halving the target) and embed the parts."""
    try:
        vec = embed_text(km, text, config.DOC_TASK_TYPE)
        yield suffix, text, vec
        return
    except TooLongError:
        sub_target = max(200, target // 2)
        subs = chunk_text(text, target=sub_target,
                          overlap=min(config.CHUNK_OVERLAP_CHARS, sub_target // 4),
                          min_tail=0)
        if len(subs) <= 1:
            raise  # cannot split any further
        for j, s in enumerate(subs):
            yield from embed_with_fallback(km, s, sub_target, f"{suffix}.{j}")
