#!/usr/bin/env python3
"""
cluster_entities.py
════════════════════════════════════════════════════════════════════════════════
Cluster the entities of `merged_ec_output_canon_subentities.json` by meaning, then
cluster the *clauses* by the entity-clusters they touch.

Universe of "entities" = every `entity_concepts` key  ∪  every `entity_sub_entities`
term (the task treats sub-entities as entities too).

Pipeline (each stage checkpoints to disk; re-running skips finished stages):

  1. collect  — gather every unique entity text + a few representative concept
                snippets                                  →  entities_index.json
  2. embed    — gemini-embedding-001 over "text + concept"  →  entity_embeddings.npy
  3. block    — cosine ≥ threshold → candidate graph → connected components
                (the only pairs ever shown to Gemini)     →  entity_blocks.json
  4. judge    — Gemini decides, per block, which entities are the SAME referent
                (strict). Union-find → global entity clusters
                                                          →  entity_clusters.json
  5. assign   — each clause gets `clause_clusters: [ids]` = the set of entity-cluster
                ids of all entities it contains            →  *_clustered.json

Approach: embeddings only generate *candidates*; Gemini is the equivalence judge,
using both the entity text and its concept. Merge policy is STRICT (same referent).

Usage:
    python cluster_entities.py                 # run all stages (resume-aware)
    python cluster_entities.py --stage block   # run a single stage
    python cluster_entities.py --force         # ignore checkpoints, redo everything
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from tqdm import tqdm

# Reuse the battle-tested key rotator + helpers from the sub-entity pipeline.
from break_subentities import (
    GEMINI_MODEL,
    ApiKeyRotator,
    _call_with_watchdog,
    generate_with_rotation,
    setup_logging,
)

logger = logging.getLogger("subentity_pipeline")  # share the rotator's logger

# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "merged_ec_output_canon_subentities.json"
OUTPUT_FILE = SCRIPT_DIR / "merged_ec_output_canon_clustered.json"

# Paid key (has billing credit) — used for embeddings, which the free keys have
# exhausted their daily quota on. Free keys (separate generateContent quota) are
# added on top for the judge stage to parallelise across more keys.
EMBED_KEYS_FILE = SCRIPT_DIR / "gemini_api.txt"
FREE_KEYS_FILE = SCRIPT_DIR / "all_api.txt"

ENTITIES_INDEX = SCRIPT_DIR / "entities_index.json"
EMBEDDINGS_FILE = SCRIPT_DIR / "entity_embeddings.npy"
BLOCKS_FILE = SCRIPT_DIR / "entity_blocks.json"
CLUSTERS_FILE = SCRIPT_DIR / "entity_clusters.json"
JUDGE_CKPT = SCRIPT_DIR / "entity_judge_ckpt.json"

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_BATCH = 100                 # entities per embed_content call (API hard max)
EMBED_DELAY = 0.2                 # pacing between embed calls (s)
MAX_CONCEPTS = 3                  # representative concept snippets kept per entity
CONCEPT_CHARS = 240              # max chars of a concept snippet used for embed/judge

SIM_THRESHOLD = 0.92             # cosine ≥ this → candidate pair (Gemini decides)
TOP_K = 12                       # max candidate neighbours kept per entity
MAX_BLOCK = 24                   # max entities shown to Gemini in one judge call
SIM_CHUNK = 1024                 # rows per cosine matmul chunk

REQUEST_TIMEOUT = 90
EMBED_WATCHDOG = 90
JUDGE_WATCHDOG = 90
CONCURRENCY = 16                 # parallel judge workers (≤ number of keys)


# ════════════════════════════════════════════════════════════════════════════════
# Stage 1 — collect unique entities + representative concepts
# ════════════════════════════════════════════════════════════════════════════════

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def stage_collect(force: bool = False) -> dict:
    if ENTITIES_INDEX.exists() and not force:
        logger.info(f"[collect] reusing {ENTITIES_INDEX.name}")
        return json.loads(ENTITIES_INDEX.read_text(encoding="utf-8"))

    logger.info(f"[collect] reading {INPUT_FILE.name}")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    # key = casefolded text; value accumulates display-casing votes + concepts
    display_votes: dict[str, Counter] = defaultdict(Counter)
    concepts: dict[str, list[str]] = defaultdict(list)
    is_key: dict[str, bool] = defaultdict(bool)
    freq: Counter = Counter()

    def add_concept(norm: str, text: str) -> None:
        text = _clean(text)[:CONCEPT_CHARS]
        if text and text not in concepts[norm] and len(concepts[norm]) < MAX_CONCEPTS:
            concepts[norm].append(text)

    for doc in data["merged_documents"]:
        for elem in doc["processed_clauses"]:
            ec = elem.get("entity_concepts") or {}
            se = elem.get("entity_sub_entities") or {}
            for ent, cons in ec.items():
                norm = _clean(ent).casefold()
                if not norm:
                    continue
                display_votes[norm][_clean(ent)] += 1
                freq[norm] += 1
                is_key[norm] = True
                for c in (cons or []):
                    add_concept(norm, c)
            # sub-entities: borrow the parent entity's concept as context
            for parent, subs in se.items():
                parent_cons = ec.get(parent) or []
                ctx = parent_cons[0] if parent_cons else ""
                for s in (subs or []):
                    norm = _clean(s).casefold()
                    if not norm:
                        continue
                    display_votes[norm][_clean(s)] += 1
                    freq[norm] += 1
                    if ctx:
                        add_concept(norm, f"(as part of '{_clean(parent)}') {ctx}")

    entities = []
    for norm in display_votes:
        display = display_votes[norm].most_common(1)[0][0]
        entities.append({
            "id": 0,  # filled below after stable sort
            "norm": norm,
            "text": display,
            "concepts": concepts.get(norm, []),
            "is_entity_key": is_key[norm],
            "freq": freq[norm],
        })
    # stable, deterministic order
    entities.sort(key=lambda e: e["norm"])
    for i, e in enumerate(entities):
        e["id"] = i

    index = {"count": len(entities), "entities": entities}
    ENTITIES_INDEX.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[collect] {len(entities)} unique entities → {ENTITIES_INDEX.name}")
    return index


# ════════════════════════════════════════════════════════════════════════════════
# Stage 2 — embeddings
# ════════════════════════════════════════════════════════════════════════════════

def _embed_text(ent: dict) -> str:
    parts = [ent["text"]]
    if ent["concepts"]:
        parts.append("Concept: " + " | ".join(ent["concepts"][:2]))
    return " ".join(parts)[:2000]


def _embed_batch_with_rotation(rotator: ApiKeyRotator, texts: list[str]) -> list[list[float]] | None:
    cfg = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY",
                                   output_dimensionality=EMBED_DIM)
    while True:
        if not rotator.has_any_usable_key():
            return None
        key = rotator.current_key
        if (key in rotator.blacklisted_keys or key in rotator.exhausted_keys
                or time.time() < rotator.cooldown_until.get(key, 0)
                or rotator.usage_count.get(key, 0) >= rotator.calls_per_day):
            if not rotator.rotate_key():
                wait = rotator.wait_for_cooldown_expiry()
                if wait <= 0:
                    return None
                time.sleep(wait + 1)
            continue
        try:
            client = rotator.get_client()
            resp = _call_with_watchdog(
                lambda: client.models.embed_content(model=EMBED_MODEL, contents=texts, config=cfg),
                EMBED_WATCHDOG)
            rotator.report_success()
            return [e.values for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            etype = rotator.report_error(e)
            if etype == "transient":
                time.sleep(3)
            if not rotator.rotate_key():
                wait = rotator.wait_for_cooldown_expiry()
                if wait <= 0:
                    return None
                time.sleep(wait + 1)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    """Write via a temp file + rename so a kill mid-write can never corrupt an
    existing good embeddings file (which is exactly what bit us once). Note np.save
    appends '.npy' if the path lacks it, so we name the temp accordingly."""
    tmp = path.with_name(path.stem + ".tmp.npy")   # e.g. entity_embeddings.tmp.npy
    np.save(tmp, arr)
    os.replace(tmp, path)


def _embeddings_cache_healthy(n: int) -> bool:
    """True iff the embeddings cache exists with the right shape and no zero rows."""
    if not EMBEDDINGS_FILE.exists():
        return False
    arr = np.load(EMBEDDINGS_FILE, mmap_mode="r")
    if arr.shape[0] != n:
        return False
    return int((np.linalg.norm(np.asarray(arr), axis=1) < 1e-6).sum()) == 0


def stage_embed(index: dict, rotator: ApiKeyRotator | None, force: bool = False) -> np.ndarray:
    n = index["count"]
    if EMBEDDINGS_FILE.exists() and not force:
        arr = np.load(EMBEDDINGS_FILE)
        zero_rows = int((np.linalg.norm(arr, axis=1) < 1e-6).sum()) if arr.size else n
        if arr.shape[0] == n and zero_rows == 0:
            logger.info(f"[embed] reusing {EMBEDDINGS_FILE.name} ({arr.shape}, healthy)")
            return arr
        logger.warning(f"[embed] cached embeddings unusable "
                       f"(shape {arr.shape}, {zero_rows} zero rows) — recomputing")
    if rotator is None:
        raise SystemExit("[embed] cache missing/unhealthy but no embed key available "
                         "(paid key excluded). Restore gemini_api.txt to re-embed.")

    ents = index["entities"]
    texts = [_embed_text(e) for e in ents]
    # accumulate into a separate buffer; never overwrite the live cache until the
    # full, normalized result is ready (atomic rename at the end).
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    done = 0
    for start in tqdm(range(0, n, EMBED_BATCH), desc="embed", unit="batch"):
        batch = texts[start:start + EMBED_BATCH]
        vecs = _embed_batch_with_rotation(rotator, batch)
        if vecs is None:
            logger.error("[embed] no usable keys; aborting WITHOUT touching cache")
            raise SystemExit(1)
        embs[start:start + len(vecs)] = np.asarray(vecs, dtype=np.float32)
        done += len(vecs)
        if EMBED_DELAY:
            time.sleep(EMBED_DELAY)
    # L2-normalize for cosine via dot product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    _save_npy_atomic(EMBEDDINGS_FILE, embs)
    logger.info(f"[embed] embedded {done} entities → {EMBEDDINGS_FILE.name}")
    return embs


# ════════════════════════════════════════════════════════════════════════════════
# Stage 3 — candidate judge-units via cosine neighbourhoods + greedy set cover
# ════════════════════════════════════════════════════════════════════════════════
#
# A low cosine threshold on this (compressed) embedding space transitively chains
# nearly everything into one giant component, so we do NOT pre-merge candidates.
# Instead each entity contributes a small local neighbourhood (its top-K above-
# threshold neighbours); greedy set cover turns those into bounded "judge units"
# such that every candidate pair lands in at least one unit. Gemini then decides
# equivalence within each unit, and only Gemini-confirmed pairs are ever unioned.

class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def stage_block(index: dict, embs: np.ndarray, force: bool = False) -> dict:
    if BLOCKS_FILE.exists() and not force:
        logger.info(f"[block] reusing {BLOCKS_FILE.name}")
        return json.loads(BLOCKS_FILE.read_text(encoding="utf-8"))

    n = embs.shape[0]
    logger.info(f"[block] cosine candidate search (threshold {SIM_THRESHOLD}, top-{TOP_K})")
    neigh: dict[int, list[int]] = {}               # node → neighbours, strongest first
    for start in tqdm(range(0, n, SIM_CHUNK), desc="block", unit="chunk"):
        end = min(start + SIM_CHUNK, n)
        sims = embs[start:end] @ embs.T            # (chunk, n)
        for i in range(end - start):
            row = sims[i]
            gi = start + i
            row[gi] = -1.0                          # ignore self
            cand = np.where(row >= SIM_THRESHOLD)[0]
            if cand.size > TOP_K:
                cand = cand[np.argpartition(row[cand], -TOP_K)[-TOP_K:]]
            order = cand[np.argsort(-row[cand])]
            neigh[gi] = [int(j) for j in order]

    # Greedy set cover: seed highest-degree nodes first, pull in their still-
    # uncovered neighbours, mark all in-unit pairs covered. Bounded by MAX_BLOCK.
    covered: set[tuple[int, int]] = set()
    units: list[list[int]] = []
    for i in sorted(range(n), key=lambda x: -len(neigh[x])):
        un = [j for j in neigh[i] if (min(i, j), max(i, j)) not in covered]
        if not un:
            continue
        unit = [i] + un[:MAX_BLOCK - 1]
        for a in range(len(unit)):
            for b in range(a + 1, len(unit)):
                covered.add((min(unit[a], unit[b]), max(unit[a], unit[b])))
        units.append(unit)

    in_units = len({x for u in units for x in u})
    sizes = [len(u) for u in units]
    logger.info(f"[block] {len(units)} judge units; entities covered {in_units}/{n}; "
                f"unit size avg={np.mean(sizes):.1f} max={max(sizes) if sizes else 0}")
    out = {"threshold": SIM_THRESHOLD, "units": units}
    BLOCKS_FILE.write_text(json.dumps(out), encoding="utf-8")
    logger.info(f"[block] → {BLOCKS_FILE.name}")
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Stage 4 — Gemini equivalence judging per block
# ════════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """You are disambiguating entities extracted from Bangladeshi customs/tax legal documents.

Below is a numbered list of candidate entities. Each has its TEXT and one or more
CONCEPT notes describing what it refers to in context.

Decide which entities refer to the EXACT SAME real-world referent (STRICT).
- Merge only true equivalents: abbreviations/aliases of the same thing
  (e.g. "NBR" = "National Board of Revenue"), singular/plural or spelling variants
  of the same term (e.g. "notification" = "notifications"), and the same named
  act/section/body written differently.
- Do NOT merge things that are merely related, the same category, or a part/whole
  (e.g. "goods" ≠ "imported goods" ≠ "raw materials"; "Customs Act 1969" ≠
  "Customs Act 2023"; "duty" ≠ "customs duty" only if they denote different scopes).
- When unsure, keep them SEPARATE.

Entities:
{listing}

Return STRICT JSON: {{"groups": [[ids of one referent], [ids of another], ...]}}.
Every id from 0..{maxid} must appear in exactly one group. Singletons are groups of one.
Return only the JSON object.
"""


JUDGE_CFG = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)


def _build_judge_prompt(ents: list[dict]) -> str:
    lines = []
    for i, e in enumerate(ents):
        c = " | ".join(e["concepts"][:2]) if e["concepts"] else "(no concept)"
        lines.append(f'{i}. TEXT: "{e["text"]}"  CONCEPT: {c[:CONCEPT_CHARS]}')
    return JUDGE_PROMPT.format(listing="\n".join(lines), maxid=len(ents) - 1)


def _judge_block(rotator: ApiKeyRotator, ents: list[dict]) -> list[list[int]]:
    """Sequential single-rotator judge (used by the validation harness)."""
    text = generate_with_rotation(rotator, _build_judge_prompt(ents))
    groups = _parse_groups(text, len(ents))
    if groups is None:
        return [[i] for i in range(len(ents))]   # all-separate is safe under strict
    return groups


# ── thread-safe key pool for parallel judging ──────────────────────────────────

class KeyPool:
    """Hand out one key at a time per worker; park rate-limited keys on a cooldown,
    drop permanently-dead ones. Each key is used by at most one in-flight call,
    which keeps per-key RPM in check while running CONCURRENCY calls at once."""

    def __init__(self, keys: list[str]):
        self.keys = list(dict.fromkeys(keys))           # dedupe, preserve order
        self.clients: dict[str, genai.Client] = {}
        self.avail: collections.deque[str] = collections.deque(self.keys)
        self.cooldown: dict[str, float] = {}
        self.dead: set[str] = set()
        self.cv = threading.Condition()

    def client(self, k: str) -> genai.Client:
        c = self.clients.get(k)
        if c is None:
            c = genai.Client(api_key=k,
                             http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000))
            self.clients[k] = c
        return c

    def acquire(self) -> str | None:
        with self.cv:
            while True:
                now = time.time()
                for _ in range(len(self.avail)):
                    k = self.avail.popleft()
                    if k in self.dead:
                        continue
                    if self.cooldown.get(k, 0) > now:
                        self.avail.append(k)
                        continue
                    return k
                if len(self.dead) >= len(self.keys):
                    return None                          # every key permanently dead
                waits = [self.cooldown[k] - now for k in self.keys
                         if k not in self.dead and self.cooldown.get(k, 0) > now]
                self.cv.wait(timeout=min([w for w in waits if w > 0] + [2.0]))

    def release(self, k: str, cooldown: float = 0.0, dead: bool = False) -> None:
        with self.cv:
            if dead:
                self.dead.add(k)
            else:
                if cooldown > 0:
                    self.cooldown[k] = time.time() + cooldown
                self.avail.append(k)
            self.cv.notify_all()


def _classify_err(error: Exception) -> str:
    """Reuse the rotator's three-way error taxonomy (method is effectively static)."""
    return ApiKeyRotator.classify_error_type(None, error)  # type: ignore[arg-type]


def _judge_unit_pool(pool: KeyPool, ents: list[dict]) -> list[list[int]]:
    """Thread-safe judge of one unit using a key from the pool. Returns groups of
    local indices; falls back to all-separate (strict-safe) on repeated failure."""
    prompt = _build_judge_prompt(ents)
    n = len(ents)
    for _ in range(6):
        k = pool.acquire()
        if k is None:
            break
        try:
            resp = _call_with_watchdog(
                lambda: pool.client(k).models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=JUDGE_CFG),
                JUDGE_WATCHDOG)
            pool.release(k)
            groups = _parse_groups(resp.text, n)
            return groups if groups is not None else [[i] for i in range(n)]
        except Exception as e:  # noqa: BLE001
            etype = _classify_err(e)
            if etype == "permanent":
                pool.release(k, dead=True)
            elif etype == "rate_limit":
                delay = ApiKeyRotator._extract_retry_delay(e)
                pool.release(k, cooldown=max(delay, 30))
            else:
                pool.release(k, cooldown=3)
    return [[i] for i in range(n)]


def _parse_groups(text: str, n: int) -> list[list[int]] | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list):
        return None
    seen, clean = set(), []
    for g in groups:
        if not isinstance(g, list):
            continue
        ids = []
        for x in g:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= xi < n and xi not in seen:
                seen.add(xi)
                ids.append(xi)
        if ids:
            clean.append(ids)
    # any ids the model dropped become their own singleton groups
    for i in range(n):
        if i not in seen:
            clean.append([i])
    return clean


def stage_judge(index: dict, embs: np.ndarray, blocks_data: dict,
                rotator: ApiKeyRotator, force: bool = False) -> dict:
    if CLUSTERS_FILE.exists() and not force:
        logger.info(f"[judge] reusing {CLUSTERS_FILE.name}")
        return json.loads(CLUSTERS_FILE.read_text(encoding="utf-8"))

    ents = index["entities"]
    n = index["count"]
    dsu = _DSU(n)

    judge_units: list[list[int]] = blocks_data["units"]

    # resume support: replay confirmed pairs, skip already-judged unit indices
    confirmed_pairs: list[list[int]] = []
    done: set[int] = set()
    if JUDGE_CKPT.exists() and not force:
        ck = json.loads(JUDGE_CKPT.read_text(encoding="utf-8"))
        confirmed_pairs = ck.get("pairs", [])
        done = set(ck.get("done", []))
        for a, b in confirmed_pairs:
            dsu.union(a, b)
        logger.info(f"[judge] resuming: {len(done)}/{len(judge_units)} units done, "
                    f"{len(confirmed_pairs)} pairs replayed")

    logger.info(f"[judge] {len(judge_units)} judge units over "
                f"{len({x for u in judge_units for x in u})} candidate entities "
                f"({CONCURRENCY} workers)")

    pool = KeyPool(rotator.keys)
    todo = [u_idx for u_idx in range(len(judge_units)) if u_idx not in done]
    lock = threading.Lock()

    def _work(u_idx: int) -> tuple[int, list[list[int]]]:
        sub_ents = [ents[i] for i in judge_units[u_idx]]
        return u_idx, _judge_unit_pool(pool, sub_ents)

    completed = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for u_idx, groups in tqdm(ex.map(_work, todo), total=len(todo),
                                  desc="judge", unit="unit"):
            unit = judge_units[u_idx]
            with lock:
                for g in groups:
                    for k in range(1, len(g)):
                        a, b = unit[g[0]], unit[g[k]]
                        dsu.union(a, b)
                        confirmed_pairs.append([a, b])
                done.add(u_idx)
                completed += 1
                if completed % 300 == 0:
                    JUDGE_CKPT.write_text(json.dumps({"done": sorted(done),
                                                      "pairs": confirmed_pairs}),
                                          encoding="utf-8")

    # finalize global clusters (singletons included)
    root_to_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        root_to_members[dsu.find(i)].append(i)

    clusters = []
    for members in root_to_members.values():
        members.sort()
        rep_id = max(members, key=lambda i: ents[i]["freq"])  # most frequent text = label
        clusters.append({"members": members, "rep": ents[rep_id]["text"]})
    # deterministic cluster ids: largest/most-frequent first
    clusters.sort(key=lambda c: (-sum(ents[i]["freq"] for i in c["members"]),
                                 -len(c["members"]), c["members"][0]))

    entity_to_cluster: dict[str, int] = {}      # norm text → cluster id
    cluster_summary = []
    for cid, c in enumerate(clusters):
        member_texts = [ents[i]["text"] for i in c["members"]]
        for i in c["members"]:
            entity_to_cluster[ents[i]["norm"]] = cid
        cluster_summary.append({
            "cluster_id": cid,
            "label": c["rep"],
            "size": len(c["members"]),
            "members": member_texts,
        })

    merged = sum(1 for c in clusters if len(c["members"]) > 1)
    logger.info(f"[judge] {len(clusters)} entity clusters "
                f"({merged} contain ≥2 merged entities)")
    out = {"num_clusters": len(clusters),
           "entity_to_cluster": entity_to_cluster,
           "clusters": cluster_summary}
    CLUSTERS_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if JUDGE_CKPT.exists():
        JUDGE_CKPT.unlink()
    logger.info(f"[judge] → {CLUSTERS_FILE.name}")
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Stage 5 — assign clause cluster membership
# ════════════════════════════════════════════════════════════════════════════════

def stage_assign(clusters: dict) -> None:
    logger.info(f"[assign] tagging clauses → {OUTPUT_FILE.name}")
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    e2c: dict[str, int] = clusters["entity_to_cluster"]

    tagged = 0
    membership_counts = Counter()
    for doc in data["merged_documents"]:
        for elem in doc["processed_clauses"]:
            ec = elem.get("entity_concepts") or {}
            se = elem.get("entity_sub_entities") or {}
            cids: set[int] = set()
            for ent in ec:
                cid = e2c.get(_clean(ent).casefold())
                if cid is not None:
                    cids.add(cid)
            for subs in se.values():
                for s in (subs or []):
                    cid = e2c.get(_clean(s).casefold())
                    if cid is not None:
                        cids.add(cid)
            elem["clause_clusters"] = sorted(cids)
            if cids:
                tagged += 1
                membership_counts[len(cids)] += 1

    OUTPUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    avg = (sum(k * v for k, v in membership_counts.items()) / tagged) if tagged else 0
    logger.info(f"[assign] tagged {tagged} elements; "
                f"avg clusters/clause={avg:.1f}; → {OUTPUT_FILE.name}")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "embed", "block", "judge", "assign", "all"],
                    default="all")
    ap.add_argument("--keys-file", default="./all_api.txt")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    index = stage_collect(args.force)
    if args.stage == "collect":
        return

    # Embeddings: paid key only (free keys are out of embedding quota). Skip building
    # the paid-key rotator entirely when a healthy cache already exists, so resuming
    # the judge stage needs no paid key.
    embed_rotator = None
    if args.force or not _embeddings_cache_healthy(index["count"]):
        embed_rotator = ApiKeyRotator(keys_file=str(EMBED_KEYS_FILE), rate_limit_delay=0.0)
    embs = stage_embed(index, embed_rotator, args.force)
    if args.stage == "embed":
        return
    blocks_data = stage_block(index, embs, args.force)
    if args.stage == "block":
        return

    # Judge: paid key + free keys (separate generateContent quota) for throughput.
    judge_rotator = _build_judge_rotator()
    clusters = stage_judge(index, embs, blocks_data, judge_rotator, args.force)
    if args.stage == "judge":
        return
    stage_assign(clusters)
    stats = judge_rotator.get_usage_stats()
    logger.info(f"Done. judge API calls: {stats['total_used']}, "
                f"active={stats['active_keys']}, blacklisted={stats['blacklisted_keys']}.")


def _build_judge_rotator() -> ApiKeyRotator:
    """Rotator over the free keys only (the paid key is intentionally excluded —
    judging runs on the free generateContent quota)."""
    rot = ApiKeyRotator(keys_file=str(FREE_KEYS_FILE), rate_limit_delay=0.0)
    logger.info(f"[judge] rotator has {len(rot.keys)} free keys (paid key excluded)")
    return rot


if __name__ == "__main__":
    main()
