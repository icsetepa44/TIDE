#!/usr/bin/env python3
"""
concept_cluster.py
════════════════════════════════════════════════════════════════════════════════
Within each entity-cluster, find how its concept EVOLVES over time and record, as
a new `concept_cluster` key on the per-cluster files, the evolving concept
thread(s) plus the EXACT clauses that belong to each.

Scope (as chosen):
  • Start from the 492 entity-clusters that merged ≥2 entity variants.
  • Concept-cluster only those with ≥2 DISTINCT DATES (evolution needs ≥2 time
    points). The rest get a stub `concept_cluster` with a skip reason.
  • Giant clusters (huge input) are analysed in date-ordered CHUNKS, then a
    synthesis pass merges the partial threads (clause ids are global, so merging
    is by id).

For each evolving thread Gemini returns: title, summary, a sorted timeline, the
dimensions along which it changed (policy model, legal basis, inspection regime,
duty structure, eligibility, scope, …), the member clauses, and the
static/replicated clauses that look related but are NOT evolution (false
positives). Clauses are referenced by a short integer index in the prompt and
mapped back to their real `text_id` on the way out.

Outputs:
  • adds `concept_cluster` to each clusters_by_id/cluster_<id>.json (existing keys
    untouched)
  • concept_results.json        — raw per-cluster results (resume checkpoint)
  • concept_clusters_index.json — flat index of every evolving thread

Usage:
    python concept_cluster.py
    python concept_cluster.py --limit 5      # smoke-test on a few clusters
    python concept_cluster.py --only 326     # one cluster by id
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.genai import types
from tqdm import tqdm

from break_subentities import GEMINI_MODEL, _call_with_watchdog, setup_logging
from cluster_entities import (
    ApiKeyRotator,
    KeyPool,
    _build_judge_rotator,
    _classify_err,
)

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTERED_FILE = SCRIPT_DIR / "merged_ec_output_canon_clustered.json"
ENTITY_CLUSTERS = SCRIPT_DIR / "entity_clusters.json"
CLUSTERS_DIR = SCRIPT_DIR / "clusters_by_id"
RESULTS_FILE = SCRIPT_DIR / "concept_results.json"
INDEX_FILE = SCRIPT_DIR / "concept_clusters_index.json"

CONCEPT_KEY = "concept_cluster"
MIN_OBJECTS = 10                 # only concept-cluster clusters with >= this many objects
TEXT_TRUNC = 600                 # chars of resolved_text per clause
NOTES_PER_CLAUSE = 4             # max concept notes per clause
CHARS_BUDGET = 80000             # max prompt chars before a cluster is chunked
CONCURRENCY = 16
WATCHDOG = 120
JUDGE_CFG = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)


# ════════════════════════════════════════════════════════════════════════════════
# Build per-clause compact records
# ════════════════════════════════════════════════════════════════════════════════

def _date_of(elem: dict) -> str | None:
    m = re.match(r"(\d{8})", elem.get("text_id", ""))
    if not m:
        return None
    d = m.group(1)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _source_token(elem: dict) -> str:
    src = elem.get("source_file", "") or ""
    base = src.split("/")[-1].replace("_output.json", "").replace(".json", "")
    return base[:60]


def _concept_notes(elem: dict, member_set: set[str]) -> list[str]:
    notes: list[str] = []
    ec = elem.get("entity_concepts") or {}
    for ent, cons in ec.items():
        if _clean(ent).casefold() in member_set:
            for c in (cons or []):
                notes.append(_clean(c))
    se = elem.get("entity_sub_entities") or {}
    subs_here = sorted({_clean(s) for subs in se.values() for s in (subs or [])
                        if _clean(s).casefold() in member_set})
    if subs_here and len(notes) < NOTES_PER_CLAUSE:
        notes.append("mentions: " + ", ".join(subs_here[:6]))
    # dedupe, cap
    seen, out = set(), []
    for nshort in notes:
        if nshort and nshort not in seen:
            seen.add(nshort)
            out.append(nshort)
        if len(out) >= NOTES_PER_CLAUSE:
            break
    return out


def _build_items(objs: list[dict], member_set: set[str]) -> list[dict]:
    """One compact record per clause, sorted by date then clause id."""
    recs = []
    for e in objs:
        recs.append({
            "date": _date_of(e) or "undated",
            "clause_id": e.get("text_id", ""),
            "type": e.get("type", ""),
            "src": _source_token(e),
            "notes": _concept_notes(e, member_set),
            "text": _clean(e.get("resolved_text", ""))[:TEXT_TRUNC],
        })
    recs.sort(key=lambda r: (r["date"], r["clause_id"]))
    for i, r in enumerate(recs):
        r["idx"] = i
    return recs


def _item_str(r: dict) -> str:
    notes = (" | ".join(r["notes"])) if r["notes"] else "(no concept note)"
    return (f'[{r["idx"]}] date={r["date"]} src={r["src"]} {r["type"]}\n'
            f'    concept: {notes}\n'
            f'    text: {r["text"]}')


# ════════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════════

ANALYZE_PROMPT = """You are analysing how a single legal/policy CONCEPT evolves over time across
Bangladeshi customs/tax instruments (SROs, acts, NBR directives).

The concept is anchored on the entity cluster: "{label}"
(equivalent entity names merged into it: {members})

Below are the clauses that touch this concept, each with a short [index], its DATE,
source instrument, distilled concept notes, and text. They are sorted by date.

{chunk_note}Your job: identify the EVOLVING concept thread(s) — a concept whose SUBSTANCE changes
over time (policy model, legal basis, rates/structure, eligibility, classifications,
scope, procedure, institutions). A cluster may contain ONE or SEVERAL distinct
evolving threads; split them when they are genuinely different.

Rules:
- Group clauses into evolving threads by what actually CHANGES across dates.
- Mark clauses that merely RESTATE materially-unchanged content (same procedure/
  definition repeated across years) as static/replicated — these are NOT evolution
  and must go in "static_replicated_clause_indices", not the member list.
- Clauses unrelated to any evolving thread go in "unassigned_clause_indices".
- Reference clauses ONLY by their integer index. Use real dates from the records.

Clauses:
{listing}

Return STRICT JSON:
{{
  "concept_clusters": [
    {{
      "title": "<short name of this evolving thread>",
      "summary": "<2-4 sentence narrative of how it evolved>",
      "timeline": [
        {{"date": "<YYYY-MM-DD>", "instrument": "<e.g. SRO 155/2016>",
          "clause_indices": [<int>], "change": "<what changed at this step>"}}
      ],
      "evolution": {{"<dimension>": "<how it changed across time>"}},
      "member_clause_indices": [<int>, ...],
      "static_replicated_clause_indices": [<int>, ...]
    }}
  ],
  "unassigned_clause_indices": [<int>, ...]
}}
Return only the JSON object."""

SYNTH_PROMPT = """You are merging partial analyses of how a single concept ("{label}") evolves over
time. Each partial below was computed on a CHUNK of the clauses; clause indices are
GLOBAL and consistent across chunks.

Merge partial threads that describe the SAME evolving concept into one (union their
clause indices). Keep genuinely distinct threads separate. Produce a clean, rich
final set of evolving concept threads.

Partial analyses (JSON):
{partials}

Return STRICT JSON with the SAME schema as a partial:
{{
  "concept_clusters": [
    {{"title": "...", "summary": "...",
      "timeline": [{{"date":"...","instrument":"...","clause_indices":[int],"change":"..."}}],
      "evolution": {{"<dimension>": "..."}},
      "member_clause_indices": [int], "static_replicated_clause_indices": [int]}}
  ],
  "unassigned_clause_indices": [int]
}}
Return only the JSON object."""


# ════════════════════════════════════════════════════════════════════════════════
# LLM call (thread-safe via KeyPool) + parsing
# ════════════════════════════════════════════════════════════════════════════════

def _call(pool: KeyPool, prompt: str) -> str | None:
    for _ in range(6):
        k = pool.acquire()
        if k is None:
            return None
        try:
            resp = _call_with_watchdog(
                lambda: pool.client(k).models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=JUDGE_CFG),
                WATCHDOG)
            pool.release(k)
            return resp.text
        except Exception as e:  # noqa: BLE001
            etype = _classify_err(e)
            if etype == "permanent":
                pool.release(k, dead=True)
            elif etype == "rate_limit":
                pool.release(k, cooldown=max(ApiKeyRotator._extract_retry_delay(e), 30))
            else:
                pool.release(k, cooldown=3)
    return None


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _valid_idxs(seq, n: int) -> list[int]:
    out, seen = [], set()
    for x in (seq or []):
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < n and xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Per-cluster processing
# ════════════════════════════════════════════════════════════════════════════════

def _chunks_by_budget(items: list[dict]) -> list[list[dict]]:
    chunks, cur, size = [], [], 0
    for r in items:
        s = len(_item_str(r))
        if cur and size + s > CHARS_BUDGET:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += s
    if cur:
        chunks.append(cur)
    return chunks


def _analyze_chunk(pool: KeyPool, label: str, members: str, items: list[dict],
                   n_total: int, chunk_note: str) -> dict | None:
    listing = "\n".join(_item_str(r) for r in items)
    prompt = ANALYZE_PROMPT.format(label=label, members=members, listing=listing,
                                   chunk_note=chunk_note)
    data = _parse_json(_call(pool, prompt))
    if not isinstance(data, dict):
        return None
    return data


def _clean_result(data: dict, n: int) -> dict:
    ccs = []
    for cc in (data.get("concept_clusters") or []):
        if not isinstance(cc, dict):
            continue
        tl = []
        for t in (cc.get("timeline") or []):
            if isinstance(t, dict):
                t = dict(t)
                t["clause_indices"] = _valid_idxs(t.get("clause_indices"), n)
                tl.append(t)
        ccs.append({
            "title": str(cc.get("title", ""))[:200],
            "summary": str(cc.get("summary", "")),
            "timeline": tl,
            "evolution": cc.get("evolution") if isinstance(cc.get("evolution"), dict) else {},
            "member_clause_indices": _valid_idxs(cc.get("member_clause_indices"), n),
            "static_replicated_clause_indices": _valid_idxs(
                cc.get("static_replicated_clause_indices"), n),
        })
    return {"concept_clusters": ccs,
            "unassigned_clause_indices": _valid_idxs(data.get("unassigned_clause_indices"), n)}


def process_cluster(pool: KeyPool, cid: int, label: str, members: list[str],
                    objs: list[dict]) -> dict:
    member_set = {m.casefold() for m in members}
    items = _build_items(objs, member_set)
    n = len(items)
    members_str = ", ".join(members[:20])

    chunks = _chunks_by_budget(items)
    if len(chunks) == 1:
        data = _analyze_chunk(pool, label, members_str, items, n, "")
        result = _clean_result(data, n) if data else {"concept_clusters": [],
                                                       "unassigned_clause_indices": []}
    else:
        partials = []
        for ci, ch in enumerate(chunks):
            note = (f"(This is chunk {ci+1}/{len(chunks)}; clause indices are global "
                    f"and continue across chunks.)\n\n")
            d = _analyze_chunk(pool, label, members_str, ch, n, note)
            if d:
                partials.append(_clean_result(d, n))
        if not partials:
            result = {"concept_clusters": [], "unassigned_clause_indices": []}
        else:
            synth = _parse_json(_call(pool, SYNTH_PROMPT.format(
                label=label, partials=json.dumps(partials, ensure_ascii=False)[:120000])))
            if isinstance(synth, dict) and synth.get("concept_clusters") is not None:
                result = _clean_result(synth, n)
            else:  # synthesis failed → union partials
                merged = {"concept_clusters": [], "unassigned_clause_indices": []}
                for p in partials:
                    merged["concept_clusters"].extend(p["concept_clusters"])
                    merged["unassigned_clause_indices"].extend(p["unassigned_clause_indices"])
                result = merged

    # map indices → real clause text_ids
    idx2id = [r["clause_id"] for r in items]

    def ids(idxs):
        return [idx2id[i] for i in idxs]

    for cc in result["concept_clusters"]:
        cc["member_clause_ids"] = ids(cc["member_clause_indices"])
        cc["static_replicated_clause_ids"] = ids(cc["static_replicated_clause_indices"])
        for t in cc["timeline"]:
            t["clause_ids"] = ids(t.get("clause_indices", []))
    result["unassigned_clause_ids"] = ids(result["unassigned_clause_indices"])
    result["cluster_id"] = cid
    result["label"] = label
    result["num_objects"] = n
    result["num_chunks"] = len(chunks)
    result["num_concept_threads"] = len(result["concept_clusters"])
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=int, default=None, help="process a single cluster id")
    ap.add_argument("--min-objects", type=int, default=MIN_OBJECTS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-write-files", action="store_true",
                    help="compute results but don't patch clusters_by_id files")
    args = ap.parse_args()
    setup_logging(False)

    ecj = json.loads(ENTITY_CLUSTERS.read_text(encoding="utf-8"))
    clustered = json.loads(CLUSTERED_FILE.read_text(encoding="utf-8"))

    info = {c["cluster_id"]: c for c in ecj["clusters"]}
    objs: dict[int, list[dict]] = defaultdict(list)
    for doc in clustered["merged_documents"]:
        for e in doc["processed_clauses"]:
            for cid in (e.get("clause_clusters") or []):
                objs[cid].append(e)

    # candidate clusters = those with a significant number of objects (>= MIN_OBJECTS),
    # regardless of entity-variant count (this is what "evolving concept" needs).
    candidates = [c for c in ecj["clusters"]
                  if len(objs.get(c["cluster_id"], [])) >= args.min_objects]
    targets, skipped = [], {}
    for c in candidates:
        cid = c["cluster_id"]
        es = objs.get(cid, [])
        dates = {d for d in (_date_of(e) for e in es) if d}
        if len(dates) < 2:
            skipped[cid] = "single date (no temporal evolution)"
        else:
            targets.append(c)

    if args.only is not None:
        targets = [c for c in ecj["clusters"] if c["cluster_id"] == args.only]
        skipped.pop(args.only, None)
        candidates = targets
    elif args.limit:
        targets = sorted(targets, key=lambda c: -len(objs[c["cluster_id"]]))[:args.limit]

    logger.info(f"{len(candidates)} clusters with ≥{args.min_objects} objects; "
                f"{len(targets)} to concept-cluster (≥2 dates); {len(skipped)} skipped")

    # resume
    results: dict[str, dict] = {}
    if RESULTS_FILE.exists() and not args.force:
        results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        logger.info(f"resuming: {len(results)} clusters already done")
    todo = [c for c in targets if str(c["cluster_id"]) not in results]

    pool = KeyPool(_build_judge_rotator().keys)
    lock = threading.Lock()
    done = 0

    def work(c):
        cid = c["cluster_id"]
        return cid, process_cluster(pool, cid, c["label"], c["members"], objs[cid])

    if todo:
        # small clusters first: as_completed flushes them quickly for smooth progress
        # and frequent checkpoints; the few giants tail at the end without blocking.
        todo = sorted(todo, key=lambda c: len(objs[c["cluster_id"]]))
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = [ex.submit(work, c) for c in todo]
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="concept", unit="cluster"):
                cid, res = fut.result()
                with lock:
                    results[str(cid)] = res
                    done += 1
                    if done % 20 == 0:
                        RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False),
                                                encoding="utf-8")
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    logger.info(f"analyzed {len(results)} clusters → {RESULTS_FILE.name}")

    if not args.no_write_files:
        _write_outputs(results, skipped, candidates)


def _write_outputs(results: dict, skipped: dict, candidates: list) -> None:
    """Patch each clusters_by_id file with a `concept_cluster` key, and write an index."""
    pad = 5
    index = []
    patched = 0
    for c in candidates:
        cid = c["cluster_id"]
        fp = CLUSTERS_DIR / f"cluster_{cid:0{pad}d}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        if str(cid) in results:
            r = results[str(cid)]
            data[CONCEPT_KEY] = {
                "status": "analyzed",
                "num_concept_threads": r["num_concept_threads"],
                "concept_clusters": [
                    {k: cc[k] for k in ("title", "summary", "timeline", "evolution",
                                        "member_clause_ids", "static_replicated_clause_ids")}
                    for cc in r["concept_clusters"]
                ],
                "unassigned_clause_ids": r["unassigned_clause_ids"],
            }
            for cc in r["concept_clusters"]:
                index.append({"cluster_id": cid, "label": r["label"],
                              "title": cc["title"], "num_member_clauses": len(cc["member_clause_ids"])})
        else:
            data[CONCEPT_KEY] = {"status": "skipped",
                                 "reason": skipped.get(cid, "not a multi-date cluster"),
                                 "concept_clusters": []}
        fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        patched += 1
    INDEX_FILE.write_text(json.dumps({"total_threads": len(index), "threads": index},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"patched {patched} cluster files with '{CONCEPT_KEY}'; "
                f"{len(index)} threads → {INDEX_FILE.name}")


if __name__ == "__main__":
    main()
