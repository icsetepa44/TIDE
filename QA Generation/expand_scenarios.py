#!/usr/bin/env python3
"""
expand_scenarios.py
════════════════════════════════════════════════════════════════════════════════
For every Q&A of type "scenario_short_answer" in concept_qa/*_QA.json, REWRITE
ONLY the "scenario" key into a longer, story-based, real-world narrative.

Everything else is left untouched — question, answer, date_or_instrument, type and
all other pairs/keys are preserved exactly. The new scenario must stay consistent
with the same question/answer (same facts, same time period) and must NOT reveal
the answer; it just sets up the same question with a vivid, realistic story.

In-place update of the QA files. A one-time backup of the directory is made at
concept_qa_backup/ before anything is modified. Resumable via a sidecar checkpoint.

Keys/model: free keys only (all_api.txt) + gemini-3.1-flash-lite.

Usage:
    python expand_scenarios.py
    python expand_scenarios.py --limit 3            # smoke test
    python expand_scenarios.py --concurrency 8
    python expand_scenarios.py --force              # re-expand everything
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.genai import types
from tqdm import tqdm

from break_subentities import GEMINI_MODEL, _call_with_watchdog
from break_subentities import setup_logging
from cluster_entities import ApiKeyRotator, KeyPool, _build_judge_rotator, _classify_err

logger = logging.getLogger("subentity_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa"
BACKUP_DIR = SCRIPT_DIR / "concept_qa_backup"
DONE_FILE = SCRIPT_DIR / "expand_scenarios_done.json"

SCEN_TYPE = "scenario_short_answer"
CONCURRENCY = 8
WATCHDOG = 90
GEN_CFG = types.GenerateContentConfig(response_mime_type="application/json",
                                      temperature=0.7, max_output_tokens=4096)

PROMPT = """You rewrite SCENARIO setups for a customs/tax Q&A dataset about Bangladeshi policy.

For each item below you are given the current short scenario plus the QUESTION, the
ANSWER, and the relevant DATE/INSTRUMENT (for your reference only). Rewrite ONLY the
scenario into a longer, vivid, REAL-WORLD STORY (about 3-6 sentences) that naturally
leads up to the SAME question.

Strict rules:
- Keep it fully CONSISTENT with the given answer and time period. Do NOT change any
  facts (dates, rates, instruments, entities, categories).
- Do NOT reveal or state the answer in the scenario.
- Do NOT add invented specific figures (rates, durations, amounts) — keep the story
  qualitative; the precise facts belong to the answer.
- Give it a realistic persona/setting (an importer, manufacturer, customs officer,
  trader, passenger, etc.) and concrete but non-contradicting context.
- Keep the same situation as the original scenario, just richer and story-like.

Thread topic: "{title}" (concept: {label})

Items (JSON):
{items}

Return STRICT JSON: {{"rewrites": [{{"i": <item index>, "scenario": "<new story-based scenario>"}}]}}
Include every index exactly once. Return only the JSON object."""


# ════════════════════════════════════════════════════════════════════════════════
# LLM call
# ════════════════════════════════════════════════════════════════════════════════

def _call(pool: KeyPool, prompt: str) -> str | None:
    for _ in range(15):                         # patient: ride out per-minute (RPM) throttling
        k = pool.acquire()
        if k is None:
            return None
        try:
            resp = _call_with_watchdog(
                lambda: pool.client(k).models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=GEN_CFG), WATCHDOG)
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


def _salvage_objects(text: str) -> list[dict]:
    """Recover complete top-level {...} objects from possibly-truncated JSON."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        out.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return out


def _parse_rewrites(text: str) -> dict[int, str]:
    if not text:
        return {}
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    rewrites = []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            rewrites = data.get("rewrites") or []
    except json.JSONDecodeError:
        rewrites = []
    if not rewrites:
        # truncated / malformed → salvage each complete rewrite object
        rewrites = [o for o in _salvage_objects(raw) if "i" in o and "scenario" in o]

    out: dict[int, str] = {}
    for r in rewrites:
        if isinstance(r, dict) and "i" in r and isinstance(r.get("scenario"), str):
            try:
                out[int(r["i"])] = r["scenario"].strip()
            except (TypeError, ValueError):
                continue
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Per-file processing
# ════════════════════════════════════════════════════════════════════════════════

def expand_file(pool: KeyPool, path: Path) -> int:
    """Rewrite scenarios in one QA file in place. Returns number of scenarios updated.
    Idempotent: only scenarios still identical to the backup (i.e. not yet expanded)
    are rewritten, so re-running never re-stories an already-expanded scenario."""
    d = json.loads(path.read_text(encoding="utf-8"))
    pairs = d.get("qa_pairs", [])
    bpairs = []
    bpath = BACKUP_DIR / path.name
    if bpath.exists():
        bpairs = json.loads(bpath.read_text(encoding="utf-8")).get("qa_pairs", [])

    def already_expanded(pos: int) -> bool:
        return pos < len(bpairs) and isinstance(bpairs[pos], dict) and \
            bpairs[pos].get("scenario") != pairs[pos].get("scenario")

    scen_positions = [i for i, q in enumerate(pairs)
                      if isinstance(q, dict) and q.get("type") == SCEN_TYPE and q.get("scenario")
                      and not already_expanded(i)]
    if not scen_positions:
        return 0

    # build items with a LOCAL index (0..k-1) mapped back to qa_pairs positions
    items = []
    for li, pos in enumerate(scen_positions):
        q = pairs[pos]
        items.append({"i": li, "scenario": q.get("scenario", ""),
                      "question": q.get("question", ""), "answer": q.get("answer", ""),
                      "date_or_instrument": q.get("date_or_instrument", "")})
    prompt = PROMPT.format(title=d.get("title", ""), label=d.get("cluster_label", ""),
                           items=json.dumps(items, ensure_ascii=False, indent=2))
    rewrites = _parse_rewrites(_call(pool, prompt))

    updated = 0
    for li, pos in enumerate(scen_positions):
        new = rewrites.get(li)
        if new and len(new) > len(items[li]["scenario"]):   # only replace with a fuller story
            pairs[pos]["scenario"] = new
            updated += 1
    if updated:
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging(False)

    if not BACKUP_DIR.exists():
        shutil.copytree(QA_DIR, BACKUP_DIR)
        logger.info(f"backed up {QA_DIR.name} → {BACKUP_DIR.name}")

    done = set()
    if DONE_FILE.exists() and not args.force:
        done = set(json.loads(DONE_FILE.read_text()))

    files = sorted(QA_DIR.glob("*_QA.json"))
    todo = [f for f in files if f.name not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info(f"{len(files)} QA files; {len(todo)} to expand "
                f"({len(files)-len(todo)} already done)")

    pool = KeyPool(_build_judge_rotator().keys)
    lock = threading.Lock()
    total_updated = 0

    def task(f: Path):
        return f.name, expand_file(pool, f)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(task, f) for f in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="scenarios", unit="file"):
                name, n = fut.result()
                with lock:
                    total_updated += n
                    done.add(name)
                    if len(done) % 20 == 0:
                        DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    DONE_FILE.write_text(json.dumps(sorted(done)), encoding="utf-8")
    logger.info(f"updated {total_updated} scenarios across {len(todo)} files")


if __name__ == "__main__":
    main()
