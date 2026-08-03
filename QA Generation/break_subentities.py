#!/usr/bin/env python3
"""
break_subentities.py
════════════════════════════════════════════════════════════════════════════════
Break the high-level entities in `merged_ec_output_canon.json` into concrete
sub-entities, using the per-entity *concept* descriptions as the source of truth.

The existing `entity_concepts` field is NOT replaced. For every element
(clause / metadata) we add a new sibling field:

    "entity_sub_entities": {
        "<Entity Name>": ["<sub-entity 1>", "<sub-entity 2>", ...],
        ...
    }

Example
-------
Input  entity_concepts:
    "Customs Duty": [
        "The duty rate applicable for a specified weight of gold or
         gold-related items has been reduced ..."
    ]
Output entity_sub_entities:
    "Customs Duty": ["gold", "gold-related items"]

One Gemini call handles ALL entities of a single element at once
(≈4.4k calls total — well within the 38-key × 500/day budget).

The API-key handling (ApiKeyRotator) is ported verbatim from the project's
existing EC-extraction pipeline.

Usage
-----
    python break_subentities.py                 # run / resume
    python break_subentities.py --limit 20      # smoke-test on first 20 elements
    python break_subentities.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

INPUT_FILE = "merged_ec_output_canon.json"
OUTPUT_FILE = "merged_ec_output_canon_subentities.json"
SUB_ENTITIES_KEY = "entity_sub_entities"        # new field added to each element

API_KEYS_FILE = "./all_api.txt"
GEMINI_MODEL = "gemini-3.1-flash-lite"
CALLS_PER_DAY = 500                             # max API calls per key per day
RATE_LIMIT_DELAY = 0.0                          # delay between calls (s); flash-lite is fast
MAX_RETRIES = 3                                 # same-call transient retries before rotating
TRANSIENT_COOLDOWN = 30                         # park a transiently-failing key for N s
CHECKPOINT_INTERVAL = 100                       # flush output every N processed elements
REQUEST_TIMEOUT = 60                            # per-call HTTP timeout (s) — SDK-level
WATCHDOG_TIMEOUT = 75                            # hard wall-clock cap per call (s) — thread-level
                                                # WATCHDOG > REQUEST so the HTTP timeout normally
                                                # fires first; the watchdog only catches true hangs.

logger = logging.getLogger("subentity_pipeline")


# ════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ════════════════════════════════════════════════════════════════════════════════

class _TqdmLoggingHandler(logging.Handler):
    """Emit log records via tqdm.write so they don't corrupt an active bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = _TqdmLoggingHandler()
    console.setFormatter(fmt)
    console.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(console)
    logger.setLevel(level)


# ════════════════════════════════════════════════════════════════════════════════
# Client / call helpers
# ════════════════════════════════════════════════════════════════════════════════

def _make_client(api_key: str) -> genai.Client:
    """Build a genai.Client with an SDK-level HTTP timeout so a stalled request
    raises instead of hanging the whole pipeline forever."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),  # ms
    )


def _call_with_watchdog(fn, timeout_s: float):
    """Run fn() in a daemon thread and join with a hard timeout. If it overruns,
    raise TimeoutError instead of blocking. The orphaned thread is daemon and its
    underlying request carries its own HTTP timeout, so it cannot leak forever."""
    box: dict = {}

    def _run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 — surface any failure to caller
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"generate_content exceeded watchdog timeout of {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ════════════════════════════════════════════════════════════════════════════════
# API Key Management  (ported faithfully from the EC-extraction pipeline)
# ════════════════════════════════════════════════════════════════════════════════

class ApiKeyRotator:
    """Multiple Gemini API keys with round-robin rotation, per-key usage limits,
    and three-way error classification (permanent / rate_limit / transient)."""

    def __init__(self, keys_file: str | None = None,
                 calls_per_day: int = CALLS_PER_DAY,
                 rate_limit_delay: float = RATE_LIMIT_DELAY):
        self.keys: list[str] = []
        self.calls_per_day = calls_per_day
        self.rate_limit_delay = rate_limit_delay
        self._current_index = 0
        self._total_calls = 0

        self.usage_count: dict[str, int] = {}
        self.failure_count: dict[str, int] = {}
        self.cooldown_until: dict[str, float] = {}
        self._rate_limit_hits: dict[str, int] = {}
        self.exhausted_keys: set[str] = set()
        self.blacklisted_keys: set[str] = set()
        self._clients: dict[str, genai.Client] = {}

        self._load_keys(keys_file)
        if not self.keys:
            raise ValueError(
                "No API keys found. Provide keys via:\n"
                f"  1. File: {API_KEYS_FILE} (one key per line / comma-separated)\n"
                "  2. Env vars: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...\n"
                "  3. Env var: GEMINI_API_KEY (single key)"
            )

        for key in self.keys:
            self.usage_count[key] = 0
            self.failure_count[key] = 0
            self.cooldown_until[key] = 0.0
            self._rate_limit_hits[key] = 0

        self._clients[self.keys[0]] = _make_client(self.keys[0])
        logger.info(f"ApiKeyRotator initialized with {len(self.keys)} key(s), "
                    f"calls_per_day={calls_per_day}, delay={rate_limit_delay}s")

    def _load_keys(self, keys_file: str | None) -> None:
        paths_to_try = []
        if keys_file:
            paths_to_try.append(keys_file)
        paths_to_try.append(API_KEYS_FILE)

        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, "r") as fh:
                        content = fh.read().strip()
                    if content:
                        raw = re.split(r"[,\n]+", content)
                        self.keys = [k.strip() for k in raw if k.strip()]
                        if self.keys:
                            logger.info(f"Loaded {len(self.keys)} API key(s) from {path}")
                            return
                except Exception as e:
                    logger.warning(f"Failed to read keys from {path}: {e}")

        i = 1
        while True:
            key = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
            if not key:
                break
            self.keys.append(key)
            i += 1
        if self.keys:
            logger.info(f"Loaded {len(self.keys)} API key(s) from environment variables")
            return

        single = os.environ.get("GEMINI_API_KEY", "").strip()
        if single:
            self.keys = [single]
            logger.info("Loaded 1 API key from GEMINI_API_KEY env var")

    @property
    def current_key(self) -> str:
        return self.keys[self._current_index]

    @property
    def current_key_masked(self) -> str:
        key = self.current_key
        return key[:4] + "..." + key[-4:] if len(key) > 8 else "****"

    def get_client(self) -> genai.Client:
        key = self.current_key
        if key not in self._clients:
            self._clients[key] = _make_client(key)
        return self._clients[key]

    def rotate_key(self) -> bool:
        attempts = 0
        while attempts < len(self.keys):
            self._current_index = (self._current_index + 1) % len(self.keys)
            attempts += 1
            key = self.keys[self._current_index]
            if key in self.blacklisted_keys:
                continue
            if key in self.exhausted_keys:
                continue
            if time.time() < self.cooldown_until.get(key, 0):
                continue
            if self.usage_count.get(key, 0) >= self.calls_per_day:
                self.exhausted_keys.add(key)
                continue
            logger.info(f"Rotated to key [{self._current_index}] "
                        f"(masked: {self.current_key_masked}, "
                        f"usage: {self.usage_count.get(key, 0)}/{self.calls_per_day})")
            return True
        logger.error("All API keys are exhausted, blacklisted, or in cooldown!")
        return False

    def has_any_usable_key(self) -> bool:
        for key in self.keys:
            if key in self.blacklisted_keys:
                continue
            if key in self.exhausted_keys:
                continue
            return True
        return False

    def park_key(self, key: str, seconds: float) -> None:
        if key in self.keys:
            self.cooldown_until[key] = time.time() + seconds

    def wait_for_cooldown_expiry(self) -> float:
        now = time.time()
        min_wait = float("inf")
        for key in self.keys:
            if key in self.blacklisted_keys or key in self.exhausted_keys:
                continue
            remaining = self.cooldown_until.get(key, 0) - now
            if remaining <= 0:
                return 0
            min_wait = min(min_wait, remaining)
        return min_wait if min_wait != float("inf") else 0

    def report_success(self) -> None:
        key = self.current_key
        self.usage_count[key] = self.usage_count.get(key, 0) + 1
        self.failure_count[key] = 0
        self._rate_limit_hits[key] = 0
        self._total_calls += 1

    def blacklist_key(self, key: str | None = None) -> None:
        key = key or self.current_key
        self.blacklisted_keys.add(key)
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
        idx = self.keys.index(key) if key in self.keys else -1
        logger.warning(f"Key [{idx}] ({masked}) PERMANENTLY BLACKLISTED — will not be retried.")

    def report_error(self, error: Exception) -> str:
        key = self.current_key
        self.failure_count[key] = self.failure_count.get(key, 0) + 1
        error_type = self.classify_error_type(error)

        if error_type == "permanent":
            self.blacklist_key(key)
        elif error_type == "rate_limit":
            self._rate_limit_hits[key] = self._rate_limit_hits.get(key, 0) + 1
            hits = self._rate_limit_hits[key]
            if hits >= 3:
                self.exhausted_keys.add(key)
                logger.warning(f"Key [{self._current_index}] ({self.current_key_masked}) "
                               f"EXHAUSTED — hit rate-limit {hits} times.")
            else:
                retry_delay = self._extract_retry_delay(error)
                cooldown = max(retry_delay, 45)
                self.cooldown_until[key] = time.time() + cooldown
                logger.warning(f"Key [{self._current_index}] ({self.current_key_masked}) "
                               f"rate-limited (hit {hits}/3), cooldown {cooldown:.0f}s.")
        else:
            logger.warning(f"Key [{self._current_index}] ({self.current_key_masked}) "
                           f"transient error: {str(error)[:150]}")
        return error_type

    @staticmethod
    def _extract_retry_delay(error: Exception) -> float:
        msg = str(error)
        match = re.search(r"retry\s+in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"retryDelay.*?'(\d+)s'", msg)
        if match:
            return float(match.group(1))
        return 0

    def classify_error_type(self, error: Exception) -> str:
        msg = str(error).lower()
        permanent_indicators = [
            "leaked", "api_key_invalid", "api key invalid",
            "permission_denied", "permission denied", "forbidden",
            "invalid api key", "key not found",
            "api key not valid", "api key expired", "revoked", "disabled",
        ]
        quota_indicators_in_403 = ["quota", "rate limit", "rate_limit", "resource_exhausted"]
        if any(ind in msg for ind in permanent_indicators):
            if any(qi in msg for qi in quota_indicators_in_403):
                return "rate_limit"
            return "permanent"
        rate_limit_indicators = [
            "429", "quota", "rate limit", "rate_limit",
            "resource_exhausted", "too many requests", "resourceexhausted",
        ]
        if any(ind in msg for ind in rate_limit_indicators):
            return "rate_limit"
        return "transient"

    def wait_rate_limit(self) -> None:
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

    def get_usage_stats(self) -> dict:
        total_used = sum(self.usage_count.values())
        total_available = len(self.keys) * self.calls_per_day
        return {
            "total_used": total_used,
            "total_available": total_available,
            "active_keys": len(self.keys) - len(self.exhausted_keys) - len(self.blacklisted_keys),
            "exhausted_keys": len(self.exhausted_keys),
            "blacklisted_keys": len(self.blacklisted_keys),
        }

    @property
    def total_calls(self) -> int:
        return self._total_calls


# ════════════════════════════════════════════════════════════════════════════════
# Gemini call with rotation / retry
# ════════════════════════════════════════════════════════════════════════════════

def generate_with_rotation(rotator: ApiKeyRotator, prompt: str) -> str | None:
    """Call Gemini with the current key; on error classify, retry transient calls
    (same key, quick backoff), rotate on rate_limit / permanent / exhausted retries.
    Returns the raw text response, or None if all keys are unavailable."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
    )

    while True:
        if not rotator.has_any_usable_key():
            logger.error("No usable API keys remain — aborting this element.")
            return None

        # Skip a key that is cooling down / over quota before spending the call.
        key = rotator.current_key
        if (key in rotator.blacklisted_keys or key in rotator.exhausted_keys
                or time.time() < rotator.cooldown_until.get(key, 0)
                or rotator.usage_count.get(key, 0) >= rotator.calls_per_day):
            if not rotator.rotate_key():
                wait = rotator.wait_for_cooldown_expiry()
                if wait <= 0:
                    return None
                logger.info(f"All keys cooling down — sleeping {wait:.0f}s.")
                time.sleep(wait + 1)
            continue

        transient_attempts = 0
        while transient_attempts < MAX_RETRIES:
            try:
                client = rotator.get_client()
                resp = _call_with_watchdog(
                    lambda: client.models.generate_content(
                        model=GEMINI_MODEL, contents=prompt, config=config),
                    WATCHDOG_TIMEOUT,
                )
                rotator.report_success()
                rotator.wait_rate_limit()
                return resp.text
            except Exception as e:  # noqa: BLE001 — broad by design (SDK raises many types)
                etype = rotator.report_error(e)
                if etype == "transient":
                    transient_attempts += 1
                    backoff = min(2 ** transient_attempts, 8)
                    time.sleep(backoff)
                    continue
                break  # rate_limit / permanent → leave inner loop, rotate

        # Park a key that exhausted its transient retries, then rotate away.
        if transient_attempts >= MAX_RETRIES:
            rotator.park_key(rotator.current_key, TRANSIENT_COOLDOWN)
        if not rotator.rotate_key():
            wait = rotator.wait_for_cooldown_expiry()
            if wait <= 0:
                return None
            logger.info(f"All keys busy — sleeping {wait:.0f}s before retry.")
            time.sleep(wait + 1)


# ════════════════════════════════════════════════════════════════════════════════
# Prompt + parsing
# ════════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """You are an information-extraction assistant working on legal/customs documents.

You are given a set of HIGH-LEVEL entities. Each entity comes with one or more
"concept" sentences that describe what the entity refers to in a specific clause.
Your job is to BREAK DOWN each high-level entity into the concrete, specific
SUB-ENTITIES that the concept sentences actually mention.

Rules:
- Extract sub-entities ONLY from the concept text of that entity. Do not invent
  anything that is not grounded in the concepts.
- Sub-entities are the concrete things, items, goods, instruments, sections,
  amounts-as-things, parties, or named objects referenced (e.g. for a concept
  about "gold or gold-related items", the sub-entities are ["gold", "gold-related items"]).
- Keep each sub-entity short (a noun / noun-phrase). Deduplicate.
- Preserve the original language of the term where it is a proper/legal term.
- If an entity's concepts contain no meaningful sub-entities, return an empty list for it.
- Return STRICT JSON: an object whose keys are EXACTLY the given entity names and
  whose values are arrays of sub-entity strings. No commentary, no extra keys.

Entities and their concepts (JSON):
{payload}

Return only the JSON object mapping each entity name to its array of sub-entities.
"""


def build_prompt(entity_concepts: dict) -> str:
    payload = json.dumps(entity_concepts, ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.format(payload=payload)


def parse_response(text: str, entity_names: list[str]) -> dict[str, list[str]] | None:
    """Parse the model JSON and coerce it into {entity: [str, ...]} restricted to
    the known entity names. Returns None if nothing usable was parsed."""
    if not text:
        return None
    raw = text.strip()
    # Strip ```json fences if present.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort: grab the outermost {...}
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None

    name_set = set(entity_names)
    out: dict[str, list[str]] = {}
    for ent in entity_names:
        val = data.get(ent)
        if val is None:
            out[ent] = []
            continue
        if isinstance(val, str):
            val = [val]
        if isinstance(val, list):
            seen, clean = set(), []
            for item in val:
                s = str(item).strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    clean.append(s)
            out[ent] = clean
        else:
            out[ent] = []
    # Carry over any extra keys the model returned that match (defensive).
    for k, v in data.items():
        if k in name_set or not isinstance(v, list):
            continue
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════════

def iter_elements(doc_root: dict):
    """Yield every clause/metadata element that carries entity_concepts."""
    for doc in doc_root.get("merged_documents", []):
        for elem in doc.get("processed_clauses", []):
            ec = elem.get("entity_concepts")
            if ec:
                yield elem


def save_output(doc_root: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc_root, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Break high-level entities into sub-entities.")
    ap.add_argument("--input", default=INPUT_FILE)
    ap.add_argument("--output", default=OUTPUT_FILE)
    ap.add_argument("--keys-file", default=API_KEYS_FILE)
    ap.add_argument("--limit", type=int, default=0, help="Process at most N elements (0 = all).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-process elements that already have sub-entities.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    script_dir = Path(__file__).resolve().parent
    input_path = (script_dir / args.input) if not os.path.isabs(args.input) else Path(args.input)
    output_path = (script_dir / args.output) if not os.path.isabs(args.output) else Path(args.output)

    # Resume from existing output if present, else from the source file.
    if output_path.exists() and not args.overwrite:
        logger.info(f"Resuming from existing output: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            doc_root = json.load(f)
    else:
        logger.info(f"Loading source: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            doc_root = json.load(f)

    rotator = ApiKeyRotator(keys_file=args.keys_file)

    elements = list(iter_elements(doc_root))
    todo = [e for e in elements
            if args.overwrite or not e.get(SUB_ENTITIES_KEY)]
    if args.limit:
        todo = todo[:args.limit]

    logger.info(f"{len(elements)} elements carry entity_concepts; "
                f"{len(todo)} to process this run "
                f"({len(elements) - len(todo)} already done / skipped).")

    processed = 0
    failed = 0
    for elem in tqdm(todo, desc="Sub-entities", unit="elem"):
        ec = elem["entity_concepts"]
        entity_names = list(ec.keys())
        prompt = build_prompt(ec)

        text = generate_with_rotation(rotator, prompt)
        if text is None:
            logger.error("Aborting: no usable API keys left. Progress saved.")
            break

        parsed = parse_response(text, entity_names)
        if parsed is None:
            failed += 1
            tid = elem.get("text_id", "?")
            logger.warning(f"Failed to parse response for element {tid}; "
                           f"storing empty breakdown.")
            parsed = {ent: [] for ent in entity_names}

        elem[SUB_ENTITIES_KEY] = parsed
        processed += 1

        if processed % CHECKPOINT_INTERVAL == 0:
            save_output(doc_root, output_path)
            stats = rotator.get_usage_stats()
            logger.info(f"Checkpoint: {processed} processed, {failed} parse-failures. "
                        f"API calls used {stats['total_used']}/{stats['total_available']}, "
                        f"active keys {stats['active_keys']}.")

    save_output(doc_root, output_path)
    logger.info(f"Done. Processed {processed} elements this run "
                f"({failed} parse-failures). Output → {output_path}")
    stats = rotator.get_usage_stats()
    logger.info(f"API usage: {stats['total_used']}/{stats['total_available']} calls, "
                f"active={stats['active_keys']}, exhausted={stats['exhausted_keys']}, "
                f"blacklisted={stats['blacklisted_keys']}.")


if __name__ == "__main__":
    main()
