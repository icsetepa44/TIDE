#!/usr/bin/env python3
"""
Entity-Concept Extraction Pipeline
====================================

Processes legal JSON documents (Bangladesh Customs Act OCR outputs),
extracts entity-concept pairs from each clause/metadata element using
the Gemini API, and outputs structured results.

Features:
  - Walks all subfolders of a root directory automatically
  - Resolves [table_n] placeholders with actual table data inline
  - One Gemini API call per clause/metadata element
  - API key rotation (round-robin) from file or environment variables
  - Robust multi-key retry: three-way error classification
    (permanent → blacklist, rate_limit → cooldown/exhaust, transient → backoff),
    server retry-delay parsing, and cooldown-waiting (ported from cluster_gemini.py)
  - Rate limiting with configurable delay
  - Robust JSON response parsing (strips markdown fences)
  - Global entities.txt accumulation (deduplicated, sorted)
  - Comprehensive logging with progress reporting

Usage:
  python entity_concept_pipeline.py
  python entity_concept_pipeline.py --root-dir ./custom_acts_SRO
  python entity_concept_pipeline.py --api-keys-file my_keys.txt --delay 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback — progress bars become no-ops
    class tqdm:  # type: ignore
        """Minimal stand-in if tqdm isn't installed: iterates transparently and
        routes .write() to print, so the pipeline still runs (without bars)."""
        def __init__(self, iterable=None, **kwargs):
            self._iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix_str(self, *a, **k):
            pass

        def close(self):
            pass

        @staticmethod
        def write(msg, file=None):
            print(msg, file=file)

# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "./Custom_Data/Data/customs_sros_OCR/"           # Root folder containing subfolders with JSONs
OUTPUT_SUFFIX = "_ec_output.json"       # Suffix for per-file output JSON
GLOBAL_ENTITIES_FILE = "entities.txt"   # Name of the global entities file (at ROOT_DIR)
MAX_RETRIES = 3                         # Max retry attempts per API call
GEMINI_MODEL = "gemini-3.1-flash-lite"       # Gemini model to use
API_KEYS_FILE = "./all_api.txt"          # Default API keys file path
CALLS_PER_DAY = 500                     # Max API calls per key per day
RATE_LIMIT_DELAY = 5.0                  # Delay between API calls (seconds)
MAX_PREFERRED_NAMES = 40                # Max preferred canonical names injected per prompt (relevance-ranked)
CHECKPOINT_INTERVAL = 100                # Save a per-file checkpoint every N processed elements
TRANSIENT_COOLDOWN = 30                 # Seconds to park a key after its same-call transient retries are exhausted

# ════════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ════════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("ec_pipeline")


class _TqdmLoggingHandler(logging.Handler):
    """Emit log records via tqdm.write so they don't corrupt an active progress
    bar. Bars and logs both share sys.stdout, so they never interleave badly."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging to stdout, tqdm-aware so progress bars and
    log lines coexist cleanly."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = _TqdmLoggingHandler()
    console.setFormatter(fmt)
    console.setLevel(level)
    logger.handlers.clear()   # avoid duplicate handlers if called more than once
    logger.addHandler(console)
    logger.setLevel(level)


# ════════════════════════════════════════════════════════════════════════════════
# API Key Management
# ════════════════════════════════════════════════════════════════════════════════

class ApiKeyRotator:
    """
    Manages multiple Gemini API keys with automatic rotation and error handling.

    API-handling strategy ported faithfully from cluster_gemini.py's
    GeminiApiKeyManager.

    Features:
      - Round-robin rotation through keys
      - Per-key usage tracking with calls_per_day limit
      - Three-way error classification: permanent / rate_limit / transient
          * permanent (leaked/revoked/invalid key) → blacklist forever
          * rate_limit (429/quota): temporary cooldown for the first 2 hits,
            then mark the key exhausted after 3 consecutive hits
          * transient (server/network): exponential cooldown, retry-able
      - Server retry-delay extraction from 429 messages ("retry in 36.8s")
      - Per-key client caching (one genai.Client per API key)
      - Distinguishes "all keys permanently dead" from "all just cooling down"

    Loads keys from:
      1. A specified file (one key per line, or comma-separated)
      2. Environment variables: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
      3. Fallback single key: GEMINI_API_KEY
    """

    def __init__(
        self,
        keys_file: str | None = None,
        calls_per_day: int = CALLS_PER_DAY,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ):
        self.keys: list[str] = []
        self.calls_per_day = calls_per_day
        self.rate_limit_delay = rate_limit_delay
        self._current_index: int = 0
        self._total_calls: int = 0

        # Per-key tracking
        self.usage_count: dict[str, int] = {}
        self.failure_count: dict[str, int] = {}
        self.cooldown_until: dict[str, float] = {}
        self._rate_limit_hits: dict[str, int] = {}  # consecutive 429s per key
        self.exhausted_keys: set[str] = set()       # quota burned (may recover next day)
        self.blacklisted_keys: set[str] = set()     # permanently invalid (leaked/revoked)
        self._clients: dict[str, genai.Client] = {}

        self._load_keys(keys_file)
        if not self.keys:
            raise ValueError(
                "No API keys found. Provide keys via:\n"
                f"  1. File: {API_KEYS_FILE} (one key per line)\n"
                "  2. Env vars: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...\n"
                "  3. Env var: GEMINI_API_KEY (single key)"
            )

        # Initialize per-key tracking dicts
        for key in self.keys:
            self.usage_count[key] = 0
            self.failure_count[key] = 0
            self.cooldown_until[key] = 0.0
            self._rate_limit_hits[key] = 0

        # Cache the first client
        self._clients[self.keys[0]] = genai.Client(api_key=self.keys[0])

        logger.info(
            f"ApiKeyRotator initialized with {len(self.keys)} key(s), "
            f"calls_per_day={calls_per_day}, delay={rate_limit_delay}s"
        )

    def _load_keys(self, keys_file: str | None) -> None:
        """Load API keys from file or environment variables."""
        # Try file-based loading
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

        # Try numbered environment variables: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
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

        # Try single environment variable
        single = os.environ.get("GEMINI_API_KEY", "").strip()
        if single:
            self.keys = [single]
            logger.info("Loaded 1 API key from GEMINI_API_KEY env var")

    @property
    def current_key(self) -> str:
        """Return the current API key."""
        return self.keys[self._current_index]

    @property
    def current_key_masked(self) -> str:
        """Return the current key with middle chars masked."""
        key = self.current_key
        return key[:4] + "..." + key[-4:] if len(key) > 8 else "****"

    def get_client(self) -> genai.Client:
        """Return a cached genai.Client for the current API key."""
        key = self.current_key
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def rotate_key(self) -> bool:
        """
        Rotate to the next available API key.
        Skips blacklisted, exhausted, and cooldown keys.
        Returns True if a valid key was found, False if all keys are unavailable.
        """
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
            logger.info(
                f"Rotated to key [{self._current_index}] "
                f"(masked: {self.current_key_masked}, "
                f"usage: {self.usage_count.get(key, 0)}/{self.calls_per_day})"
            )
            return True
        logger.error("All API keys are exhausted, blacklisted, or in cooldown!")
        return False

    def has_any_usable_key(self) -> bool:
        """
        True if any key could still be used (not blacklisted, not exhausted —
        possibly in cooldown but will recover). False only if every key is
        permanently dead.
        """
        for key in self.keys:
            if key in self.blacklisted_keys:
                continue
            if key in self.exhausted_keys:
                continue
            return True
        return False

    def park_key(self, key: str, seconds: float) -> None:
        """Apply a temporary cooldown to a specific key (used by the caller to
        park a transiently-failing key once it rotates away from it)."""
        if key in self.keys:
            self.cooldown_until[key] = time.time() + seconds

    def wait_for_cooldown_expiry(self) -> float:
        """
        Shortest remaining cooldown across all non-blacklisted, non-exhausted
        keys. Returns 0 if a key is already available.
        """
        now = time.time()
        min_wait = float("inf")
        for key in self.keys:
            if key in self.blacklisted_keys or key in self.exhausted_keys:
                continue
            remaining = self.cooldown_until.get(key, 0) - now
            if remaining <= 0:
                return 0  # already available
            min_wait = min(min_wait, remaining)
        return min_wait if min_wait != float("inf") else 0

    def report_success(self) -> None:
        """Report a successful API call — increments usage count, resets failure count."""
        key = self.current_key
        self.usage_count[key] = self.usage_count.get(key, 0) + 1
        self.failure_count[key] = 0
        self._rate_limit_hits[key] = 0
        self._total_calls += 1

    def blacklist_key(self, key: str | None = None) -> None:
        """Permanently blacklist a key (leaked/revoked/invalid) for this run."""
        key = key or self.current_key
        self.blacklisted_keys.add(key)
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
        idx = self.keys.index(key) if key in self.keys else -1
        logger.warning(
            f"Key [{idx}] ({masked}) PERMANENTLY BLACKLISTED — will not be retried."
        )

    def blacklisted_key_indices(self) -> list[int]:
        """Positions (in self.keys) of currently-blacklisted keys — for checkpointing
        without writing raw key strings to disk."""
        return sorted(self.keys.index(k) for k in self.blacklisted_keys if k in self.keys)

    def restore_blacklisted_key_indices(self, indices: list[int]) -> None:
        """Re-blacklist keys from a checkpoint by index (keys load in stable order)."""
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(self.keys):
                self.blacklisted_keys.add(self.keys[i])
                logger.info(f"Restored blacklisted key [{i}] from checkpoint")

    def report_error(self, error: Exception) -> str:
        """
        Report an API call error; classify and handle it.

        Returns the error type: 'permanent', 'rate_limit', or 'transient'.

        Rate-limit handling: the first 2 hits on a key apply a temporary
        cooldown (using the server's retry delay if given, min 45s); a 3rd
        consecutive hit marks the key exhausted (daily quota truly burned).
        """
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
                logger.warning(
                    f"Key [{self._current_index}] ({self.current_key_masked}) "
                    f"EXHAUSTED — hit rate-limit {hits} times, daily quota likely used up."
                )
            else:
                retry_delay = self._extract_retry_delay(error)
                cooldown = max(retry_delay, 45)
                self.cooldown_until[key] = time.time() + cooldown
                logger.warning(
                    f"Key [{self._current_index}] ({self.current_key_masked}) "
                    f"rate-limited (hit {hits}/3), cooldown {cooldown:.0f}s. "
                    f"Will retry after cooldown."
                )
        else:  # transient
            # Deliberately NO cooldown here. Same-call quick backoff + retry is the
            # caller's job; a cooldown is applied (via park_key) only when the loop
            # gives up on this key and rotates to another. This prevents a single-key
            # run from stalling on a long cooldown over a brief network blip.
            logger.warning(
                f"Key [{self._current_index}] ({self.current_key_masked}) "
                f"transient error: {str(error)[:150]}"
            )

        return error_type

    @staticmethod
    def _extract_retry_delay(error: Exception) -> float:
        """
        Extract the retry delay from a 429 error message.
        Handles 'retry in 36.8s' and 'retryDelay: "36s"'. Returns 0 if absent.
        """
        msg = str(error)
        match = re.search(r"retry\s+in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"retryDelay.*?'(\d+)s'", msg)
        if match:
            return float(match.group(1))
        return 0

    def classify_error_type(self, error: Exception) -> str:
        """
        Classify an API error:
          - 'permanent': key is dead forever (leaked, revoked, invalid)
          - 'rate_limit': key hit quota/rate limits (may recover later)
          - 'transient': temporary server/network error (retry after cooldown)
        """
        msg = str(error).lower()

        permanent_indicators = [
            "leaked", "api_key_invalid", "api key invalid",
            "permission_denied", "permission denied", "forbidden",
            "invalid api key", "key not found",
            "api key not valid", "api key expired", "revoked", "disabled",
        ]
        quota_indicators_in_403 = ["quota", "rate limit", "rate_limit", "resource_exhausted"]

        if any(ind in msg for ind in permanent_indicators):
            # A 403 that is actually a quota error should be treated as rate_limit
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
        """Wait for the configured rate limit delay."""
        time.sleep(self.rate_limit_delay)

    def get_usage_stats(self) -> dict:
        """Return usage statistics across all keys."""
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

    def save_key_status_report(self, output_path: Path) -> None:
        """Write a human-readable report of per-key status to a text file."""
        lines: list[str] = ["=" * 70, "API KEY STATUS REPORT",
                            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            "=" * 70, "",
                            f"Total keys: {len(self.keys)}",
                            f"Total API calls made: {self._total_calls}",
                            f"Calls per day limit: {self.calls_per_day}", ""]
        for idx, key in enumerate(self.keys):
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
            usage = self.usage_count.get(key, 0)
            if key in self.blacklisted_keys:
                status = "PERMANENTLY BLACKLISTED — leaked/invalid"
            elif key in self.exhausted_keys:
                status = "EXHAUSTED"
            elif usage > 0:
                status = f"ACTIVE — remaining {self.calls_per_day - usage}"
            else:
                status = "INTACT (unused)"
            lines.append(f"  Key [{idx}] {masked}  — usage: {usage}/{self.calls_per_day}  [{status}]")
        lines.append("=" * 70)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"Saved API key status report: {output_path}")
        except OSError as e:
            logger.warning(f"Failed to save key status report: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Folder Walking
# ════════════════════════════════════════════════════════════════════════════════

def discover_json_files(root_dir: str) -> list[Path]:
    """
    Walk all subfolders of root_dir and find the JSON file in each.
    Returns a sorted list of Path objects to *_output.json files.
    """
    root = Path(root_dir)
    if not root.is_dir():
        logger.error(f"Root directory does not exist: {root_dir}")
        return []

    json_files: list[Path] = []
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        # Skip hidden directories
        if subfolder.name.startswith("."):
            continue
        # Find JSON files in this subfolder (expect exactly one *_output.json)
        jsons_in_folder = sorted(subfolder.glob("*_output.json"))
        if not jsons_in_folder:
            # Fallback: any .json file
            jsons_in_folder = sorted(subfolder.glob("*.json"))
            # Exclude checkpoint/hidden files
            jsons_in_folder = [
                j for j in jsons_in_folder
                if not j.name.startswith(".")
                and "checkpoint" not in j.name.lower()
                and "_ec_output" not in j.name  # skip our own output files
            ]

        if jsons_in_folder:
            json_files.append(jsons_in_folder[0])
            logger.debug(f"  Found: {jsons_in_folder[0].relative_to(root)}")
        else:
            logger.warning(f"  No JSON file found in {subfolder.name}/")

    logger.info(f"Discovered {len(json_files)} JSON file(s) under {root_dir}/")
    return json_files


# ════════════════════════════════════════════════════════════════════════════════
# JSON Loading
# ════════════════════════════════════════════════════════════════════════════════

def load_json_file(json_path: Path) -> dict | None:
    """Load and return the parsed JSON data from a file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load JSON file {json_path}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# Clause Filtering
# ════════════════════════════════════════════════════════════════════════════════

def filter_processable_elements(parsed_text: list[dict]) -> list[dict]:
    """
    Filter parsed_text to only include elements that should be processed:
    - All elements of type "clause" (they don't have metadata_category)
    - Metadata elements where metadata_category == "document"
    
    Excludes metadata elements where metadata_category == "page".
    """
    processable = []
    for element in parsed_text:
        elem_type = element.get("type", "")
        if elem_type == "clause":
            processable.append(element)
        elif elem_type == "metadata":
            if element.get("metadata_category") == "document":
                processable.append(element)
            # Skip "page" metadata
        # Skip any other unknown types
    return processable


# ════════════════════════════════════════════════════════════════════════════════
# Table Resolution
# ════════════════════════════════════════════════════════════════════════════════

def _table_data_to_text(table: dict) -> str:
    """
    Convert a table's data (headers + rows) into a clean, readable
    plain-text representation using pipe-separated format.
    """
    lines = []
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if headers and any(h.strip() for h in headers if isinstance(h, str)):
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        lines.append(header_line)
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(separator)

    for row in rows:
        if isinstance(row, list):
            # Clean cell values: replace newlines within cells with spaces
            cells = []
            for cell in row:
                cell_str = str(cell) if cell is not None else ""
                cell_str = cell_str.replace("\n", " ").strip()
                cells.append(cell_str)
            lines.append("| " + " | ".join(cells) + " |")
        elif isinstance(row, dict):
            # Handle dict-style rows (unlikely but defensive)
            vals = [str(v).replace("\n", " ").strip() for v in row.values()]
            lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def resolve_clause_text(clause: dict, all_tables: list[dict]) -> str:
    """
    Resolve all [table_n] placeholders in a clause's text field.
    
    1. Takes the clause dict and the full tables list from the JSON
    2. Uses regex to find all [table_n] placeholders in clause["text"]
    3. For each placeholder, finds the matching table where:
       - table_id matches the placeholder reference (e.g., "table_1")
       - parent_text_id matches the clause's text_id
    4. Converts the matched table's data into readable plain-text
    5. Replaces the placeholder with the full table text inline
    6. If no matching table found, leaves placeholder as-is with a warning
    7. Returns the fully resolved text string
    """
    text = clause.get("text", "")
    if not text:
        return ""

    clause_id = clause.get("text_id", "")

    # Find all [table_n] placeholders (case-insensitive for robustness)
    placeholders = re.findall(r"\[table_(\d+)\]", text, re.IGNORECASE)

    if not placeholders:
        return text  # No tables to resolve

    resolved_text = text
    for table_num in placeholders:
        placeholder = f"[table_{table_num}]"
        table_id = f"table_{table_num}"

        # Search for matching table
        matched_table = None
        for table in all_tables:
            if (table.get("table_id") == table_id
                    and table.get("parent_text_id") == clause_id):
                matched_table = table
                break

        if matched_table is None:
            # Fallback: try matching by table_id alone (less strict)
            for table in all_tables:
                if table.get("table_id") == table_id:
                    matched_table = table
                    logger.debug(
                        f"  Table {table_id} matched by ID only "
                        f"(parent_text_id mismatch: expected '{clause_id}', "
                        f"got '{table.get('parent_text_id', '')}')"
                    )
                    break

        if matched_table is not None:
            table_text = _table_data_to_text(matched_table)
            resolved_text = resolved_text.replace(placeholder, f"\n{table_text}\n", 1)
        else:
            logger.warning(
                f"  No matching table found for placeholder {placeholder} "
                f"in clause '{clause_id}' — leaving placeholder as-is"
            )

    return resolved_text


# ════════════════════════════════════════════════════════════════════════════════
# Document-Level (Hierarchy) Detection
# ════════════════════════════════════════════════════════════════════════════════

# Maps a category folder fragment → canonical hierarchy level key.
# Detected from the folder name (customs_acts_OCR, customs_sros_OCR, ...),
# NOT from the OCR `document_type` field, which is frequently mislabeled
# (e.g., a Baggage RULE tagged "SRO", a GO tagged "Finance Act").
_LEVEL_BY_CATEGORY = {
    "customs_acts": "ACT",
    "customs_sros": "SRO",
    "customs_rules": "RULE",
    "customs_gos": "GO",
}

# Per-level guidance: tells the model what KIND of claim a concept should
# capture, matching each document's function in the legal hierarchy
# (Act → SRO → Rule → GO).
LEVEL_GUIDANCE = {
    "ACT": (
        "DOCUMENT LEVEL: ACT — the parent statute passed by Parliament; the "
        "supreme law and the source of all powers exercised below it.\n"
        "For this level, concepts should capture FOUNDATIONAL matter: legal "
        "definitions of terms, grants of power/authority to bodies like the "
        "National Board of Revenue, and the core substantive rule, offence, or "
        "obligation that each entity establishes."
    ),
    "SRO": (
        "DOCUMENT LEVEL: SRO (Statutory Regulatory Order) — a binding legal "
        "instrument the NBR/Government issues under powers granted by the Act.\n"
        "For this level, concepts should capture the SPECIFIC CHANGE this order "
        "makes: a new or amended duty/tax rate, an exemption or concession, a "
        "condition attached to it, or the repeal/replacement of an earlier "
        "instrument — and, where stated, the section of the Act under which it "
        "is issued."
    ),
    "RULE": (
        "DOCUMENT LEVEL: RULE — a permanent procedural framework that "
        "operationalizes the Act and SROs.\n"
        "For this level, concepts should capture the PROCEDURE or STRUCTURE: "
        "eligibility, thresholds, limits, quantities, definitions, and the "
        "step-by-step mechanism each entity is subject to (e.g., exact baggage "
        "weight/gold limits, licensing steps)."
    ),
    "GO": (
        "DOCUMENT LEVEL: GO / STANDING ORDER — an operational instruction issued "
        "by customs leadership to field officers; the lowest-altitude, most "
        "operational instrument, carrying out higher-level law in practice.\n"
        "For this level, concepts should capture the PRACTICAL INSTRUCTION: who "
        "does what, which route/desk/office/form to use, and how a higher-level "
        "provision (an Act, SRO, or Rule) is to be carried out day-to-day."
    ),
}


def detect_doc_level(json_path: Path, root_dir: str) -> str | None:
    """
    Determine the hierarchy level (ACT/SRO/RULE/GO) from the category folder
    in the path or the root_dir. Returns None if it cannot be determined.
    """
    haystack = (str(json_path) + " " + str(root_dir)).lower()
    for fragment, level in _LEVEL_BY_CATEGORY.items():
        if fragment in haystack:
            return level
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Preferred-Name Relevance Filtering
# ════════════════════════════════════════════════════════════════════════════════

# Generic legal tokens that appear in almost every clause/name and would create
# spurious "relevance" matches if counted (e.g., every name shares "customs").
_NAME_STOPWORDS = {
    "the", "of", "and", "for", "with", "any", "this", "that", "such",
    "customs", "act", "rule", "rules", "order", "orders", "general",
    "bangladesh", "national", "section", "sub", "schedule", "government",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 3 (ASCII; ignores Bangla)."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}


def _acronym(name: str) -> str:
    """Initials of the capitalized words in a name (e.g., 'Value Added Tax' → 'VAT')."""
    words = [w for w in re.findall(r"[A-Za-z]+", name) if w[:1].isupper()]
    return "".join(w[0] for w in words).upper() if len(words) >= 2 else ""


def select_relevant_names(
    clause_text: str,
    candidates: set[str] | list[str],
    max_names: int = MAX_PREFERRED_NAMES,
) -> list[str]:
    """
    Rank previously-seen canonical entity names by lexical relevance to the
    current clause and return the top `max_names`.

    Scoring (English-side only — clause Bangla is not matched):
      - full canonical name appearing verbatim in the clause → strong
      - shared significant tokens (years and longer tokens weighted higher)
      - the name's acronym appearing as a word in the clause (VAT, NBR, AEO)

    Names with no lexical signal are dropped. This keeps prompts compact and
    focused; purely-Bangla references are left to the model's own knowledge and
    the few-shot examples. Returns names sorted by score, then alphabetically.
    """
    if not candidates:
        return []

    clause_lower = clause_text.lower()
    clause_tokens = _tokenize(clause_text)

    scored: list[tuple[int, str]] = []
    for name in candidates:
        nlow = name.lower()
        score = 0

        # Full-name verbatim match (multi-word English names in the clause)
        if len(nlow) >= 4 and nlow in clause_lower:
            score += 100

        # Significant-token overlap (skip generic legal stopwords)
        for t in (_tokenize(name) - _NAME_STOPWORDS):
            if t in clause_tokens:
                if t.isdigit() and len(t) == 4:   # year, e.g. 1969
                    score += 4
                elif len(t) >= 7:
                    score += 3
                elif len(t) >= 5:
                    score += 2
                else:
                    score += 1

        # Acronym appearing as a standalone word in the raw clause
        ac = _acronym(name)
        if ac and len(ac) >= 2 and re.search(rf"\b{re.escape(ac)}\b", clause_text):
            score += 3

        if score > 0:
            scored.append((score, name))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:max_names]]


# ════════════════════════════════════════════════════════════════════════════════
# Gemini Prompt Construction
# ════════════════════════════════════════════════════════════════════════════════

def build_gemini_prompt(
    resolved_text: str,
    document_info: dict | None = None,
    element: dict | None = None,
    doc_level: str | None = None,
    preferred_names: list[str] | None = None,
) -> str:
    """
    Build the Gemini prompt for entity-concept extraction from a single
    resolved clause text. Uses precise definitions for entity and concept,
    includes few-shot examples, negative examples, and document context.
    
    Args:
        resolved_text: The clause text with table placeholders resolved.
        document_info: The document_info dict from the source JSON (Act name, date, type).
        element: The original element dict (clause or metadata) for context.
    """
    # ── Build document context header ──
    context_block = ""
    if document_info or element:
        context_parts = []
        if document_info:
            doc_name = document_info.get("file_name", "").replace(".pdf", "")
            doc_type = document_info.get("document_type", "")
            doc_date = document_info.get("document_date", "")
            if doc_name:
                context_parts.append(f"DOCUMENT: {doc_name}")
            if doc_type:
                context_parts.append(f"DOCUMENT_TYPE: {doc_type}")
            if doc_date:
                context_parts.append(f"DOCUMENT_DATE: {doc_date}")
        if element:
            elem_type = element.get("type", "")
            context_parts.append(f"ELEMENT_TYPE: {elem_type}")
            if elem_type == "clause":
                clause_num = element.get("clause_number", "")
                clause_title = element.get("clause_title", "")
                if clause_num:
                    context_parts.append(f"CLAUSE_NUMBER: {clause_num}")
                if clause_title:
                    context_parts.append(f"CLAUSE_TITLE: {clause_title}")
            elif elem_type == "metadata":
                meta_title = element.get("metadata_title", "")
                if meta_title:
                    context_parts.append(f"METADATA_TITLE: {meta_title}")
        if context_parts:
            # Caveat about DOCUMENT_TYPE only makes sense if that field is shown;
            # only point at the block below if there actually is one (doc_level set).
            has_doc_type = any(p.startswith("DOCUMENT_TYPE:") for p in context_parts)
            if has_doc_type and doc_level:
                caveat = (
                    "\nNOTE: the DOCUMENT_TYPE above is OCR-derived and often "
                    "inaccurate — rely on the DOCUMENT TYPE & FUNCTION block below "
                    "for the document's true level in the legal hierarchy."
                )
            elif has_doc_type:
                caveat = (
                    "\nNOTE: the DOCUMENT_TYPE above is OCR-derived and may be "
                    "inaccurate; treat it with caution."
                )
            else:
                caveat = ""
            context_block = (
                "═══════════════════════════════════════\n"
                "DOCUMENT CONTEXT\n"
                "═══════════════════════════════════════\n"
                "\n"
                + "\n".join(context_parts)
                + "\n\n"
                "Use this context to resolve ambiguous references like "
                "'উক্ত Act' (the said Act), 'বোর্ড' (the Board), or "
                "'this Act'. Map them to the correct canonical entity name "
                "based on the document information above."
                + caveat
                + "\n\n"
            )

    # ── Build document-type/function block (from the category folder) ──
    level_block = ""
    if doc_level and doc_level in LEVEL_GUIDANCE:
        level_block = (
            "═══════════════════════════════════════\n"
            "DOCUMENT TYPE & FUNCTION\n"
            "═══════════════════════════════════════\n"
            "\n"
            "In Bangladesh customs law the ACT (parent statute passed by "
            "Parliament) is supreme. Below it sit subordinate instruments — "
            "SROs, RULES and GOs — that all derive their force from the Act. "
            "Their relative authority is NOT fixed by document type: it depends "
            "on the enabling section of the Act, on specificity, and on recency. "
            "Note that 'SRO' is partly a publication form (Rules are frequently "
            "issued AS an SRO), so the type below indicates a document's typical "
            "FUNCTION and altitude, not a strict legal rank. Use it to judge the "
            "KIND of claim each concept should capture — do NOT use it to assert "
            "that one document overrides another.\n"
            "\n"
            + LEVEL_GUIDANCE[doc_level]
            + "\n\n"
        )

    # ── Build preferred-canonical-names block (cross-document consistency) ──
    preferred_block = ""
    if preferred_names:
        names = sorted({n.strip() for n in preferred_names if n and n.strip()})
        if names:
            preferred_block = (
                "═══════════════════════════════════════\n"
                "PREFERRED CANONICAL NAMES\n"
                "═══════════════════════════════════════\n"
                "\n"
                "These entity names have ALREADY been used elsewhere in this "
                "benchmark. If a subject in the clause below refers to the SAME "
                "thing as one of these, REUSE the exact spelling shown here "
                "instead of inventing a new variant. This keeps entities "
                "linkable across the Act, SROs, Rules, and GOs.\n"
                "\n"
                + "\n".join(f"  - {n}" for n in names)
                + "\n\n"
            )

    return (
        "You are an expert legal analyst specializing in Bangladeshi customs, "
        "taxation, and trade law. Your task is to extract entity-concept pairs "
        "from a legal clause or metadata element.\n"
        "\n"
        + context_block
        + level_block
        + preferred_block
        + "═══════════════════════════════════════\n"
        "DEFINITIONS\n"
        "═══════════════════════════════════════\n"
        "\n"
        "ENTITY: An entity is a canonical, re-identifiable SUBJECT of Bangladesh "
        "customs law — a thing that keeps the SAME identity wherever it appears "
        "across the legal hierarchy (the Act that defines it, the SRO that "
        "amends it, the Rule that operationalizes it, and the GO that gives "
        "instructions about it). It may be a law or instrument, a tax "
        "instrument, a government body or role, a commodity or schedule, a "
        "process/mechanism, or a formally defined legal term. An entity must be "
        "a noun or noun phrase. There can be multiple entities in a single "
        "clause.\n"
        "\n"
        "ENTITY NAMES ARE CROSS-DOCUMENT ANCHORS — consistency is critical:\n"
        "  - Always use the canonical, official English name, spelled IDENTICALLY "
        "every time (e.g., 'Value Added Tax', never 'VAT'; 'National Board of "
        "Revenue', never 'NBR' or 'বোর্ড'; 'Customs Act 1969', never 'the Act').\n"
        "  - Prefer the specific, registered name over a generic reference "
        "(e.g., 'Passenger (Non-Tourist) Baggage (Import) Rules 2012', not 'the "
        "Rules'; the instrument's own number, e.g., 'S.R.O. 169-AIN/2010', not "
        "'this notification').\n"
        "  - If a PREFERRED CANONICAL NAMES list is given above and the subject "
        "matches one, REUSE that exact spelling.\n"
        "\n"
        "ENTITY CATEGORIES (use these to guide consistent naming):\n"
        "  - Laws/Acts: e.g., 'Customs Act 1969', 'Finance Act 2022', 'Right to Information Act 2009'\n"
        "  - Tax Instruments: e.g., 'Customs Duty', 'Value Added Tax', 'Supplementary Duty'\n"
        "  - Government Bodies: e.g., 'National Board of Revenue', 'Customs Authority'\n"
        "  - Roles: e.g., 'Commissioner of Customs', 'Appropriate Officer of Customs'\n"
        "  - Processes/Mechanisms: e.g., 'Bonded Warehouse Operation', 'Customs Clearance', 'Risk Analysis'\n"
        "  - Documents/Schedules: e.g., 'First Schedule', 'Bangladesh Customs Tariff'\n"
        "  - Infrastructure: e.g., 'Customs Computerised Entry Processing System'\n"
        "\n"
        "CONCEPT: A concept is a meaningful, descriptive explanation of what is "
        "being said ABOUT a specific entity in this clause. It captures the "
        "central theme, action, rule, condition, or policy being applied to or "
        "associated with that entity. IMPORTANT: the KIND of claim a concept "
        "should capture depends on the document's level (see the DOCUMENT TYPE & "
        "FUNCTION block above, when present) — a foundational rule/power for an "
        "ACT, a specific legal change for an SRO, a procedure or threshold for a "
        "RULE, and a practical operational instruction for a GO. "
        "Frame each concept to match that level. A concept must be at least one "
        "complete, meaningful sentence and at most 3–4 sentences. Concepts should "
        "be written in simple, plain English using generic terms (avoid citing "
        "specific monetary amounts, exact dates, or proper names — describe them "
        "generically instead). A single entity can have multiple concepts if the "
        "clause discusses multiple distinct aspects of it.\n"
        "\n"
        "═══════════════════════════════════════\n"
        "ENTITY NAMING — RIGHT vs WRONG\n"
        "═══════════════════════════════════════\n"
        "\n"
        "WRONG entity names → RIGHT entity names:\n"
        "  ✗ 'VAT' → ✓ 'Value Added Tax'\n"
        "  ✗ 'duty' → ✓ 'Customs Duty'\n"
        "  ✗ 'the Act' → ✓ 'Customs Act 1969'\n"
        "  ✗ 'NBR' → ✓ 'National Board of Revenue'\n"
        "  ✗ 'শুল্ক আইন' → ✓ 'Customs Act 1969' (always English)\n"
        "  ✗ 'section 156' → ✓ 'Customs Offence Penalties' (entity = the subject, not the section number)\n"
        "\n"
        "═══════════════════════════════════════\n"
        "SPECIAL CLAUSE RULES\n"
        "═══════════════════════════════════════\n"
        "\n"
        "DEFINITION CLAUSES (সংজ্ঞা / Definitions):\n"
        "When the clause defines legal terms (typically using numbered or lettered "
        "sub-clauses like (ক), (খ), (a), (b), or quoted phrases followed by "
        "'অর্থ...' / 'means...'), extract EACH formally defined term as its own "
        "entity. The concept for each defined term should paraphrase the definition "
        "in plain English. Do not merge all definitions into one entity.\n"
        "\n"
        "TABLE GRANULARITY:\n"
        "When the clause contains an embedded table, extract row-level entities "
        "only when a row introduces a distinct legal subject (e.g., a specific "
        "offence, a commodity class, a form or schedule, a penalty category). "
        "For questionnaires, checklists, or compliance requirement tables, "
        "summarize by section or theme rather than extracting one entity per row.\n"
        "\n"
        "═══════════════════════════════════════\n"
        "FEW-SHOT EXAMPLES\n"
        "═══════════════════════════════════════\n"
        "\n"
        "EXAMPLE 1 (English clause with table):\n"
        "CLAUSE: \"156. Punishment for offences.- (1) Whoever commits any offence "
        "described in column 1 of the Table below shall, in addition to and not in "
        "derogation of any punishment to which he may be liable under any other law, "
        "be liable to the penalty mentioned against that offence in column 2 "
        "of the said Table. | 47A. If any person without lawful authority gains access "
        "to any Customs computerised entry processing system | such person shall be "
        "liable to a penalty not exceeding twice the amount of duty taxes involved |\"\n"
        "\n"
        "OUTPUT:\n"
        '{"entities": [{"entity": "Customs Offence Penalties", "concepts": '
        '["This clause establishes a penalty framework for customs offences, where each '
        'offence listed in a statutory table has a corresponding penalty that applies in '
        'addition to any punishment under other laws."]}, '
        '{"entity": "Customs Computerised Entry Processing System", "concepts": '
        '["Unauthorized access to or attempts to access the customs computerized system '
        'constitutes a punishable offence, with penalties proportional to the duty and '
        'taxes involved or a fixed maximum amount, whichever is higher."]}]}\n'
        "\n"
        "EXAMPLE 2 (Bangla metadata):\n"
        "CLAUSE: \"তফসিল-.... [ অর্থ আইন, ২০২২ এর ধারা ..... দ্রষ্টব্য ] "
        "[Customs Act, 1969 (Act No. IV of 1969) Gi FIRST SCHEDULE এর প্রতিস্থাপন] "
        "FIRST SCHEDULE [ Bangladesh Customs Tariff ] (See Section 18)\"\n"
        "\n"
        "OUTPUT:\n"
        '{"entities": [{"entity": "First Schedule", "concepts": '
        '["The First Schedule of the Customs Act is being replaced in its entirety by a '
        'new schedule introduced through the Finance Act."]}, '
        '{"entity": "Bangladesh Customs Tariff", "concepts": '
        '["The Bangladesh Customs Tariff is contained within the First Schedule and is '
        'referenced by Section 18 of the Customs Act."]}]}\n'
        "\n"
        "EXAMPLE 3 (Bangla definitions clause):\n"
        "CLAUSE: \"২। সংজ্ঞা।- (ক) 'বোর্ড' অর্থ জাতীয় রাজস্ব বোর্ড; "
        "(খ) 'অথরাইজড ইকোনোমিক অপারেটর' অর্থ এমন কোনো ব্যক্তি যিনি...\"\n"
        "\n"
        "OUTPUT:\n"
        '{"entities": [{"entity": "National Board of Revenue", "concepts": '
        '["The term Board is defined as the National Board of Revenue for the purposes of this law."]}, '
        '{"entity": "Authorized Economic Operator", "concepts": '
        '["An Authorized Economic Operator is defined as a person who meets specific '
        'criteria established under these rules."]}]}\n'
        "\n"
        "EXAMPLE 4 (SRO — capture the specific legal CHANGE; entities are the instruments themselves):\n"
        "CLAUSE: \"এস,আর,ও নং ১৬৯-আইন/২০১০/২২৮৭/শুল্ক — Customs Act, 1969 (IV of 1969) "
        "এর section 18 এর sub-section (2) এ প্রদত্ত ক্ষমতাবলে সরকার, অত্র বিভাগ কর্তৃক "
        "জারীকৃত নিম্নবর্ণিত প্রজ্ঞাপন এতদ্বারা বাতিল করিল, যথা :- (১) এস,আর,ও নং "
        "৪৬-আইন/২০০৯/২২৭৪/শুল্ক; এবং (২) এস,আর,ও নং ৬২-আইন/২০১০/২২৭৭/শুল্ক।\"\n"
        "OUTPUT:\n"
        '{"entities": [{"entity": "S.R.O. 46-AIN/2009", "concepts": '
        '["This earlier customs notification is repealed under the authority of Section 18 '
        'of the Customs Act."]}, '
        '{"entity": "S.R.O. 62-AIN/2010", "concepts": '
        '["This earlier customs notification is likewise rescinded by the present order."]}, '
        '{"entity": "Customs Act 1969", "concepts": '
        '["Section 18(2) of the Act empowers the Government to issue customs notifications '
        'and, by a later notification, to rescind them."]}]}\n'
        "\n"
        "EXAMPLE 5 (RULE — capture the PROCEDURE/threshold; describe the limit generically, not the amount):\n"
        "CLAUSE: \"৪। স্থল পথে আগত যাত্রীর জন্য সুবিধা।- বিদেশে অবস্থানের মেয়াদ "
        "নির্বিশেষে স্থলপথে আগত একজন যাত্রী সর্বোচ্চ ৪০০ (চারশত) মার্কিন ডলার মূল্যের "
        "ব্যাগেজ সকল প্রকার শুল্ক ও কর পরিশোধ ব্যতিরেকে আমদানি করিতে পারিবেন।\"\n"
        "OUTPUT:\n"
        '{"entities": [{"entity": "Passenger Baggage Allowance", "concepts": '
        '["The rules let a passenger arriving by land route import baggage free of all '
        'duties and taxes up to a specified maximum value, regardless of how long they '
        'stayed abroad."]}]}\n'
        "\n"
        "EXAMPLE 6 (GO / Standing Order — capture the practical OPERATIONAL INSTRUCTION):\n"
        "CLAUSE: \"৩। ট্রানজিট ও ট্রানশিপমেন্ট রুট: নির্ধারিত রুট হইবে ভারত ও বাংলাদেশের "
        "মধ্যে স্বাক্ষরিত এতদসংশ্লিষ্ট চুক্তিতে অন্তর্ভুক্ত এবং পরিশিষ্ট-৩ এ উল্লিখিত রুট।\"\n"
        "OUTPUT:\n"
        '{"entities": [{"entity": "Transit and Transshipment", "concepts": '
        '["This standing order directs that transit and transshipment movements between '
        'India and Bangladesh use only the routes fixed in the relevant agreement and '
        'listed in the order\'s appendix."]}]}\n'
        "\n"
        "EXAMPLE 7 (clause with no extractable entities):\n"
        "CLAUSE: \"3\"\n"
        "OUTPUT:\n"
        '{"entities": []}\n'
        "\n"
        "═══════════════════════════════════════\n"
        "INSTRUCTIONS\n"
        "═══════════════════════════════════════\n"
        "\n"
        "1. Read the clause text below carefully.\n"
        "2. The clause may be in English, Bangla, or mixed Bangla-English — "
        "but ALL entity and concept values MUST be written in English regardless "
        "of the source language.\n"
        "3. Be STRICT and CONSISTENT in entity naming — always use canonical, "
        "official English names. Use the same form every time.\n"
        "4. Be DESCRIPTIVE in concept formation — each concept must be a "
        "meaningful sentence (1–4 sentences) that clearly explains the legal "
        "provision or rule being applied to the entity.\n"
        "5. TABLES ARE ENTITY-RICH: When the clause text contains embedded table data "
        "(pipe-separated rows), treat the table content as equally important to the "
        "surrounding clause text. Tables in legal clauses often introduce distinct "
        "entities not mentioned in the main text — such as specific offence types, "
        "penalty structures, commodity categories, systems, or section references. "
        "Extract entities from BOTH the clause text AND the table content. Ignore "
        "only the formatting noise (pipe characters, repeated headers, cell borders) "
        "but carefully read the semantic content of every table row and column.\n"
        "6. If no meaningful entities can be extracted (e.g., the text is a "
        "page number, blank, or purely navigational), return an empty entities list.\n"
        "\n"
        "EXACT OUTPUT SCHEMA:\n"
        '{\n'
        '  "entities": [\n'
        '    {\n'
        '      "entity": "<canonical English entity name>",\n'
        '      "concepts": [\n'
        '        "<meaningful sentence describing what the clause says about this entity>",\n'
        '        "<another meaningful sentence if the clause discusses a different aspect>"\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "\n"
        "If no meaningful entities can be extracted, return:\n"
        '{\n'
        '  "entities": []\n'
        '}\n'
        "\n"
        "CLAUSE TEXT:\n"
        "---\n"
        f"{resolved_text}\n"
        "---\n"
    )


# ════════════════════════════════════════════════════════════════════════════════
# Gemini API Call with Retry
# ════════════════════════════════════════════════════════════════════════════════

_EC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity", "concepts"],
            },
        }
    },
    "required": ["entities"],
}


def call_gemini_api(
    prompt: str,
    key_rotator: ApiKeyRotator,
    model: str = GEMINI_MODEL,
    max_retries: int = MAX_RETRIES,
) -> str | None:
    """
    Call the Gemini API with the given prompt, using the multi-key retry
    strategy ported faithfully from cluster_gemini.py's get_embedding():

      - Before each attempt, skip the current key if it is blacklisted,
        exhausted, or in cooldown — rotate to the next usable key.
      - Classify every error as permanent / rate_limit / transient:
          * permanent  → blacklist the key, rotate, retry the SAME item
          * rate_limit → cooldown/exhaust the key, rotate (or wait for the
            shortest cooldown), retry the item
          * transient  → up to max_retries quick backoffs on the same key,
            then rotate away
      - Only give up when no key can serve the request (all permanently dead),
        returning None so the caller records an `api_failed` status.

    Returns the raw response text, or None if every key is unavailable.
    """
    max_total_attempts = max_retries * len(key_rotator.keys)
    last_error = None
    consecutive_transient_on_same_key = 0

    for attempt in range(max_total_attempts):
        # Ensure the current key is usable; rotate (or wait) if not.
        current_key = key_rotator.current_key
        if (
            current_key in key_rotator.blacklisted_keys
            or current_key in key_rotator.exhausted_keys
            or time.time() < key_rotator.cooldown_until.get(current_key, 0)
        ):
            if not key_rotator.rotate_key():
                if key_rotator.has_any_usable_key():
                    wait_time = key_rotator.wait_for_cooldown_expiry()
                    if wait_time > 0:
                        logger.info(
                            f"  All keys in cooldown. Waiting {wait_time:.0f}s "
                            f"for earliest recovery..."
                        )
                        time.sleep(wait_time + 1)
                        continue
                break  # truly all keys are dead
            consecutive_transient_on_same_key = 0
            continue

        try:
            key_rotator.wait_rate_limit()
            client = key_rotator.get_client()

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_EC_RESPONSE_SCHEMA,
                ),
            )

            result_text = response.text
            if result_text:
                result_text = result_text.strip()

            key_rotator.report_success()
            return result_text

        except Exception as e:
            last_error = e
            error_type = key_rotator.report_error(e)

            if error_type == "permanent":
                logger.info(
                    f"  Permanent key failure; switching to next key "
                    f"(attempt {attempt + 1})."
                )
                if not key_rotator.rotate_key() and not key_rotator.has_any_usable_key():
                    break
                consecutive_transient_on_same_key = 0

            elif error_type == "rate_limit":
                logger.info(
                    f"  Rate-limit on key; switching to next key "
                    f"(attempt {attempt + 1})."
                )
                if not key_rotator.rotate_key():
                    if key_rotator.has_any_usable_key():
                        wait_time = key_rotator.wait_for_cooldown_expiry()
                        if wait_time > 0:
                            logger.info(
                                f"  All keys temporarily unavailable. "
                                f"Waiting {wait_time:.0f}s..."
                            )
                            time.sleep(wait_time + 1)
                    else:
                        break
                consecutive_transient_on_same_key = 0

            else:  # transient
                consecutive_transient_on_same_key += 1
                if consecutive_transient_on_same_key < max_retries:
                    # Quick backoff, then retry the SAME key. No cooldown was set,
                    # so the top-of-loop guard won't rotate us away mid-backoff.
                    backoff = min(2 ** consecutive_transient_on_same_key, 10)
                    logger.info(
                        f"  Transient error; retrying same key in {backoff}s "
                        f"(transient attempt {consecutive_transient_on_same_key}/{max_retries})."
                    )
                    time.sleep(backoff)
                else:
                    # Same-key retries exhausted. Try another key; park the failing
                    # one ONLY if we successfully rotate away (so a single-key run
                    # isn't forced to wait out a cooldown — it just gives up here).
                    leaving_key = key_rotator.current_key
                    logger.info(
                        f"  Key had {consecutive_transient_on_same_key} consecutive "
                        f"transient failures; trying another key."
                    )
                    # rotate_key() returns True even for a single key (it "rotates"
                    # to itself), so confirm we actually moved to a DIFFERENT key
                    # before parking — otherwise a single-key run would park its only
                    # key and stall the next call.
                    if key_rotator.rotate_key() and key_rotator.current_key != leaving_key:
                        key_rotator.park_key(leaving_key, TRANSIENT_COOLDOWN)
                        consecutive_transient_on_same_key = 0
                    else:
                        logger.warning(
                            "  No alternate key available after transient failures; "
                            "giving up this call."
                        )
                        break

    logger.error(
        f"  All keys exhausted/blacklisted after {max_total_attempts} total "
        f"attempts. Last error: {last_error}"
    )
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Response Parsing
# ════════════════════════════════════════════════════════════════════════════════

def parse_gemini_response(raw_text: str | None) -> tuple[dict, str]:
    """
    Parse the Gemini response text as JSON.
    
    - Strips markdown fences (```json ... ```) if present
    - Attempts direct JSON parse
    - Falls back to regex extraction of JSON object
    
    Returns:
        A tuple of (parsed_dict, extraction_status) where extraction_status
        is one of: "success", "api_failed", "parse_failed".
    """
    if not raw_text:
        return {"entities": []}, "api_failed"

    text = raw_text.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, "success"
        if isinstance(result, list):
            return {"entities": result}, "success"
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find a JSON object in the text
    brace_start = text.find("{")
    if brace_start != -1:
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = text[brace_start : i + 1]
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, dict):
                            return result, "success"
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    # All parsing failed
    logger.warning(f"  Failed to parse JSON from Gemini response. Raw (first 500 chars): {raw_text[:500]}")
    return {"entities": []}, "parse_failed"


def extract_entity_concepts(parsed_response: dict) -> dict[str, list[str]]:
    """
    Convert the Gemini response format to the output entity_concepts dict.
    
    Input format:  {"entities": [{"entity": "X", "concepts": ["a", "b"]}, ...]}
    Output format: {"X": ["a", "b"], ...}
    """
    entity_concepts: dict[str, list[str]] = {}
    entities_list = parsed_response.get("entities", [])

    if not isinstance(entities_list, list):
        return entity_concepts

    for item in entities_list:
        if not isinstance(item, dict):
            continue
        entity_name = item.get("entity", "").strip()
        concepts = item.get("concepts", [])
        if not entity_name:
            continue
        if not isinstance(concepts, list):
            concepts = [str(concepts)]
        # Clean concepts
        clean_concepts = [str(c).strip() for c in concepts if str(c).strip()]
        if not clean_concepts:
            # A benchmark entity-concept pair must carry at least one concept;
            # drop entities the model returned with no usable concept.
            continue
        if entity_name in entity_concepts:
            # Merge concepts for duplicate entity names
            existing = set(entity_concepts[entity_name])
            for c in clean_concepts:
                if c not in existing:
                    entity_concepts[entity_name].append(c)
                    existing.add(c)
        else:
            entity_concepts[entity_name] = clean_concepts

    return entity_concepts


# ════════════════════════════════════════════════════════════════════════════════
# Output Writing
# ════════════════════════════════════════════════════════════════════════════════

def write_output_json(
    output_path: Path,
    source_file: str,
    processed_clauses: list[dict],
) -> None:
    """Write the per-file output JSON with all processed clauses."""
    output_data = {
        "source_file": source_file,
        "processed_clauses": processed_clauses,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"  Output written to: {output_path}")
    except OSError as e:
        logger.error(f"  Failed to write output to {output_path}: {e}")


def write_global_entities(root_dir: str, all_entities: set[str]) -> Path:
    """
    Write the global entities.txt file at the root of the root_dir.
    Contains one unique entity per line, sorted alphabetically.
    Overwrites any existing file completely.
    """
    entities_path = Path(root_dir) / GLOBAL_ENTITIES_FILE
    sorted_entities = sorted(all_entities)

    try:
        with open(entities_path, "w", encoding="utf-8") as f:
            for entity in sorted_entities:
                f.write(entity + "\n")
        logger.info(f"Global entities written to: {entities_path} ({len(sorted_entities)} entities)")
    except OSError as e:
        logger.error(f"Failed to write entities file: {e}")

    return entities_path


# ════════════════════════════════════════════════════════════════════════════════
# Element Identifier Helper
# ════════════════════════════════════════════════════════════════════════════════

def get_element_identifier(element: dict) -> str:
    """
    Get a human-readable identifier for a parsed_text element.
    For clauses: clause_number or text_id
    For metadata: metadata_title or text_id
    """
    elem_type = element.get("type", "unknown")
    if elem_type == "clause":
        return element.get("clause_number", element.get("text_id", "unknown_clause"))
    elif elem_type == "metadata":
        title = element.get("metadata_title", "").strip()
        if title:
            return f"metadata: {title[:60]}"
        return element.get("text_id", "unknown_metadata")
    return element.get("text_id", "unknown")


# ════════════════════════════════════════════════════════════════════════════════
# Output Entry Builder
# ════════════════════════════════════════════════════════════════════════════════

def _build_output_entry(
    element: dict,
    resolved_text: str,
    entity_concepts: dict[str, list[str]],
    extraction_status: str = "success",
) -> dict:
    """
    Build a type-aware output entry that preserves the correct fields
    depending on whether the element is a clause or metadata.
    
    - Clauses get: type, clause_number, clause_title, text_id
    - Metadata gets: type, metadata_category, metadata_title, text_id
    - Both get: resolved_text, entity_concepts, extraction_status
    
    extraction_status is one of:
      - "success": Gemini returned valid JSON with entities
      - "empty_text": clause had empty text, skipped
      - "api_failed": all Gemini API retries failed
      - "parse_failed": Gemini responded but JSON parsing failed
      - "error": unexpected exception during processing
    """
    elem_type = element.get("type", "unknown")
    entry: dict = {"type": elem_type}

    if elem_type == "clause":
        entry["clause_number"] = element.get("clause_number", "")
        entry["clause_title"] = element.get("clause_title", "")
    elif elem_type == "metadata":
        entry["metadata_category"] = element.get("metadata_category", "")
        entry["metadata_title"] = element.get("metadata_title", "")

    entry["text_id"] = element.get("text_id", "")
    entry["resolved_text"] = resolved_text
    entry["entity_concepts"] = entity_concepts
    entry["extraction_status"] = extraction_status
    return entry


# ════════════════════════════════════════════════════════════════════════════════
# Checkpoint / Resume (ported from cluster_gemini.py's checkpoint machinery)
# ════════════════════════════════════════════════════════════════════════════════
#
# The embedding pipeline checkpoints embeddings arrays every CHECKPOINT_INTERVAL
# items and resumes mid-run. The EC pipeline's analogue: the unit of work is a
# processable element (one Gemini call), and the natural output boundary is the
# per-file `_ec_output.json`. So we checkpoint the partial `processed_clauses`
# list (one entry per element) plus a small state file inside a `checkpoints/`
# subfolder next to the output. On restart the pipeline:
#   - skips a file whose `_ec_output.json` is already complete,
#   - salvages an incomplete final output as a checkpoint,
#   - resumes from the last checkpointed element otherwise.
#
# Deviation from the source: blacklisted keys are persisted by INDEX, not by raw
# key string — checkpoint files must never contain plaintext API keys.

def _ec_checkpoint_paths(output_path: Path) -> tuple[Path, Path]:
    """Return (clauses_checkpoint, state_checkpoint) paths for a given output file."""
    ckpt_dir = output_path.parent / "checkpoints"
    stem = output_path.stem
    return (
        ckpt_dir / f"checkpoint_{stem}_clauses.json",
        ckpt_dir / f"checkpoint_{stem}_state.json",
    )


def _harvest_entities(processed_clauses: list[dict], global_entities: set[str]) -> None:
    """Add every entity name in already-processed clauses to the global set, so
    resumed/skipped work still contributes to entities.txt and preferred-names."""
    for clause in processed_clauses:
        for name in (clause.get("entity_concepts") or {}):
            if name:
                global_entities.add(name)


def _save_ec_checkpoint(
    processed_clauses: list[dict],
    elements_completed: int,
    total_elements: int,
    clauses_path: Path,
    state_path: Path,
    source_file: str,
    key_rotator: ApiKeyRotator,
) -> None:
    """Persist partial progress: the clauses list + a state file."""
    try:
        clauses_path.parent.mkdir(parents=True, exist_ok=True)
        with open(clauses_path, "w", encoding="utf-8") as f:
            json.dump(processed_clauses, f, ensure_ascii=False, indent=2)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "source_file": source_file,
                "elements_completed": elements_completed,
                "total_elements": total_elements,
                "clauses_saved": len(processed_clauses),
                "blacklisted_key_indices": key_rotator.blacklisted_key_indices(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        logger.info(
            f"  Checkpoint saved: {elements_completed}/{total_elements} elements "
            f"({len(processed_clauses)} clauses)"
        )
    except OSError as e:
        logger.warning(f"  Failed to save checkpoint: {e}")


def _cleanup_ec_checkpoint(clauses_path: Path, state_path: Path) -> None:
    """Remove checkpoint files after a file finalizes; drop the dir if now empty."""
    for p in (clauses_path, state_path):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    ckpt_dir = clauses_path.parent
    try:
        if ckpt_dir.name == "checkpoints" and ckpt_dir.exists() and not any(ckpt_dir.iterdir()):
            ckpt_dir.rmdir()
    except OSError:
        pass


def _load_resume_state(
    output_path: Path,
    clauses_ckpt: Path,
    state_ckpt: Path,
    total_elements: int,
    relative_source: str,
    key_rotator: ApiKeyRotator,
) -> tuple[list[dict], int, bool]:
    """
    Decide where to resume from. Returns (processed_clauses, start_element, is_complete).

    - is_complete=True  → the output file already covers every element; skip it.
    - start_element > 0  → resume the per-element loop from that index.
    - start_element == 0 → start fresh.
    """
    processed_clauses: list[dict] = []
    start_element = 0

    # 1) Skip a fully-complete output, or salvage an incomplete one as a checkpoint.
    if output_path.exists():
        existing = load_json_file(output_path)
        existing_clauses = (existing or {}).get("processed_clauses", [])
        if total_elements > 0 and len(existing_clauses) == total_elements:
            logger.debug(f"  [resume] Output already complete ({total_elements} elements) — skipping.")
            return existing_clauses, total_elements, True
        if 0 < len(existing_clauses) < total_elements and not state_ckpt.exists():
            logger.info(
                f"  [resume] Found incomplete output ({len(existing_clauses)}/{total_elements}); "
                f"salvaging as checkpoint."
            )
            _save_ec_checkpoint(
                existing_clauses, len(existing_clauses), total_elements,
                clauses_ckpt, state_ckpt, relative_source, key_rotator,
            )

    # 2) Resume from a checkpoint if present and consistent with this file.
    if state_ckpt.exists() and clauses_ckpt.exists():
        try:
            with open(state_ckpt, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("total_elements") == total_elements and state.get("elements_completed", 0) > 0:
                with open(clauses_ckpt, "r", encoding="utf-8") as f:
                    saved_clauses = json.load(f)
                if len(saved_clauses) == state.get("clauses_saved", -1):
                    processed_clauses = saved_clauses
                    start_element = state["elements_completed"]
                    logger.info(
                        f"  [resume] Resumed from checkpoint: "
                        f"{start_element}/{total_elements} elements already done."
                    )
                else:
                    logger.warning("  [resume] Checkpoint clause-count mismatch; starting fresh.")
            else:
                logger.info("  [resume] Checkpoint total mismatch or empty; starting fresh.")
            key_rotator.restore_blacklisted_key_indices(state.get("blacklisted_key_indices", []))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"  [resume] Failed to load checkpoint: {e}. Starting fresh.")

    return processed_clauses, start_element, False


# ════════════════════════════════════════════════════════════════════════════════
# Main Processing Pipeline
# ════════════════════════════════════════════════════════════════════════════════

def process_single_file(
    json_path: Path,
    root_dir: str,
    key_rotator: ApiKeyRotator,
    global_entities: set[str],
    model: str = GEMINI_MODEL,
    max_retries: int = MAX_RETRIES,
    seed_entities: set[str] | None = None,
    max_preferred_names: int = MAX_PREFERRED_NAMES,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    resume: bool = True,
) -> dict:
    """
    Process a single JSON file: filter elements, resolve tables,
    call Gemini for each, collect results.

    Checkpoint/resume: a partial `processed_clauses` list is saved every
    `checkpoint_interval` elements; on restart an already-complete output is
    skipped and an interrupted file resumes from its last checkpoint
    (disabled with resume=False).

    Returns a stats dict: {clauses_processed, entities_found, skipped}
    """
    stats = {"clauses_processed": 0, "entities_found": 0, "skipped": 0}

    # Load JSON
    data = load_json_file(json_path)
    if data is None:
        stats["skipped"] = 1
        return stats

    parsed_text = data.get("parsed_text", [])
    all_tables = data.get("tables", [])
    document_info = data.get("document_info", {})

    # Hierarchy level (ACT/SRO/RULE/GO) from the category folder — authoritative,
    # unlike the often-mislabeled document_info["document_type"].
    doc_level = detect_doc_level(json_path, root_dir)
    if doc_level:
        logger.debug(f"  Document level: {doc_level}")
    else:
        logger.warning("  Could not detect document level from path — "
                       "level-aware concept guidance will be omitted")

    # Filter to processable elements
    elements = filter_processable_elements(parsed_text)
    total_elements = len(elements)
    logger.debug(f"  {total_elements} processable elements "
                 f"(from {len(parsed_text)} total parsed_text entries)")
    if document_info:
        logger.debug(f"  Document: {document_info.get('file_name', 'unknown')} "
                     f"({document_info.get('document_type', '')})")

    # Build relative path for output
    try:
        relative_source = str(json_path.relative_to(Path(root_dir).parent))
    except ValueError:
        relative_source = str(json_path)

    # Output + checkpoint paths
    output_path = json_path.parent / (json_path.stem + OUTPUT_SUFFIX)
    clauses_ckpt, state_ckpt = _ec_checkpoint_paths(output_path)

    # ── Resume bookkeeping ──
    if resume:
        processed_clauses, start_element, is_complete = _load_resume_state(
            output_path, clauses_ckpt, state_ckpt,
            total_elements, relative_source, key_rotator,
        )
        if is_complete:
            # Already done in a previous run; still contribute its entities.
            _harvest_entities(processed_clauses, global_entities)
            _cleanup_ec_checkpoint(clauses_ckpt, state_ckpt)
            return stats
    else:
        processed_clauses, start_element = [], 0

    # Entities from any resumed clauses must seed preferred-names / the global set.
    if processed_clauses:
        _harvest_entities(processed_clauses, global_entities)

    # Resume-aware progress bar: counts from start_element up to total_elements,
    # so a resumed document shows the right starting fraction. leave=False so it
    # disappears when the document finishes (the document bar in main() persists).
    elem_bar = tqdm(
        range(start_element, total_elements),
        total=total_elements,
        initial=start_element,
        desc=json_path.stem[:28],
        unit="clause",
        leave=False,
        position=1,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for pos in elem_bar:
        element = elements[pos]
        elem_id = get_element_identifier(element)
        elem_bar.set_postfix_str(str(elem_id)[:28])
        # Per-clause detail is DEBUG (the bar is the live INFO-level view).
        logger.debug(f"  Clause {pos + 1}/{total_elements}: {elem_id}")

        try:
            # Step 1: Resolve table placeholders
            resolved_text = resolve_clause_text(element, all_tables)

            if not resolved_text.strip():
                logger.debug("    Empty text — skipping")
                processed_clauses.append(
                    _build_output_entry(element, "", {}, extraction_status="empty_text")
                )
                stats["skipped"] += 1
            else:
                # Step 2: Build Gemini prompt with document context.
                # From the entities seen so far (seed + this run), inject only
                # those lexically relevant to THIS clause as preferred canonical
                # names — keeps prompts compact while preserving consistency.
                candidate_names = global_entities | (seed_entities or set())
                preferred = select_relevant_names(
                    resolved_text, candidate_names, max_names=max_preferred_names
                )
                prompt = build_gemini_prompt(
                    resolved_text,
                    document_info=document_info,
                    element=element,
                    doc_level=doc_level,
                    preferred_names=preferred or None,
                )

                # Step 3: Call Gemini API
                raw_response = call_gemini_api(
                    prompt=prompt,
                    key_rotator=key_rotator,
                    model=model,
                    max_retries=max_retries,
                )

                # Step 4: Parse response (now returns status)
                parsed, extraction_status = parse_gemini_response(raw_response)
                entity_concepts = extract_entity_concepts(parsed)

                # Step 5: Collect entities into global set
                for entity_name in entity_concepts:
                    global_entities.add(entity_name)

                # Step 6: Build type-aware output entry with status
                processed_clauses.append(
                    _build_output_entry(
                        element, resolved_text, entity_concepts,
                        extraction_status=extraction_status,
                    )
                )

                stats["clauses_processed"] += 1
                stats["entities_found"] += len(entity_concepts)

                if entity_concepts:
                    entity_names = list(entity_concepts.keys())
                    logger.debug(
                        f"    [{extraction_status}] Extracted {len(entity_concepts)} entit(ies): "
                        f"{entity_names[:5]}{'...' if len(entity_names) > 5 else ''}"
                    )
                else:
                    logger.debug(f"    [{extraction_status}] No entities extracted")

        except Exception as e:
            logger.error(f"    Error processing element '{elem_id}': {e}")
            processed_clauses.append(
                _build_output_entry(
                    element, element.get("text", ""), {},
                    extraction_status="error",
                )
            )
            stats["skipped"] += 1

        # ── Periodic checkpoint (skip the last element — finalize handles it) ──
        items_done = pos + 1
        if (checkpoint_interval > 0
                and items_done % checkpoint_interval == 0
                and items_done < total_elements):
            _save_ec_checkpoint(
                processed_clauses, items_done, total_elements,
                clauses_ckpt, state_ckpt, relative_source, key_rotator,
            )

    elem_bar.close()

    # Write per-file output JSON, then drop the checkpoint
    write_output_json(output_path, relative_source, processed_clauses)
    _cleanup_ec_checkpoint(clauses_ckpt, state_ckpt)

    return stats


def main():
    """Main entry point for the entity-concept extraction pipeline."""

    # ── Argument parsing ──
    parser = argparse.ArgumentParser(
        description="Entity-Concept Extraction Pipeline for Legal JSON Documents"
    )
    parser.add_argument(
        "--root-dir",
        default=ROOT_DIR,
        help=f"Root folder containing subfolders with JSON files (default: {ROOT_DIR})",
    )
    parser.add_argument(
        "--api-keys-file",
        default=None,
        help=f"Path to file containing Gemini API keys (default: {API_KEYS_FILE})",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help=f"Gemini model name (default: {GEMINI_MODEL})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=RATE_LIMIT_DELAY,
        help=f"Delay between API calls in seconds (default: {RATE_LIMIT_DELAY})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Max retry attempts per API call (default: {MAX_RETRIES})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--calls-per-day",
        type=int,
        default=CALLS_PER_DAY,
        help=f"Max API calls per key per day (default: {CALLS_PER_DAY})",
    )
    parser.add_argument(
        "--max-preferred-names",
        type=int,
        default=MAX_PREFERRED_NAMES,
        help=f"Max relevance-ranked preferred canonical names injected per "
             f"prompt (default: {MAX_PREFERRED_NAMES})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Save a per-file checkpoint every N processed elements "
             f"(default: {CHECKPOINT_INTERVAL}; 0 disables periodic saves)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing outputs/checkpoints and reprocess every element "
             "from scratch (overwrites _ec_output.json).",
    )
    parser.add_argument(
        "--seed-entities",
        default=None,
        help="Path to an existing entities.txt whose names are injected as "
             "PREFERRED CANONICAL NAMES for cross-document consistency (e.g., "
             "the Act's entities.txt when processing SROs/Rules/GOs). These "
             "seed names are NOT written to this run's entities.txt.",
    )
    args = parser.parse_args()

    # ── Setup ──
    setup_logging(verbose=args.verbose)
    logger.info("=" * 70)
    logger.info("Entity-Concept Extraction Pipeline — Starting")
    logger.info("=" * 70)
    logger.info(f"Root directory:  {args.root_dir}")
    logger.info(f"Model:           {args.model}")
    logger.info(f"API delay:       {args.delay}s")
    logger.info(f"Max retries:     {args.max_retries}")
    logger.info(f"Calls per day:   {args.calls_per_day}")
    logger.info(f"Resume:          {'disabled (--no-resume)' if args.no_resume else 'enabled'}"
                f", checkpoint every {args.checkpoint_interval} element(s)")

    # ── Initialize API key rotator ──
    try:
        key_rotator = ApiKeyRotator(
            keys_file=args.api_keys_file,
            calls_per_day=args.calls_per_day,
            rate_limit_delay=args.delay,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Discover JSON files ──
    json_files = discover_json_files(args.root_dir)
    if not json_files:
        logger.error("No JSON files found to process. Exiting.")
        sys.exit(1)

    # ── Load seed entities (cross-document canonical vocabulary) ──
    seed_entities: set[str] = set()
    if args.seed_entities:
        if os.path.exists(args.seed_entities):
            try:
                with open(args.seed_entities, "r", encoding="utf-8") as fh:
                    seed_entities = {ln.strip() for ln in fh if ln.strip()}
                logger.info(
                    f"Loaded {len(seed_entities)} seed entit(ies) from "
                    f"{args.seed_entities} (used as preferred canonical names)"
                )
            except OSError as e:
                logger.warning(f"Failed to read seed entities file: {e}")
        else:
            logger.warning(f"Seed entities file not found: {args.seed_entities}")

    # ── Process all files ──
    global_entities: set[str] = set()
    total_stats = {
        "files_processed": 0,
        "total_clauses": 0,
        "total_entities": 0,
        "total_skipped": 0,
    }

    # Outer progress bar over documents (position 0); the per-clause bar inside
    # process_single_file sits on position 1.
    file_bar = tqdm(
        json_files,
        desc="Documents",
        unit="doc",
        position=0,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for json_path in file_bar:
        rel = json_path.relative_to(Path(args.root_dir))
        file_bar.set_postfix_str(str(rel.parent)[:40])
        logger.debug(f"Processing {rel}")

        file_stats = process_single_file(
            json_path=json_path,
            root_dir=args.root_dir,
            key_rotator=key_rotator,
            global_entities=global_entities,
            model=args.model,
            max_retries=args.max_retries,
            seed_entities=seed_entities,
            max_preferred_names=args.max_preferred_names,
            checkpoint_interval=args.checkpoint_interval,
            resume=not args.no_resume,
        )

        total_stats["files_processed"] += 1
        total_stats["total_clauses"] += file_stats["clauses_processed"]
        total_stats["total_entities"] += file_stats["entities_found"]
        total_stats["total_skipped"] += file_stats["skipped"]
    file_bar.close()

    # ── Write global entities.txt ──
    entities_path = write_global_entities(args.root_dir, global_entities)

    # ── Write API key status report ──
    key_rotator.save_key_status_report(Path(args.root_dir) / "api_key_status.txt")

    # ── Print summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total files processed:   {total_stats['files_processed']}")
    logger.info(f"  Total clauses processed: {total_stats['total_clauses']}")
    logger.info(f"  Total clauses skipped:   {total_stats['total_skipped']}")
    logger.info(f"  Total unique entities:   {len(global_entities)}")
    logger.info(f"  Global entities file:    {entities_path}")
    logger.info(f"  Total API calls made:    {key_rotator.total_calls}")
    usage = key_rotator.get_usage_stats()
    logger.info(f"  Active keys remaining:   {usage['active_keys']}/{len(key_rotator.keys)}")
    logger.info(f"  Exhausted keys:          {usage['exhausted_keys']}")
    logger.info(f"  Blacklisted keys:        {usage['blacklisted_keys']}")
    logger.info(f"  API capacity used:       {usage['total_used']}/{usage['total_available']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
