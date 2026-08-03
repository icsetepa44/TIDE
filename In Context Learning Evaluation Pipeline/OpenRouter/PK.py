"""
Parametric knowledge evaluation of Gemini on the ICSETEP concept-QA benchmark.

Evaluates Gemini (answering from its own weights — no reference documents) on
the 8 QA dataset types under concept_qa_by_type/, using deterministic scoring
where possible (temporal_mcq, event_sorting) and — rubric v2 — an atomic
LLM-council (llm_council.judge_atoms) for the free-text types: 3 heterogeneous
judges (gemini-3.5-flash on the same rotating keys as answering, plus
anthropic/claude-opus-4.6 and openai/gpt-5.1 via OpenRouter) whose atomic
booleans are combined by per-boolean strict majority (2-of-3 quorum). Judges
emit atomic booleans only; every score is computed deterministically in
rubric_scoring.py (dates are a hard gate: per-item final_score in [0, 1],
stored with full subscores and the raw judge JSON, rubric_version
"2.0"/"2.1").

API-key rotation, error classification, cooldown, and backoff behavior are
ported faithfully from cluster_gemini_semantic.py's GeminiApiKeyManager; the
Gemini calling convention (google-genai SDK, generate_content + structured
output via response_schema) follows entity_concept_pipeline.py.

Usage:
    python PK.py                          # full run, all 8 datasets
    python PK.py --limit 2                # smoke test
    python PK.py --datasets open_ended temporal_mcq
    python PK.py --no-resume
    python PK.py --openrouter-key-file openrouter_api.txt
                                                       # OpenRouter key for the Claude/
                                                       #   GPT judges (OPENROUTER_API_KEY
                                                       #   env var takes precedence)
    python PK.py --calibration-dump 25    # 25 judged items/dataset ->
                                                       #   calibration_dump.csv for
                                                       #   human labeling           # ignore existing results

Outputs (under --results-dir, default ./param_check_results):
    <dataset>/results.json   per-instance records (checkpointed after EVERY instance)
    <dataset>/metrics.json   aggregate metrics per dataset
    summary.json             combined metrics across datasets
    param_check.log          run log
    api_key_status.txt       per-key usage/exhaustion report
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from scipy.stats import kendalltau
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

import llm_council
import rubric_scoring
from openrouter_caller import OpenRouterCaller, load_openrouter_key, \
    OPENROUTER_KEY_FILE

logger = logging.getLogger("param_check")

# ════════════════════════════════════════════════════════════════════════════════
# Configuration Defaults
# ════════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL = "gemini-3.5-flash" 
MAX_RETRIES = 3                          # max retry attempts per key before rotating
RATE_LIMIT_DELAY = 10.0                   # seconds between API calls
CALLS_PER_DAY = 500                     # max API calls per key per day
API_KEYS_FILE_PATH = "all_api.txt"       # default API keys file
RESULTS_DIR = "param_check_results"

SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR / "concept_qa_by_type"

# Fixed decoding budget for judge calls (rigor: bounded judge decoding). The
# atom JSON is small; 4096 leaves ample room for the reasoning field.
JUDGE_MAX_OUTPUT_TOKENS = 4096

# The LLM council: 3 heterogeneous judges, per-boolean strict majority voting.
# Seat 1 is native Gemini on the same rotating key pool as answering (but a
# different model than the answering model); seats 2-3 go through OpenRouter.
GEMINI_JUDGE_MODEL = "gemini-3.5-flash"
OPENROUTER_JUDGE_MODELS = ("anthropic/claude-opus-4.6", "openai/gpt-5.1")
JUDGE_COUNCIL_MODELS = (GEMINI_JUDGE_MODEL, *OPENROUTER_JUDGE_MODELS)

# Judge configuration recorded in every metrics.json/summary.json
# (module-global on purpose so evaluator and metrics signatures stay
# unchanged; ICL.py reads the same dict).
JUDGE_RUN_CONFIG = {
    "judge_council": list(JUDGE_COUNCIL_MODELS),
    "judge_quorum": 2,
    "judge_aggregation": ("per-boolean strict majority over surviving judges; "
                          "1-1 tie -> False; instance fails only if 2+ judges "
                          "fail"),
    "judge_max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    "judge_model": "council:" + "+".join(JUDGE_COUNCIL_MODELS),
    "temperature": 0.0,
}


# ════════════════════════════════════════════════════════════════════════════════
# API Key Management — ported faithfully from cluster_gemini_semantic.py
# ════════════════════════════════════════════════════════════════════════════════

class GeminiApiKeyManager:
    """
    Manages multiple Gemini API keys with automatic rotation and error handling.

    Ported faithfully from cluster_gemini_semantic.py's GeminiApiKeyManager.

    Features:
      - Round-robin rotation through keys
      - Per-key usage tracking with calls_per_day limit
      - Auto-detection of 429/quota/rate-limit errors → cooldown / exhaustion
      - Per-key failure tracking with cooldown periods
      - Client caching (one genai.Client per API key)
      - Graceful fallback when all keys are exhausted

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
        self._rate_limit_hits: dict[str, int] = {}  # track consecutive 429s
        self.exhausted_keys: set[str] = set()       # quota-exhausted (may recover next day)
        self.blacklisted_keys: set[str] = set()     # permanently invalid (leaked, revoked)
        self._clients: dict[str, genai.Client] = {}

        self._load_keys(keys_file)
        if not self.keys:
            raise ValueError(
                "No API keys found. Provide keys via:\n"
                f"  1. File: {API_KEYS_FILE_PATH} (one key per line)\n"
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
            "GeminiApiKeyManager initialized with %d key(s), "
            "calls_per_day=%d, delay=%ss",
            len(self.keys), calls_per_day, rate_limit_delay,
        )

    def _load_keys(self, keys_file: str | None) -> None:
        """Load API keys from file or environment variables."""
        # Try file-based loading
        paths_to_try: list[str] = []
        if keys_file:
            paths_to_try.append(keys_file)
        paths_to_try.append(API_KEYS_FILE_PATH)

        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    with open(path, "r") as fh:
                        content = fh.read().strip()
                    if content:
                        raw = re.split(r"[,\n]+", content)
                        self.keys = [k.strip() for k in raw if k.strip()]
                        if self.keys:
                            logger.info("Loaded %d API key(s) from %s", len(self.keys), path)
                            return
                except Exception as e:
                    logger.warning("Failed to read keys from %s: %s", path, e)

        # Try numbered environment variables: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        i = 1
        while True:
            key = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
            if not key:
                break
            self.keys.append(key)
            i += 1

        if self.keys:
            logger.info("Loaded %d API key(s) from environment variables", len(self.keys))
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
                "Rotated to key [%d] (masked: %s, usage: %d/%d)",
                self._current_index,
                self.current_key_masked,
                self.usage_count.get(key, 0),
                self.calls_per_day,
            )
            return True
        logger.error("All API keys are exhausted, blacklisted, or in cooldown!")
        return False

    def has_any_usable_key(self) -> bool:
        """
        Check if there is ANY key that could potentially be used
        (not blacklisted, not exhausted, possibly in cooldown but will recover).
        Returns False only if every key is permanently dead (blacklisted or exhausted).
        """
        for key in self.keys:
            if key in self.blacklisted_keys:
                continue
            if key in self.exhausted_keys:
                continue
            # Key exists that is either usable now or in temporary cooldown
            return True
        return False

    def report_success(self) -> None:
        """Report a successful API call — increments usage count, resets failure/rate-limit counters."""
        key = self.current_key
        self.usage_count[key] = self.usage_count.get(key, 0) + 1
        self.failure_count[key] = 0
        self._rate_limit_hits[key] = 0  # reset consecutive rate-limit counter on success
        self._total_calls += 1

    def blacklist_key(self, key: str | None = None) -> None:
        """
        Permanently blacklist a key (leaked, revoked, invalid).
        This key will never be retried during this run or any resumed run.
        """
        key = key or self.current_key
        self.blacklisted_keys.add(key)
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
        idx = self.keys.index(key) if key in self.keys else -1
        logger.warning(
            "Key [%d] (%s) PERMANENTLY BLACKLISTED — will not be retried.",
            idx, masked,
        )

    def report_error(self, error: Exception) -> str:
        """
        Report an API call error. Classifies and handles appropriately.

        Returns the error type: 'permanent', 'rate_limit', or 'transient'.

        Rate limit handling:
          - First few hits: temporary cooldown (uses retry delay from error
            if available, minimum 45s). Key stays available after cooldown.
          - After 3+ consecutive rate limits on same key: mark as exhausted
            (daily quota truly burned).
        """
        key = self.current_key
        self.failure_count[key] = self.failure_count.get(key, 0) + 1
        error_type = self.classify_error_type(error)

        if error_type == "permanent":
            self.blacklist_key(key)
        elif error_type == "rate_limit":
            # Track consecutive rate-limit hits per key
            self._rate_limit_hits[key] = self._rate_limit_hits.get(key, 0) + 1
            hits = self._rate_limit_hits[key]

            if hits >= 3:
                # This key has been rate-limited 3+ times → daily quota is
                # truly exhausted. Mark it permanently for this run.
                self.exhausted_keys.add(key)
                logger.warning(
                    "Key [%d] (%s) EXHAUSTED — hit rate-limit %d times, "
                    "daily quota likely used up.",
                    self._current_index, self.current_key_masked, hits,
                )
            else:
                # Temporary rate limit — extract retry delay and cooldown
                retry_delay = self._extract_retry_delay(error)
                cooldown = max(retry_delay, 45)  # at least 45s
                self.cooldown_until[key] = time.time() + cooldown
                logger.warning(
                    "Key [%d] (%s) rate-limited (hit %d/3), cooldown %ds. "
                    "Will retry after cooldown.",
                    self._current_index, self.current_key_masked,
                    hits, cooldown,
                )
        else:  # transient
            cooldown = min(60 * (2 ** (self.failure_count[key] - 1)), 600)
            self.cooldown_until[key] = time.time() + cooldown
            logger.warning(
                "Key [%d] (%s) transient error, cooldown %ds. Error: %s",
                self._current_index, self.current_key_masked, cooldown,
                str(error)[:150],
            )

        return error_type

    @staticmethod
    def _extract_retry_delay(error: Exception) -> float:
        """
        Extract the retry delay from a 429 error message.
        Looks for patterns like 'retry in 36.8s' or 'retryDelay: "36s"'.
        Returns 0 if no delay is found.
        """
        msg = str(error)
        # Pattern: "Please retry in 36.819990902s"
        match = re.search(r"retry\s+in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
        # Pattern: 'retryDelay': '36s'
        match = re.search(r"retryDelay.*?'(\d+)s'", msg)
        if match:
            return float(match.group(1))
        return 0

    def classify_error_type(self, error: Exception) -> str:
        """
        Classify an API error into one of three categories:
          - 'permanent': key is dead forever (leaked, revoked, invalid)
          - 'rate_limit': key hit quota/rate limits (may recover later)
          - 'transient': temporary server/network error (retry with same key after cooldown)
        """
        msg = str(error).lower()

        # Permanent errors — key will never work again
        permanent_indicators = [
            "leaked", "api_key_invalid", "api key invalid",
            "permission_denied", "permission denied",
            "forbidden",  # generic 403
            "invalid api key", "key not found",
            "api key not valid", "api key expired",
            "revoked", "disabled",
        ]
        # But exclude 403s that are actually quota errors
        quota_indicators_in_403 = ["quota", "rate limit", "rate_limit", "resource_exhausted"]

        if any(ind in msg for ind in permanent_indicators):
            # Double-check: is it actually a quota error masquerading as 403?
            if any(qi in msg for qi in quota_indicators_in_403):
                return "rate_limit"
            return "permanent"

        # Rate limit / quota errors
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
        """
        Save a human-readable report of API key status to a text file.
        Shows which keys have been used (and how much) vs. which are still intact.
        """
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("API KEY STATUS REPORT")
        lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        blacklisted_list: list[tuple[int, str, int]] = []
        used_keys: list[tuple[int, str, int]] = []
        intact_keys: list[tuple[int, str]] = []
        exhausted_list: list[tuple[int, str, int]] = []

        for idx, key in enumerate(self.keys):
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
            usage = self.usage_count.get(key, 0)

            if key in self.blacklisted_keys:
                blacklisted_list.append((idx, masked, usage))
            elif key in self.exhausted_keys:
                exhausted_list.append((idx, masked, usage))
            elif usage > 0:
                used_keys.append((idx, masked, usage))
            else:
                intact_keys.append((idx, masked))

        lines.append(f"Total keys: {len(self.keys)}")
        lines.append(f"Total API calls made: {self._total_calls}")
        lines.append(f"Calls per day limit: {self.calls_per_day}")
        lines.append("")

        lines.append("-" * 40)
        lines.append(f"BLACKLISTED KEYS ({len(blacklisted_list)}):")
        lines.append("-" * 40)
        if blacklisted_list:
            for idx, masked, usage in blacklisted_list:
                lines.append(f"  Key [{idx}] {masked}  — usage: {usage}  [PERMANENTLY BLACKLISTED — leaked/invalid]")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("-" * 40)
        lines.append(f"EXHAUSTED KEYS ({len(exhausted_list)}):")
        lines.append("-" * 40)
        if exhausted_list:
            for idx, masked, usage in exhausted_list:
                lines.append(f"  Key [{idx}] {masked}  — usage: {usage}/{self.calls_per_day}  [EXHAUSTED]")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("-" * 40)
        lines.append(f"USED (STILL ACTIVE) KEYS ({len(used_keys)}):")
        lines.append("-" * 40)
        if used_keys:
            for idx, masked, usage in used_keys:
                remaining = self.calls_per_day - usage
                lines.append(f"  Key [{idx}] {masked}  — usage: {usage}/{self.calls_per_day}  remaining: {remaining}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("-" * 40)
        lines.append(f"INTACT (UNUSED) KEYS ({len(intact_keys)}):")
        lines.append("-" * 40)
        if intact_keys:
            for idx, masked in intact_keys:
                lines.append(f"  Key [{idx}] {masked}  — usage: 0/{self.calls_per_day}  [INTACT]")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("=" * 70)

        report_text = "\n".join(lines)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Saved API key status report: %s", output_path)


# ════════════════════════════════════════════════════════════════════════════════
# Shared Gemini call wrapper (answering + judging both go through this)
# ════════════════════════════════════════════════════════════════════════════════

class GeminiCaller:
    """
    Shared Gemini text-generation wrapper with key rotation and retry.

    The retry/rotation loop is ported faithfully from
    cluster_gemini_semantic.py's get_embedding(); the API call itself uses the
    generate_content convention from entity_concept_pipeline.py.

    Every call runs at temperature=0 for reproducibility. Returns the response
    text ("" if the model returned nothing), or None if all keys are
    exhausted/blacklisted (the instance is then recorded as api_failed and
    retried on the next --resume run).
    """

    def __init__(self, key_manager: GeminiApiKeyManager, model: str = GEMINI_MODEL,
                 temperature: float = 0.0, max_output_tokens: int | None = None):
        self.key_manager = key_manager
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_calls = 0
        self.api_failures = 0
        self.rotation_count = 0

    def call(self, prompt: str, response_schema: dict | None = None,
             max_output_tokens: int | None = None) -> str | None:
        # Maximum total attempts across ALL keys to prevent infinite loops.
        max_total_attempts = MAX_RETRIES * len(self.key_manager.keys)
        last_error = None
        consecutive_transient_on_same_key = 0

        for attempt in range(max_total_attempts):
            # Before each attempt, check that the current key is still usable
            current_key = self.key_manager.current_key
            if (
                current_key in self.key_manager.blacklisted_keys
                or current_key in self.key_manager.exhausted_keys
                or time.time() < self.key_manager.cooldown_until.get(current_key, 0)
            ):
                # Current key is dead/cooling — try to rotate
                rotated = self.key_manager.rotate_key()
                if not rotated:
                    # No usable key right now
                    if self.key_manager.has_any_usable_key():
                        # Some keys are just in cooldown — wait for the shortest one
                        wait_time = self._wait_for_cooldown_expiry()
                        if wait_time > 0:
                            logger.info(
                                "All keys in cooldown. Waiting %.0fs for earliest recovery...",
                                wait_time,
                            )
                            time.sleep(wait_time + 1)
                            continue  # retry after cooldown expires
                    break  # truly all keys are dead
                self.rotation_count += 1
                consecutive_transient_on_same_key = 0
                continue  # retry immediately with new key (don't count as attempt)

            try:
                self.key_manager.wait_rate_limit()
                self.api_calls += 1

                config_kwargs: dict = {"temperature": self.temperature}
                mot = max_output_tokens or self.max_output_tokens
                if mot:
                    config_kwargs["max_output_tokens"] = mot
                if response_schema is not None:
                    config_kwargs["response_mime_type"] = "application/json"
                    config_kwargs["response_schema"] = response_schema

                response = self.key_manager.get_client().models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                result_text = response.text
                self.key_manager.report_success()
                return result_text.strip() if result_text else ""

            except Exception as e:
                last_error = e
                self.api_failures += 1
                error_type = self.key_manager.report_error(e)

                if error_type == "permanent":
                    # Key is blacklisted by report_error. Rotate immediately.
                    logger.info(
                        "Permanent key failure; switching to next key for retry "
                        "(attempt %d).", attempt + 1,
                    )
                    rotated = self.key_manager.rotate_key()
                    if not rotated:
                        if not self.key_manager.has_any_usable_key():
                            break
                    else:
                        self.rotation_count += 1
                    consecutive_transient_on_same_key = 0

                elif error_type == "rate_limit":
                    # Key cooled down / exhausted by report_error. Rotate immediately.
                    logger.info(
                        "Rate-limit on key; switching to next key for retry "
                        "(attempt %d).", attempt + 1,
                    )
                    rotated = self.key_manager.rotate_key()
                    if not rotated:
                        if self.key_manager.has_any_usable_key():
                            wait_time = self._wait_for_cooldown_expiry()
                            if wait_time > 0:
                                logger.info(
                                    "All keys temporarily unavailable. "
                                    "Waiting %.0fs...", wait_time,
                                )
                                time.sleep(wait_time + 1)
                        else:
                            break
                    else:
                        self.rotation_count += 1
                    consecutive_transient_on_same_key = 0

                else:  # transient
                    consecutive_transient_on_same_key += 1
                    if consecutive_transient_on_same_key >= MAX_RETRIES:
                        # This key keeps failing transiently — rotate away
                        logger.info(
                            "Key had %d consecutive transient failures; rotating.",
                            consecutive_transient_on_same_key,
                        )
                        rotated = self.key_manager.rotate_key()
                        if not rotated:
                            if not self.key_manager.has_any_usable_key():
                                break
                        else:
                            self.rotation_count += 1
                        consecutive_transient_on_same_key = 0
                    else:
                        backoff = min(2 ** consecutive_transient_on_same_key, 10)
                        logger.info(
                            "Transient error; retrying same key in %ds "
                            "(transient attempt %d/%d).",
                            backoff, consecutive_transient_on_same_key, MAX_RETRIES,
                        )
                        time.sleep(backoff)

        logger.error(
            "Gemini call failed — all %d keys exhausted/blacklisted after %d "
            "total attempts. Last error: %s",
            len(self.key_manager.keys), max_total_attempts, last_error,
        )
        return None

    def _wait_for_cooldown_expiry(self) -> float:
        """
        Find the shortest remaining cooldown across all non-blacklisted,
        non-exhausted keys. Returns 0 if a key is already available.
        """
        now = time.time()
        min_wait = float("inf")
        for key in self.key_manager.keys:
            if key in self.key_manager.blacklisted_keys:
                continue
            if key in self.key_manager.exhausted_keys:
                continue
            remaining = self.key_manager.cooldown_until.get(key, 0) - now
            if remaining <= 0:
                return 0  # this key is already available
            min_wait = min(min_wait, remaining)
        return min_wait if min_wait != float("inf") else 0


def make_judge_call_fn(caller):
    """call_fn for one llm_council judge seat. Wraps any judge caller
    (GeminiCaller or OpenRouterCaller — same .call signature) with the fixed
    judge decoding budget (JUDGE_MAX_OUTPUT_TOKENS, recorded in
    JUDGE_RUN_CONFIG)."""
    def judge_call(prompt: str, response_schema: dict | None = None) -> str | None:
        return caller.call(prompt, response_schema=response_schema,
                           max_output_tokens=JUDGE_RUN_CONFIG["judge_max_output_tokens"])
    return judge_call


# Set by main() (and by ICL.main via pk.JUDGE_COUNCIL /
# pk.JUDGE_CALLERS); module-global on purpose so evaluator signatures stay
# unchanged.
JUDGE_COUNCIL: list | None = None   # [(judge_name, call_fn), ...]
JUDGE_CALLERS: list = []            # callers behind the council, for run stats


def build_judge_council(key_manager: GeminiApiKeyManager,
                        openrouter_key: str) -> tuple[list, list]:
    """The 3-judge council: gemini-3.5-flash on the SAME rotating key manager
    as answering (separate GeminiCaller because the model differs), plus two
    OpenRouter judges. Returns (judges, callers): judges go to
    llm_council.judge_atoms; callers carry api_calls/api_failures for the
    run-stats report."""
    gem = GeminiCaller(key_manager, model=GEMINI_JUDGE_MODEL)
    ors = [OpenRouterCaller(m, openrouter_key) for m in OPENROUTER_JUDGE_MODELS]
    judges = [(GEMINI_JUDGE_MODEL, make_judge_call_fn(gem))]
    judges += [(c.model, make_judge_call_fn(c)) for c in ors]
    return judges, [gem, *ors]


# ════════════════════════════════════════════════════════════════════════════════
# Prompt templates
# ════════════════════════════════════════════════════════════════════════════════

PARAMETRIC_PREAMBLE = (
    "You are being tested on your knowledge of Bangladesh customs regulations: "
    "SROs (Statutory Regulatory Orders), customs and regulatory duties, baggage "
    "rules, and trade agreements. Answer from your own knowledge; no reference "
    "documents are provided."
)

OPEN_ENDED_PROMPT_TEMPLATE = """{preamble}

Question: {question}

Give a direct, factual answer in 1-4 sentences."""

SCENARIO_PROMPT_TEMPLATE = """{preamble}

Scenario:
{scenario}

Question: {question}

Give a direct, factual answer in 1-4 sentences."""

CONTEXT_MISALIGNMENT_PROMPT_TEMPLATE = """{preamble}

Context:
{context}

Question: {question}

Give a direct, factual answer in 1-4 sentences."""

MCQ_PROMPT_TEMPLATE = """{preamble}

Question: {question}

Options:
{options_block}

Choose the single best option; if you are unsure, choose the most likely one.
Think first, then answer. Respond with strict JSON only, exactly:
{{"reasoning": "<briefly weigh the options against your knowledge>", "choice_index": <0-based index of the chosen option>}}"""

EVENT_SORTING_PROMPT_TEMPLATE = """{preamble}

Task: {question}

Events (numbered, in shuffled order):
{events_block}

Arrange ALL of the numbered events in chronological order (earliest first).
Respond with strict JSON only, exactly:
{{"ordered_indices": [<each event number exactly once, earliest event first>]}}"""

PERTURB_VERDICT_PROMPT_TEMPLATE = """{preamble}

Statement:
{statement}

Is this statement factually accurate according to the actual regulations?
First check each specific detail of the statement (values, percentages, dates,
names, authorities, requirements) against your knowledge, then respond with
strict JSON only, exactly:
{{"reasoning": "<check the statement's specific facts against your knowledge>", "verdict": "CORRECT" or "INCORRECT"}}
Use "CORRECT" if the statement is accurate, "INCORRECT" if it contains an error."""

PERTURB_EXPLAIN_PROMPT_TEMPLATE = """{preamble}

Statement:
{statement}

This statement contains an error. Explain precisely WHY it is incorrect and
WHERE the error lies: identify the specific wrong detail (value, percentage,
date, name, authority, or requirement) and state what the correct fact should be."""

# Structured-output schemas (plain dicts, entity_concept_pipeline.py style)
MCQ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "choice_index": {"type": "integer"},
    },
    "required": ["reasoning", "choice_index"],
    "propertyOrdering": ["reasoning", "choice_index"],
}

EVENT_ORDER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"ordered_indices": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ordered_indices"],
}

VERDICT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["CORRECT", "INCORRECT"]},
    },
    "required": ["reasoning", "verdict"],
    "propertyOrdering": ["reasoning", "verdict"],
}

# Prompt builders for the 5 LLM-judged free-text datasets.
# Each returns (prompt, context_for_judge).
FREE_TEXT_PROMPTS = {
    # relative_time_qa questions already narrate their own anchor ("Seven
    # years after ... on June 10, 2010, ..."), so the answering prompt is
    # question-only, exactly like open_ended; seed_time/relative_offset go to
    # the JUDGE separately (anchor context + v2.1 match granularity).
    "relative_time_qa": lambda rec: (
        OPEN_ENDED_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, question=rec["question"]),
        None,
    ),
    "open_ended": lambda rec: (
        OPEN_ENDED_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, question=rec["question"]),
        None,
    ),
    "scenario_short_answer": lambda rec: (
        SCENARIO_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, scenario=rec["scenario"], question=rec["question"]),
        rec["scenario"],
    ),
    "temporal_scenario_short_answer": lambda rec: (
        SCENARIO_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, scenario=rec["scenario"], question=rec["question"]),
        rec["scenario"],
    ),
    "context_misalignment": lambda rec: (
        CONTEXT_MISALIGNMENT_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, context=rec["misaligned_context"],
            question=rec["question"]),
        rec["misaligned_context"],
    ),
}


# ════════════════════════════════════════════════════════════════════════════════
# Defensive response parsing
# ════════════════════════════════════════════════════════════════════════════════

def _load_json_defensive(raw: str):
    """Parse JSON from a model response, tolerating fences and surrounding prose."""
    cleaned = llm_council.strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_choice_index(raw: str, n_options: int) -> int | None:
    """Extract the chosen 0-based option index from an MCQ response."""
    data = _load_json_defensive(raw)
    if isinstance(data, dict) and "choice_index" in data:
        try:
            c = int(data["choice_index"])
            if 0 <= c < n_options:
                return c
        except (TypeError, ValueError):
            pass
    m = re.search(r'"?choice_index"?\s*[:=]\s*(-?\d+)', raw)
    if m:
        c = int(m.group(1))
        if 0 <= c < n_options:
            return c
    m = re.fullmatch(r"\s*\[?\s*(-?\d+)\s*\]?\s*", llm_council.strip_json_fences(raw))
    if m:
        c = int(m.group(1))
        if 0 <= c < n_options:
            return c
    return None


def parse_verdict(raw: str) -> str | None:
    """Extract CORRECT/INCORRECT from a perturbation-verdict response."""
    data = _load_json_defensive(raw)
    if isinstance(data, dict):
        v = str(data.get("verdict", "")).strip().upper()
        if v in ("CORRECT", "INCORRECT"):
            return v
    m = re.search(r'"verdict"\s*:\s*"(CORRECT|INCORRECT)"', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # \b prevents the CORRECT inside INCORRECT from matching.
    if re.search(r"\bINCORRECT\b", raw, re.IGNORECASE):
        return "INCORRECT"
    if re.search(r"\bCORRECT\b", raw, re.IGNORECASE):
        return "CORRECT"
    return None


# ── Event-sorting parsing / matching ────────────────────────────────────────────

_DATE_SUFFIX_RE = re.compile(r"\s*\((\d{4}-\d{2}-\d{2})\)\s*$")
FUZZY_MATCH_THRESHOLD = 0.85


def normalize_event_text(text: str) -> str:
    """Normalize an event string: drop the trailing ' (YYYY-MM-DD)', lowercase,
    collapse punctuation/whitespace."""
    text = _DATE_SUFFIX_RE.sub("", str(text))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _is_permutation(indices, n: int) -> bool:
    return (
        isinstance(indices, list)
        and len(indices) == n
        and all(isinstance(i, int) and not isinstance(i, bool) for i in indices)
        and sorted(indices) == list(range(n))
    )


def fuzzy_match_order(candidates: list, events_shuffled: list[str]) -> list[int] | None:
    """Map emitted event strings back to shuffled indices via normalized fuzzy
    matching. Succeeds only if a full permutation is recovered."""
    n = len(events_shuffled)
    norm_events = [normalize_event_text(e) for e in events_shuffled]
    used: set[int] = set()
    order: list[int] = []
    for cand in candidates:
        target = normalize_event_text(str(cand))
        if not target:
            continue
        best_i, best_r = None, 0.0
        for i, ne in enumerate(norm_events):
            if i in used:
                continue
            if ne == target:
                best_i, best_r = i, 1.0
                break
            r = difflib.SequenceMatcher(None, target, ne).ratio()
            if r > best_r:
                best_i, best_r = i, r
        if best_i is not None and best_r >= FUZZY_MATCH_THRESHOLD:
            used.add(best_i)
            order.append(best_i)
    return order if len(order) == n else None


def parse_event_order(raw: str, events_shuffled: list[str]) -> tuple[list[int] | None, str | None]:
    """
    Parse the model's event-sorting response into a permutation of shuffled
    indices. Returns (permutation, parse_method) or (None, None).
    Does NOT include the judge-assisted fallback (the caller handles that).
    """
    n = len(events_shuffled)
    data = _load_json_defensive(raw)

    arr = None
    if isinstance(data, dict):
        arr = data.get("ordered_indices")
        if arr is None:
            # tolerate other array-valued keys (e.g. "ordered_events")
            for v in data.values():
                if isinstance(v, list):
                    arr = v
                    break
    elif isinstance(data, list):
        arr = data

    if isinstance(arr, list):
        if _is_permutation(arr, n):
            return arr, "json_indices"
        # digit strings / floats → coerce
        try:
            coerced = [int(str(x).strip()) for x in arr]
            if _is_permutation(coerced, n):
                return coerced, "json_indices_coerced"
        except (TypeError, ValueError):
            pass
        # 1-based indices → shift down
        try:
            shifted = [int(str(x).strip()) - 1 for x in arr]
            if _is_permutation(shifted, n):
                return shifted, "json_indices_1based"
        except (TypeError, ValueError):
            pass
        # array of event strings → fuzzy map
        if all(isinstance(x, str) for x in arr):
            order = fuzzy_match_order(arr, events_shuffled)
            if order is not None:
                return order, "fuzzy_strings"

    # Plain-text fallback: try to fuzzy-match non-empty lines
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        order = fuzzy_match_order(lines, events_shuffled)
        if order is not None:
            return order, "fuzzy_lines"

    return None, None


def map_ground_truth_order(events_shuffled: list[str], answer_ordered: list[dict]) -> list[int] | None:
    """
    Map the ground-truth chronological order (answer_ordered: list of
    {event, date} dicts) onto shuffled indices. Matching is by normalized event
    text (the shuffled strings carry a trailing ' (YYYY-MM-DD)' the ordered
    ones lack), with the date suffix as fallback.
    """
    if len(events_shuffled) != len(answer_ordered):
        return None
    norm_shuffled = [normalize_event_text(e) for e in events_shuffled]
    used: set[int] = set()
    order: list[int] = []
    for item in answer_ordered:
        target = normalize_event_text(item.get("event", ""))
        candidates = [i for i, ns in enumerate(norm_shuffled) if ns == target and i not in used]
        if not candidates:
            scored = sorted(
                ((difflib.SequenceMatcher(None, target, ns).ratio(), i)
                 for i, ns in enumerate(norm_shuffled) if i not in used),
                reverse=True,
            )
            if scored and scored[0][0] >= FUZZY_MATCH_THRESHOLD:
                candidates = [scored[0][1]]
        if not candidates:
            date = str(item.get("date", ""))
            candidates = [
                i for i, e in enumerate(events_shuffled)
                if i not in used
                and (m := _DATE_SUFFIX_RE.search(e)) is not None
                and m.group(1) == date
            ]
        if not candidates:
            return None
        order.append(candidates[0])
        used.add(candidates[0])
    return order


def order_agreement_metrics(pred: list[int], gt: list[int]) -> tuple[float, float]:
    """Kendall's τ and pairwise-order accuracy between two orderings of the
    same elements (both are permutations of 0..n-1 shuffled indices)."""
    n = len(gt)
    pos_p = {e: i for i, e in enumerate(pred)}
    pos_g = {e: i for i, e in enumerate(gt)}
    tau, _ = kendalltau([pos_g[e] for e in range(n)], [pos_p[e] for e in range(n)])
    pairs = correct = 0
    for a in range(n):
        for b in range(a + 1, n):
            pairs += 1
            if (pos_p[a] < pos_p[b]) == (pos_g[a] < pos_g[b]):
                correct += 1
    pairwise = correct / pairs if pairs else 1.0
    return float(tau), pairwise


# ════════════════════════════════════════════════════════════════════════════════
# Checkpointed result store
# ════════════════════════════════════════════════════════════════════════════════

class ResultStore:
    """
    Incremental per-dataset result storage: results.json is atomically
    rewritten after EVERY instance, so an interruption never loses progress.
    Instances with a terminal status are skipped on --resume; 'api_failed'
    instances (all keys were exhausted at the time) are retried.
    """

    def __init__(self, results_dir: Path, dataset: str, model: str):
        self.dir = results_dir / dataset
        self.path = self.dir / "results.json"
        self.dataset = dataset
        self.model = model
        self.records: dict[int, dict] = {}

    def load(self) -> int:
        if not self.path.exists():
            return 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read existing %s (%s) — starting fresh.", self.path, e)
            return 0
        for rec in data.get("results", []):
            if "instance_id" in rec:
                self.records[int(rec["instance_id"])] = rec
        return len(self.records)

    def completed_ids(self) -> set[int]:
        return {iid for iid, r in self.records.items() if r.get("status") != "api_failed"}

    def upsert(self, record: dict) -> None:
        self.records[int(record["instance_id"])] = record
        self.save()

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "model": self.model,
            "results": [self.records[k] for k in sorted(self.records)],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════════
# Per-dataset evaluation
# ════════════════════════════════════════════════════════════════════════════════

def _base_record(iid: int, rec: dict) -> dict:
    return {
        "instance_id": iid,
        "type": rec.get("type"),
        "question": rec.get("question"),
        "thread_id": rec.get("thread_id"),
        "source_file": rec.get("source_file"),
    }


def _judge_anchor(dataset: str, rec: dict,
                  plan: rubric_scoring.GTDatePlan) -> str | None:
    """Anchor context handed to the judge for relative_time_qa items."""
    if dataset != "relative_time_qa" or plan.anchor_missing:
        return None
    return (f"Seed time: {rec.get('seed_time')}; "
            f"relative offset: {rec.get('relative_offset')}")


def eval_llm_judged(dataset: str, items: list[tuple[int, dict]],
                    caller: GeminiCaller, store: ResultStore) -> None:
    """Shared evaluator for the 5 free-text datasets scored under rubric v2:
    open_ended, scenario_short_answer, temporal_scenario_short_answer,
    context_misalignment, relative_time_qa. 1 answer call + 3 council judge
    calls per instance (2-of-3 quorum, per-boolean majority); judges return
    booleans only and the score is computed deterministically
    (rubric_scoring.score_base)."""
    build_prompt = FREE_TEXT_PROMPTS[dataset]
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        prompt, context = build_prompt(rec)

        # Deterministic GT-side date extraction (the judge never discovers GT
        # dates) + per-item rubric flags, recorded before any API call.
        plan = rubric_scoring.prepare_gt(dataset, rec)

        base = _base_record(iid, rec)
        base["ground_truth_answer"] = rec["answer"]
        if dataset in ("scenario_short_answer", "temporal_scenario_short_answer"):
            base["scenario"] = rec["scenario"]
        elif dataset == "context_misalignment":
            base["misaligned_context"] = rec["misaligned_context"]
            base["conflict"] = rec.get("conflict")
        elif dataset == "relative_time_qa":
            base["seed_time"] = rec.get("seed_time")
            base["relative_offset"] = rec.get("relative_offset")
        base.update(
            rubric_version=plan.rubric_version,
            gt_dates=[d.to_dict() for d in plan.gt_dates],
            required_match_granularity=plan.required_granularity,
            calendar_mismatch=plan.calendar_mismatch,
            anchor_missing=plan.anchor_missing,
            offset_granularity_unparsed=plan.offset_granularity_unparsed,
        )

        raw = caller.call(prompt)
        if raw is None:
            base.update(status="api_failed", model_answer=None, final_score=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        if not raw.strip():
            base.update(
                status="ok", model_answer=raw, answered_empty=True,
                final_score=0.0, date_score=None,
                n_gt_dates=len(plan.gt_dates), n_matched=0,
                meaning_correct=False, uncertain=False, judge_atoms=None,
                score_note="Auto-scored 0.0: model returned an empty answer "
                           "(no judge call made).",
            )
        else:
            model_dates = rubric_scoring.normalize_dates(
                raw, include_bare_years=True)
            ar = llm_council.judge_atoms(
                JUDGE_COUNCIL,
                question=rec["question"],
                ground_truth=rec["answer"],
                model_answer=raw,
                task_type=dataset,
                gt_dates=plan.gt_dates,
                context=context,
                anchor=_judge_anchor(dataset, rec, plan),
                required_granularity=plan.required_granularity,
                extra_gt=({"conflict": rec.get("conflict")}
                          if dataset == "context_misalignment" else None),
            )
            if ar.error == "judge_api_failed":
                # Council below quorum with an API outage contributing: treat
                # like an answer-call failure so the instance is retried on
                # --resume. (A single judge outage does NOT land here — the
                # council degrades to 2-of-3.)
                base.update(status="api_failed", model_answer=raw,
                            final_score=None, judge_error=ar.error,
                            judge_errors=ar.judge_errors or None)
                store.upsert(base)
                logger.error("[%s] %d/%d (#%d) JUDGE API FAILED — will retry "
                             "on resume", dataset, pos, len(items), iid)
                continue
            if ar.atoms is None:
                base.update(status="judge_failed", model_answer=raw,
                            final_score=None, judge_error=ar.error,
                            judge_errors=ar.judge_errors or None,
                            judge_raw_response=ar.raw_response)
            else:
                matches = rubric_scoring.combine_matches(
                    plan.gt_dates, model_dates, ar.atoms,
                    plan.required_granularity)
                sr = rubric_scoring.score_base(plan.gt_dates, matches, ar.atoms)
                base.update(status="ok", model_answer=raw)
                base.update(sr.to_record_fields())
                base.update(
                    meaning_rationale=ar.atoms.meaning_rationale,
                    uncertain=ar.atoms.uncertain,
                    # Judge's self-reported confidence (0-100); logged only.
                    judge_confidence=ar.atoms.confidence,
                    contradicting_dates_in_answer=
                        ar.atoms.contradicting_dates_in_answer,
                    judge_atoms={
                        "date_matched": ar.atoms.date_matched,
                        "model_expressions": ar.atoms.model_expressions,
                        "contradicting_dates_in_answer":
                            ar.atoms.contradicting_dates_in_answer,
                        "meaning_correct": ar.atoms.meaning_correct,
                        "meaning_rationale": ar.atoms.meaning_rationale,
                        "uncertain": ar.atoms.uncertain,
                        "confidence": ar.atoms.confidence,
                        "extras": ar.atoms.extras,
                    },
                    judge_atoms_raw_response=ar.raw_response,
                    judge_repair_retry_used=ar.repair_retry_used,
                    judge_council={
                        "models": list(JUDGE_COUNCIL_MODELS),
                        "per_judge_atoms": ar.per_judge_atoms,
                        "raw_responses": ar.raw_responses,
                        "errors": ar.judge_errors or {},
                        "n_ok": len(ar.per_judge_atoms),
                    },
                )
                if ar.atoms.vote_splits:
                    base["vote_splits"] = ar.atoms.vote_splits
                if dataset == "context_misalignment":
                    # Logged-only conflict-behavior atoms (never scored).
                    base["context_atoms"] = {
                        k: ar.atoms.extras.get(k)
                        for k in ("followed_context", "followed_parametric",
                                  "conflict_acknowledged", "blended_sources")}

        base["elapsed_s"] = round(time.time() - t0, 2)
        store.upsert(base)
        logger.info("[%s] %d/%d (#%d) -> score=%s (%.1fs)",
                    dataset, pos, len(items), iid, base.get("final_score"),
                    base["elapsed_s"])


def eval_temporal_mcq(dataset: str, items: list[tuple[int, dict]],
                      caller: GeminiCaller, store: ResultStore) -> None:
    """Deterministic MCQ scoring against answer_index (0-based). 1 call per
    instance, no LLM judge."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        options = rec["options"]
        options_block = "\n".join(f"Option {i}: {opt}" for i, opt in enumerate(options))
        prompt = MCQ_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, question=rec["question"],
            options_block=options_block,
        )

        base = _base_record(iid, rec)
        base["options"] = options
        base["answer_index"] = rec["answer_index"]
        base["answer_text"] = rec.get("answer")

        raw = caller.call(prompt, response_schema=MCQ_RESPONSE_SCHEMA)
        if raw is None:
            base.update(status="api_failed", model_answer=None,
                        predicted_index=None, format_failure=False, correct=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        choice = parse_choice_index(raw, len(options))
        format_failure = choice is None
        if format_failure:
            logger.warning("[%s] #%d FORMAT FAILURE — unparseable choice: %.150s",
                           dataset, iid, raw)
        base.update(
            status="ok",
            model_answer=raw,
            predicted_index=choice,
            format_failure=format_failure,
            correct=(choice == rec["answer_index"]) if choice is not None else False,
        )
        base["elapsed_s"] = round(time.time() - t0, 2)
        store.upsert(base)
        logger.info("[%s] %d/%d (#%d) pred=%s gt=%s %s (%.1fs)",
                    dataset, pos, len(items), iid, choice, rec["answer_index"],
                    "OK" if base["correct"] else "WRONG", base["elapsed_s"])


def eval_event_sorting(dataset: str, items: list[tuple[int, dict]],
                       caller: GeminiCaller, store: ResultStore) -> None:
    """Deterministic ordering evaluation. 1 answer call per instance; one
    judge-assisted extraction call ONLY if parsing/fuzzy matching fails."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        events = rec["events_shuffled"]
        gt_order = map_ground_truth_order(events, rec["answer_ordered"])

        base = _base_record(iid, rec)
        base["events_shuffled"] = events
        base["answer_ordered"] = rec["answer_ordered"]
        base["ground_truth_order_indices"] = gt_order

        if gt_order is None:
            # Data problem, not a model problem — no API call is spent.
            base.update(status="gt_mapping_failed", model_answer=None,
                        predicted_order_indices=None)
            store.upsert(base)
            logger.error("[%s] #%d could not map answer_ordered onto "
                         "events_shuffled — instance skipped", dataset, iid)
            continue

        events_block = "\n".join(f"Event {i}: {e}" for i, e in enumerate(events))
        prompt = EVENT_SORTING_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, question=rec["question"],
            events_block=events_block,
        )
        raw = caller.call(prompt, response_schema=EVENT_ORDER_RESPONSE_SCHEMA)
        if raw is None:
            base.update(status="api_failed", model_answer=None,
                        predicted_order_indices=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        pred, method = parse_event_order(raw, events)
        if pred is None:
            # Judge-assisted parsing fallback (exactly one extra call).
            logger.warning("[%s] #%d deterministic parsing failed — trying "
                           "judge-assisted extraction", dataset, iid)
            pred = llm_council.extract_event_order(caller.call, events, raw)
            if pred is not None:
                method = "judge_assisted"

        base["model_answer"] = raw
        base["parse_method"] = method
        base["judge_assisted_parse"] = method == "judge_assisted"
        if pred is None:
            base.update(status="format_failure", predicted_order_indices=None,
                        exact_match=False, kendall_tau=None, pairwise_accuracy=None)
            logger.warning("[%s] #%d FORMAT FAILURE — ordering unrecoverable: %.150s",
                           dataset, iid, raw)
        else:
            tau, pairwise = order_agreement_metrics(pred, gt_order)
            base.update(
                status="ok",
                predicted_order_indices=pred,
                exact_match=pred == gt_order,
                kendall_tau=round(tau, 4),
                pairwise_accuracy=round(pairwise, 4),
            )
        base["elapsed_s"] = round(time.time() - t0, 2)
        store.upsert(base)
        logger.info("[%s] %d/%d (#%d) exact=%s tau=%s (%.1fs)",
                    dataset, pos, len(items), iid, base.get("exact_match"),
                    base.get("kendall_tau"), base["elapsed_s"])


def eval_perturbation(dataset: str, items: list[tuple[int, dict]],
                      caller: GeminiCaller, store: ResultStore) -> None:
    """
    Two-step perturbation detection:
      Step A (1 call): CORRECT/INCORRECT verdict on the statement
      (deterministic scoring, unchanged in rubric v2).
      Step B (1 call, only if the model said INCORRECT): ask why/where it is
      wrong; the explanation is scored under rubric v2 — 3 council judge
      calls (task_type "perturbation_explanation", extras incl.
      perturbation_localized, 2-of-3 quorum with per-boolean majority), then
      rubric_scoring.score_perturbation_explanation applies the base rubric
      gated on localization. For false positives (ground truth says the
      statement is accurate) the explanation is auto-scored 0.0 without a
      judge call.
    """
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        statement = rec["statement"]

        base = _base_record(iid, rec)
        base.update(
            statement=statement,
            is_perturbed=rec["is_perturbed"],
            perturbed_fact=rec.get("perturbed_fact"),
            correct_fact=rec.get("correct_fact"),
            ground_truth_answer=rec["answer"],
            rubric_version=rubric_scoring.RUBRIC_VERSION,
        )

        # ── Step A: verdict ──
        verdict_prompt = PERTURB_VERDICT_PROMPT_TEMPLATE.format(
            preamble=PARAMETRIC_PREAMBLE, statement=statement)
        raw_verdict = caller.call(verdict_prompt, response_schema=VERDICT_RESPONSE_SCHEMA)
        if raw_verdict is None:
            base.update(status="api_failed", api_failed_stage="verdict")
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED (verdict) — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        verdict = parse_verdict(raw_verdict)
        pred_perturbed = None if verdict is None else (verdict == "INCORRECT")
        base.update(
            model_verdict_raw=raw_verdict,
            model_verdict=verdict,
            predicted_perturbed=pred_perturbed,
            detection_format_failure=verdict is None,
            detection_correct=(pred_perturbed == rec["is_perturbed"])
                              if pred_perturbed is not None else False,
        )
        if verdict is None:
            logger.warning("[%s] #%d FORMAT FAILURE — unparseable verdict: %.150s",
                           dataset, iid, raw_verdict)

        # ── Step B: explanation (only when the model said INCORRECT) ──
        if verdict == "INCORRECT":
            explain_prompt = PERTURB_EXPLAIN_PROMPT_TEMPLATE.format(
                preamble=PARAMETRIC_PREAMBLE, statement=statement)
            raw_expl = caller.call(explain_prompt)
            if raw_expl is None:
                base.update(status="api_failed", api_failed_stage="explanation")
                store.upsert(base)
                logger.error("[%s] %d/%d (#%d) API FAILED (explanation) — will retry on resume",
                             dataset, pos, len(items), iid)
                continue
            base["model_explanation"] = raw_expl

            if rec["is_perturbed"]:
                plan = rubric_scoring.prepare_gt(dataset, rec)
                base.update(
                    gt_dates=[d.to_dict() for d in plan.gt_dates],
                    calendar_mismatch=plan.calendar_mismatch,
                )
                if not raw_expl.strip():
                    base.update(
                        explanation_final_score=0.0,
                        explanation_meaning_correct=False,
                        perturbation_localized=False,
                        explanation_score_note="Auto-scored 0.0: model "
                            "returned an empty explanation (no judge call "
                            "made).",
                    )
                else:
                    ar = llm_council.judge_atoms(
                        JUDGE_COUNCIL,
                        question=rec["question"],
                        ground_truth=rec["answer"],
                        model_answer=raw_expl,
                        task_type="perturbation_explanation",
                        gt_dates=plan.gt_dates,
                        context=statement,
                        extra_gt={"perturbed_fact": rec.get("perturbed_fact"),
                                  "correct_fact": rec.get("correct_fact")},
                    )
                    if ar.error == "judge_api_failed":
                        base.update(status="api_failed",
                                    api_failed_stage="explanation_judge",
                                    explanation_judge_error=ar.error,
                                    explanation_judge_errors=
                                        ar.judge_errors or None)
                        store.upsert(base)
                        logger.error("[%s] %d/%d (#%d) JUDGE API FAILED "
                                     "(explanation) — will retry on resume",
                                     dataset, pos, len(items), iid)
                        continue
                    if ar.atoms is None:
                        base.update(
                            explanation_judge_error=ar.error,
                            explanation_judge_errors=ar.judge_errors or None,
                            explanation_judge_raw_response=ar.raw_response,
                            explanation_final_score=None,
                        )
                    else:
                        model_dates = rubric_scoring.normalize_dates(
                            raw_expl, include_bare_years=True)
                        matches = rubric_scoring.combine_matches(
                            plan.gt_dates, model_dates, ar.atoms)
                        sr = rubric_scoring.score_perturbation_explanation(
                            plan.gt_dates, matches, ar.atoms)
                        base.update(
                            explanation_final_score=sr.final_score,
                            explanation_date_score=sr.date_score,
                            explanation_n_gt_dates=sr.n_gt_dates,
                            explanation_n_matched=sr.n_matched,
                            explanation_meaning_correct=
                                sr.meaning_correct_effective,
                            explanation_hard_gate_applied=sr.hard_gate_applied,
                            explanation_date_matches=[m.to_dict()
                                                      for m in sr.date_matches],
                            explanation_uncertain=ar.atoms.uncertain,
                            explanation_judge_confidence=ar.atoms.confidence,
                            perturbation_flagged=
                                ar.atoms.extras.get("perturbation_flagged"),
                            perturbation_localized=
                                ar.atoms.extras.get("perturbation_localized"),
                            localized_element=
                                ar.atoms.extras.get("localized_element"),
                            correction_given=
                                ar.atoms.extras.get("correction_given"),
                            # Logged only — never folded into the score (v2).
                            correction_correct=
                                ar.atoms.extras.get("correction_correct"),
                            explanation_judge_atoms={
                                "date_matched": ar.atoms.date_matched,
                                "meaning_correct": ar.atoms.meaning_correct,
                                "meaning_rationale": ar.atoms.meaning_rationale,
                                "contradicting_dates_in_answer":
                                    ar.atoms.contradicting_dates_in_answer,
                                "uncertain": ar.atoms.uncertain,
                                "confidence": ar.atoms.confidence,
                                "extras": ar.atoms.extras,
                            },
                            explanation_judge_raw_response=ar.raw_response,
                            explanation_judge_council={
                                "models": list(JUDGE_COUNCIL_MODELS),
                                "per_judge_atoms": ar.per_judge_atoms,
                                "raw_responses": ar.raw_responses,
                                "errors": ar.judge_errors or {},
                                "n_ok": len(ar.per_judge_atoms),
                            },
                        )
                        if ar.atoms.vote_splits:
                            base["explanation_vote_splits"] = ar.atoms.vote_splits
            else:
                base.update(
                    explanation_final_score=0.0,
                    explanation_meaning_correct=False,
                    explanation_score_note=(
                        "Auto-scored 0.0: the ground truth says the statement "
                        "is accurate, so an explanation of why it is wrong "
                        "cannot match the ground truth (no judge call made)."),
                    explanation_autofailed_fp=True,
                )

        base["status"] = "ok"
        base["elapsed_s"] = round(time.time() - t0, 2)
        store.upsert(base)
        logger.info("[%s] %d/%d (#%d) verdict=%s gt_perturbed=%s detection=%s (%.1fs)",
                    dataset, pos, len(items), iid, verdict, rec["is_perturbed"],
                    "OK" if base["detection_correct"] else "WRONG", base["elapsed_s"])


# ════════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════════

def _score_mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    return round(sum(vals) / len(vals), 4) if vals else None


def _confidence_stats(rows: list[dict], key: str) -> tuple[float | None, int]:
    """Mean of the judge's self-reported confidence over rows where it exists.
    Auto-scored records (empty answers, false-positive explanations) carry no
    judge confidence and are excluded. Logged-only — never part of a score."""
    vals = [float(r[key]) for r in rows
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
    mean = round(sum(vals) / len(vals), 2) if vals else None
    return mean, len(vals)


def _metrics_envelope(dataset: str, model: str, n_dataset_total: int, records: dict) -> dict:
    recs = list(records.values())
    return {
        "dataset": dataset,
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_dataset_total": n_dataset_total,
        "n_with_records": len(recs),
        "n_api_failed": sum(1 for r in recs if r.get("status") == "api_failed"),
    }


def compute_rubric_metrics(records: dict, n_dataset_total: int,
                           dataset: str, model: str) -> dict:
    """Rubric-v2 aggregate metrics for the LLM-judged free-text datasets.
    All per-item scores were computed deterministically from judge atoms
    (rubric_scoring); this only averages and counts."""
    m = _metrics_envelope(dataset, model, n_dataset_total, records)
    non_failed = [r for r in records.values() if r.get("status") != "api_failed"]
    # Records from a pre-v2 run carry no rubric_version and no final_score —
    # they are excluded from every aggregate, not silently mixed in.
    legacy = [r for r in non_failed if "rubric_version" not in r]
    if legacy:
        logger.warning(
            "[%s] %d record(s) predate rubric v2 and are EXCLUDED from "
            "aggregates — re-run with a fresh --results-dir for a clean v2 run.",
            dataset, len(legacy))
    evaluated = [r for r in non_failed if "rubric_version" in r]
    scored = [r for r in evaluated
              if isinstance(r.get("final_score"), (int, float))]
    judge_error = [r for r in evaluated if r.get("status") == "judge_failed"]
    score_sum = sum(r["final_score"] for r in scored)

    prov_det = prov_judge = 0
    for r in scored:
        for dm in r.get("date_matches") or []:
            if not dm.get("matched"):
                continue
            if dm.get("match_source") == "deterministic":
                prov_det += 1
            elif dm.get("match_source") == "judge":
                prov_judge += 1

    dist: dict[str, int] = {}
    for r in scored:
        key = f"{r['final_score']:.4g}"
        dist[key] = dist.get(key, 0) + 1

    m.update({
        "n_evaluated": len(evaluated),
        "n_scored": len(scored),
        "n_judge_error": len(judge_error),
        "n_legacy_records_excluded": len(legacy),
        "n_empty_answers": sum(1 for r in evaluated if r.get("answered_empty")),
        "n_uncertain": sum(1 for r in scored if r.get("uncertain")),
        "n_calendar_mismatch": sum(1 for r in evaluated
                                   if r.get("calendar_mismatch")),
        "mean_final_score": round(score_sum / len(scored), 4) if scored else None,
        "mean_final_score_counting_judge_errors_as_zero":
            round(score_sum / (len(scored) + len(judge_error)), 4)
            if (scored or judge_error) else None,
        "mean_date_score": _score_mean(
            [r for r in scored if r.get("n_gt_dates")], "date_score"),
        "meaning_correct_rate":
            round(sum(1 for r in scored if r.get("meaning_correct"))
                  / len(scored), 4) if scored else None,
        "full_credit_rate":
            round(sum(1 for r in scored if r["final_score"] == 1.0)
                  / len(scored), 4) if scored else None,
        "zero_score_rate":
            round(sum(1 for r in scored if r["final_score"] == 0.0)
                  / len(scored), 4) if scored else None,
        "score_distribution": dict(sorted(dist.items())),
        "n_date_matches_deterministic": prov_det,
        "n_date_matches_judge_only": prov_judge,
        # Judge self-reported confidence (0-100, logged-only, never scored).
        "n_confidence_scored": _confidence_stats(scored, "judge_confidence")[1],
        "judge_confidence_mean": _confidence_stats(scored, "judge_confidence")[0],
        "judge_confidence_mean_when_meaning_correct": _confidence_stats(
            [r for r in scored if r.get("meaning_correct")],
            "judge_confidence")[0],
        "judge_confidence_mean_when_meaning_wrong": _confidence_stats(
            [r for r in scored if r.get("meaning_correct") is False],
            "judge_confidence")[0],
        "rubric_version": (rubric_scoring.RUBRIC_VERSION_RELATIVE_TIME
                           if dataset == "relative_time_qa"
                           else rubric_scoring.RUBRIC_VERSION),
        "judge_config": dict(JUDGE_RUN_CONFIG),
        "run_counters": {
            "items_scored": len(scored),
            "judge_error": len(judge_error),
            "uncertain": sum(1 for r in scored if r.get("uncertain")),
            "calendar_mismatch": sum(1 for r in evaluated
                                     if r.get("calendar_mismatch")),
            "anchor_missing": sum(1 for r in evaluated
                                  if r.get("anchor_missing")),
            "offset_granularity_unparsed":
                sum(1 for r in evaluated
                    if r.get("offset_granularity_unparsed")),
        },
        "scoring": "Rubric v2: atomic LLM-judge booleans (Gemini, "
                   "temperature=0) + deterministic Python scoring "
                   "(rubric_scoring.score_base). Dates are a hard gate; "
                   "judge errors are excluded from mean_final_score.",
    })

    if dataset == "context_misalignment":
        with_atoms = [r for r in scored if isinstance(r.get("context_atoms"), dict)]
        n_atoms = len(with_atoms)

        def rate(key: str) -> float | None:
            if not n_atoms:
                return None
            return round(sum(1 for r in with_atoms
                             if r["context_atoms"].get(key)) / n_atoms, 4)

        m["conflict_behavior"] = {
            "n_with_conflict_atoms": n_atoms,
            "context_adherence_rate": rate("followed_context"),
            "parametric_adherence_rate": rate("followed_parametric"),
            "conflict_acknowledgement_rate": rate("conflict_acknowledged"),
            "blended_sources_rate": rate("blended_sources"),
            "note": "Logged-only judge atoms; never part of final_score. A "
                    "single scalar would hide exactly the conflict-resolution "
                    "behavior this split measures.",
        }
        m["context_adherence_rate"] = m["conflict_behavior"]["context_adherence_rate"]
        m["conflict_acknowledgement_rate"] = \
            m["conflict_behavior"]["conflict_acknowledgement_rate"]

    if dataset == "relative_time_qa":
        gran_dist: dict[str, int] = {}
        for r in evaluated:
            key = str(r.get("required_match_granularity"))
            gran_dist[key] = gran_dist.get(key, 0) + 1
        m.update({
            "n_anchor_missing": sum(1 for r in evaluated
                                    if r.get("anchor_missing")),
            "n_offset_granularity_unparsed":
                sum(1 for r in evaluated
                    if r.get("offset_granularity_unparsed")),
            "granularity_distribution": dict(sorted(gran_dist.items())),
        })
    return m


def compute_mcq_metrics(records: dict, n_dataset_total: int,
                        dataset: str, model: str) -> dict:
    m = _metrics_envelope(dataset, model, n_dataset_total, records)
    evaluated = [r for r in records.values() if r.get("status") == "ok"]
    if not evaluated:
        m.update({"n_evaluated": 0, "accuracy": None})
        return m
    y_true = [r["answer_index"] for r in evaluated]
    # Format failures are scored as a non-existent class (-1): always wrong,
    # and they depress recall of the true class in the macro average.
    y_pred = [r["predicted_index"] if r["predicted_index"] is not None else -1
              for r in evaluated]
    labels = sorted({idx for idx in y_true} | {p for p in y_pred if p >= 0})
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )
    m.update({
        "n_evaluated": len(evaluated),
        "n_correct": sum(1 for r in evaluated if r.get("correct")),
        "accuracy": round(sum(1 for r in evaluated if r.get("correct")) / len(evaluated), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "labels": labels,
        "format_failure_count": sum(1 for r in evaluated if r.get("format_failure")),
        "scoring": "Deterministic: parsed choice_index vs ground-truth "
                   "answer_index (0-based). Macro-averaged P/R/F1 over option "
                   "indices as classes (sklearn, zero_division=0). Format "
                   "failures count as wrong.",
    })
    return m


def compute_event_sorting_metrics(records: dict, n_dataset_total: int,
                                  dataset: str, model: str) -> dict:
    m = _metrics_envelope(dataset, model, n_dataset_total, records)
    evaluated = [r for r in records.values()
                 if r.get("status") in ("ok", "format_failure")]
    parsed = [r for r in evaluated if r.get("status") == "ok"]
    n_eval, n_parsed = len(evaluated), len(parsed)
    taus = [r["kendall_tau"] for r in parsed if r.get("kendall_tau") is not None]
    pws = [r["pairwise_accuracy"] for r in parsed if r.get("pairwise_accuracy") is not None]
    n_exact = sum(1 for r in parsed if r.get("exact_match"))
    m.update({
        "n_evaluated": n_eval,
        "n_parsed": n_parsed,
        "format_failure_count": n_eval - n_parsed,
        "judge_assisted_parse_count": sum(1 for r in parsed if r.get("judge_assisted_parse")),
        "n_gt_mapping_failed": sum(1 for r in records.values()
                                   if r.get("status") == "gt_mapping_failed"),
        "exact_match_accuracy": round(n_exact / n_eval, 4) if n_eval else None,
        "exact_match_accuracy_parsed_only": round(n_exact / n_parsed, 4) if n_parsed else None,
        "kendall_tau_mean": round(sum(taus) / len(taus), 4) if taus else None,
        "pairwise_order_accuracy": round(sum(pws) / len(pws), 4) if pws else None,
        "pairwise_order_accuracy_counting_failures_as_zero":
            round(sum(pws) / n_eval, 4) if n_eval else None,
        "scoring": "Deterministic permutation comparison. exact_match_accuracy "
                   "is over all evaluated instances (unparseable orderings count "
                   "as wrong); kendall_tau_mean and pairwise_order_accuracy are "
                   "over parseable instances. Precision/recall/F1 are not "
                   "meaningful for a permutation task and are intentionally not "
                   "reported.",
    })
    return m


PERTURBATION_METRICS_CAVEAT = (
    "The task description assumed a single-class ground truth (all statements "
    "incorrect), under which precision would be trivially 1.0 and F1 degenerate, "
    "leaving recall (= detection rate) as the meaningful metric. The ACTUAL "
    "dataset is two-class and nearly balanced (is_perturbed: 246 true / 240 "
    "false), so accuracy, precision, recall, and F1 (positive class = "
    "perturbed/incorrect statement) are all meaningful here. Recall over the "
    "perturbed class still equals the error-detection rate."
)


def compute_perturbation_metrics(records: dict, n_dataset_total: int,
                                 dataset: str, model: str) -> dict:
    m = _metrics_envelope(dataset, model, n_dataset_total, records)
    evaluated = [r for r in records.values() if r.get("status") == "ok"]
    n_eval = len(evaluated)
    parsed = [r for r in evaluated if not r.get("detection_format_failure")]

    detection: dict = {
        "n_evaluated": n_eval,
        "n_detection_format_failures": n_eval - len(parsed),
        "class_balance_evaluated": {
            "perturbed": sum(1 for r in evaluated if r["is_perturbed"]),
            "accurate": sum(1 for r in evaluated if not r["is_perturbed"]),
        },
        "positive_class": "incorrect/perturbed statement",
    }
    if evaluated:
        detection["accuracy"] = round(
            sum(1 for r in evaluated if r.get("detection_correct")) / n_eval, 4)
    if parsed:
        y_true = [1 if r["is_perturbed"] else 0 for r in parsed]
        y_pred = [1 if r["predicted_perturbed"] else 0 for r in parsed]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0,
        )
        detection.update({
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "confusion": {
                "tp": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1),
                "fp": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),
                "tn": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0),
                "fn": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),
            },
        })

    perturbed_eval = [r for r in evaluated if r["is_perturbed"]]
    tp_records = [r for r in perturbed_eval if r.get("predicted_perturbed") is True]
    expl_scored = [r for r in tp_records
                   if isinstance(r.get("explanation_final_score"), (int, float))]
    n_expl_judge_error = sum(1 for r in tp_records
                             if r.get("explanation_judge_error")
                             and r.get("explanation_final_score") is None)
    expl_score_sum = sum(r["explanation_final_score"] for r in expl_scored)
    n_localized = sum(1 for r in expl_scored if r.get("perturbation_localized"))
    explanation = {
        "n_perturbed_evaluated": len(perturbed_eval),
        "n_detected_perturbed": len(tp_records),
        "n_explanations_scored": len(expl_scored),
        "n_explanation_judge_error": n_expl_judge_error,
        "n_false_positive_explanations_autofailed":
            sum(1 for r in evaluated if r.get("explanation_autofailed_fp")),
        "mean_explanation_final_score_over_detected":
            round(expl_score_sum / len(expl_scored), 4) if expl_scored else None,
        "mean_explanation_final_score_over_all_perturbed":
            round(expl_score_sum / len(perturbed_eval), 4)
            if perturbed_eval else None,
        "localization_rate":
            round(n_localized / len(expl_scored), 4) if expl_scored else None,
        "correction_given_rate":
            round(sum(1 for r in expl_scored if r.get("correction_given"))
                  / len(expl_scored), 4) if expl_scored else None,
        # Logged only — never folded into any score (rubric v2).
        "correction_correct_rate":
            round(sum(1 for r in expl_scored if r.get("correction_correct"))
                  / len(expl_scored), 4) if expl_scored else None,
        "n_uncertain": sum(1 for r in expl_scored
                           if r.get("explanation_uncertain")),
        "n_confidence_scored": _confidence_stats(
            expl_scored, "explanation_judge_confidence")[1],
        "explanation_judge_confidence_mean": _confidence_stats(
            expl_scored, "explanation_judge_confidence")[0],
        "note": "Rubric v2: explanation_final_score = base rubric gated on "
                "perturbation_localized (unlocalized explanations score 0.0). "
                "mean_explanation_final_score_over_all_perturbed counts missed "
                "detections and judge errors as 0.0.",
    }

    m.update({
        "detection": detection,
        "explanation": explanation,
        "rubric_version": rubric_scoring.RUBRIC_VERSION,
        "judge_config": dict(JUDGE_RUN_CONFIG),
        "run_counters": {
            "items_scored": len(expl_scored),
            "judge_error": n_expl_judge_error,
            "uncertain": explanation["n_uncertain"],
            "calendar_mismatch": sum(1 for r in evaluated
                                     if r.get("calendar_mismatch")),
            "anchor_missing": 0,
            "offset_granularity_unparsed": 0,
        },
        "metrics_caveat": PERTURBATION_METRICS_CAVEAT,
    })
    return m


DATASET_EVALUATORS = {
    "context_misalignment": eval_llm_judged,
    "event_sorting": eval_event_sorting,
    "open_ended": eval_llm_judged,
    "perturbation_detection": eval_perturbation,
    "relative_time_qa": eval_llm_judged,
    "scenario_short_answer": eval_llm_judged,
    "temporal_mcq": eval_temporal_mcq,
    "temporal_scenario_short_answer": eval_llm_judged,
}

DATASET_METRICS = {
    "context_misalignment": compute_rubric_metrics,
    "event_sorting": compute_event_sorting_metrics,
    "open_ended": compute_rubric_metrics,
    "perturbation_detection": compute_perturbation_metrics,
    "relative_time_qa": compute_rubric_metrics,
    "scenario_short_answer": compute_rubric_metrics,
    "temporal_mcq": compute_mcq_metrics,
    "temporal_scenario_short_answer": compute_rubric_metrics,
}

# LLM-judged datasets whose per-item records carry rubric-v2 subscores.
JUDGED_DATASETS = tuple(sorted(
    name for name, fn in DATASET_METRICS.items() if fn is compute_rubric_metrics))


# ════════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════════

class TqdmLoggingHandler(logging.Handler):
    """Console handler that writes through tqdm.write so log lines print
    above active progress bars instead of breaking them."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = TqdmLoggingHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(results_dir / "param_check.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    # Third-party per-request chatter (one line per HTTP call) drowns out the
    # per-instance progress lines — keep only their warnings.
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_dataset(name: str) -> list[dict]:
    path = QA_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["qa_pairs"]


def write_summary(results_dir: Path, model: str, caller: GeminiCaller,
                  key_manager: GeminiApiKeyManager,
                  judge_callers: list | None = None) -> None:
    datasets = {}
    for name in DATASET_EVALUATORS:
        metrics_path = results_dir / name / "metrics.json"
        if metrics_path.exists():
            try:
                datasets[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not include %s in summary: %s", metrics_path, e)
    # Rubric-v2 rollup: per-dataset mean scores plus the run counters the
    # rubric requires (items scored, judge_error, uncertain, calendar_mismatch,
    # anchor_missing, offset_granularity_unparsed) summed across datasets.
    score_by_dataset: dict[str, dict] = {}
    counters = {"items_scored": 0, "judge_error": 0, "uncertain": 0,
                "calendar_mismatch": 0, "anchor_missing": 0,
                "offset_granularity_unparsed": 0}
    judge_config = None
    # Judge-confidence rollup: per-dataset mean + overall mean weighted by the
    # number of confidence-carrying items (logged-only, never scored).
    conf_by_dataset: dict[str, dict] = {}
    conf_weighted_sum, conf_weighted_n = 0.0, 0
    for name, m in datasets.items():
        if m.get("mean_final_score") is not None:
            score_by_dataset[name] = {"mean_final_score": m["mean_final_score"],
                                      "n_scored": m.get("n_scored")}
        elif isinstance(m.get("explanation"), dict):
            score_by_dataset[name] = {
                "mean_explanation_final_score_over_detected":
                    m["explanation"].get(
                        "mean_explanation_final_score_over_detected"),
                "n_scored": m["explanation"].get("n_explanations_scored"),
            }
        for key in counters:
            counters[key] += (m.get("run_counters") or {}).get(key, 0)
        judge_config = m.get("judge_config") or judge_config
        conf_mean, conf_n = m.get("judge_confidence_mean"), m.get("n_confidence_scored", 0)
        if conf_mean is None and isinstance(m.get("explanation"), dict):
            conf_mean = m["explanation"].get("explanation_judge_confidence_mean")
            conf_n = m["explanation"].get("n_confidence_scored", 0)
        if conf_mean is not None and conf_n:
            conf_by_dataset[name] = {"mean": conf_mean, "n": conf_n}
            conf_weighted_sum += conf_mean * conf_n
            conf_weighted_n += conf_n
    rubric_rollup = {
        "by_dataset": score_by_dataset,
        "run_counters": counters,
        "judge_config": judge_config or dict(JUDGE_RUN_CONFIG),
        "judge_confidence": {
            "overall_mean": (round(conf_weighted_sum / conf_weighted_n, 2)
                             if conf_weighted_n else None),
            "by_dataset": conf_by_dataset,
            "note": "Judge's self-reported confidence (0-100) in its atomic "
                    "judgments, averaged per dataset and overall (weighted by "
                    "instance count). Logged-only — it never affects any "
                    "score. For perturbation_detection this covers "
                    "explanation judgments; deterministic datasets have no "
                    "judge.",
        },
        "note": "Rubric v2: per-item scores are computed deterministically "
                "from atomic judge booleans; judge_error items are excluded "
                "from means and counted here. Deterministic datasets "
                "(temporal_mcq, event_sorting) have no judge and no entry in "
                "by_dataset.",
    }
    summary = {
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": datasets,
        "rubric_v2": rubric_rollup,
        "run_stats": {
            "api_calls_this_run": caller.api_calls,
            "api_error_events_this_run": caller.api_failures,
            "key_rotations_this_run": caller.rotation_count,
            "key_usage": key_manager.get_usage_stats(),
            "judge_api_calls": {
                c.model: {"api_calls": c.api_calls,
                          "api_error_events": c.api_failures,
                          "dead": getattr(c, "dead", False)}
                for c in (judge_callers or [])},
        },
    }
    write_json_atomic(results_dir / "summary.json", summary)
    logger.info("Saved combined summary: %s", results_dir / "summary.json")


def write_calibration_dump(records: dict, path: Path, n: int,
                           dataset: str) -> None:
    """Stratified sample of judged items -> CSV for human labeling, so
    judge-human agreement (Cohen's kappa) can be measured before trusting
    aggregate numbers. Strata: final-score bucket {0, partial, 1} x uncertain
    flag, equal allocation, seeded shuffle (random.Random(0)) so dumps are
    reproducible across runs."""
    expl = dataset == "perturbation_detection"
    score_key = "explanation_final_score" if expl else "final_score"
    atoms_key = "explanation_judge_atoms" if expl else "judge_atoms"
    answer_key = "model_explanation" if expl else "model_answer"
    uncertain_key = "explanation_uncertain" if expl else "uncertain"

    rows = [r for r in records.values()
            if isinstance(r.get(score_key), (int, float))
            and r.get(atoms_key) is not None]
    if not rows or n <= 0:
        logger.info("[%s] calibration dump skipped (no judged records).", dataset)
        return

    def bucket(r: dict) -> tuple:
        s = r[score_key]
        b = "zero" if s == 0.0 else ("full" if s == 1.0 else "partial")
        return b, bool(r.get(uncertain_key))

    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        strata.setdefault(bucket(r), []).append(r)
    rng = random.Random(0)
    for group in strata.values():
        group.sort(key=lambda r: r["instance_id"])
        rng.shuffle(group)

    # Equal allocation with round-robin over strata until n items are drawn.
    sampled: list[dict] = []
    keys = sorted(strata, key=str)
    idx = {k: 0 for k in keys}
    while len(sampled) < min(n, len(rows)):
        progressed = False
        for k in keys:
            if len(sampled) >= min(n, len(rows)):
                break
            if idx[k] < len(strata[k]):
                sampled.append(strata[k][idx[k]])
                idx[k] += 1
                progressed = True
        if not progressed:
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "question", "ground_truth",
                         "model_answer", "gt_dates", "judge_atoms",
                         "n_matched", "date_score", "meaning_correct",
                         "uncertain", "judge_confidence", "final_score",
                         "rubric_version", "human_label"])
        for r in sampled:
            writer.writerow([
                r.get("instance_id"), r.get("question"),
                r.get("ground_truth_answer"), r.get(answer_key),
                json.dumps(r.get("gt_dates"), ensure_ascii=False),
                json.dumps(r.get(atoms_key), ensure_ascii=False),
                r.get("explanation_n_matched" if expl else "n_matched"),
                r.get("explanation_date_score" if expl else "date_score"),
                r.get("explanation_meaning_correct" if expl
                      else "meaning_correct"),
                r.get(uncertain_key),
                r.get("explanation_judge_confidence" if expl
                      else "judge_confidence"),
                r.get(score_key),
                r.get("rubric_version"), "",
            ])
    logger.info("[%s] calibration dump: %d item(s) -> %s",
                dataset, len(sampled), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parametric knowledge evaluation of Gemini on the ICSETEP "
                    "concept-QA benchmark.")
    parser.add_argument("--datasets", nargs="+",
                        choices=sorted(DATASET_EVALUATORS), metavar="NAME",
                        default=sorted(DATASET_EVALUATORS),
                        help="Datasets to evaluate (default: all 8). "
                             f"Choices: {', '.join(sorted(DATASET_EVALUATORS))}")
    parser.add_argument("--openrouter-key-file", default=OPENROUTER_KEY_FILE,
                        help="File holding the OpenRouter API key for the "
                             "Claude/GPT council judges (default: "
                             f"{OPENROUTER_KEY_FILE}; the OPENROUTER_API_KEY "
                             "env var takes precedence).")
    parser.add_argument("--calibration-dump", type=int, default=0, metavar="N",
                        help="Write N stratified judged items per dataset to "
                             "<dataset>/calibration_dump.csv for human "
                             "labeling (default: 0 = off).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N instances per dataset "
                             "(smoke testing).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip instances already completed in results.json "
                             "(default: on; api_failed instances are retried).")
    parser.add_argument("--keys-file", default=API_KEYS_FILE_PATH,
                        help=f"API keys file (default: {API_KEYS_FILE_PATH}).")
    parser.add_argument("--model", default=GEMINI_MODEL,
                        help=f"Gemini model for answering (default: "
                             f"{GEMINI_MODEL}). The judge council is fixed: "
                             f"{', '.join(JUDGE_COUNCIL_MODELS)}.")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help=f"Output directory (default: {RESULTS_DIR}).")
    parser.add_argument("--rate-limit-delay", type=float, default=RATE_LIMIT_DELAY,
                        help=f"Seconds to sleep before every API call "
                             f"(default: {RATE_LIMIT_DELAY}).")
    parser.add_argument("--calls-per-day", type=int, default=CALLS_PER_DAY,
                        help=f"Max API calls per key per day (default: {CALLS_PER_DAY}).")
    return parser.parse_args()


def main() -> None:
    global JUDGE_COUNCIL, JUDGE_CALLERS
    args = parse_args()
    results_dir = Path(args.results_dir)
    setup_logging(results_dir)

    logger.info("=" * 70)
    logger.info("Parametric knowledge check | model=%s | datasets=%s | limit=%s | "
                "resume=%s | judge_council=%s",
                args.model, ",".join(args.datasets), args.limit, args.resume,
                "+".join(JUDGE_COUNCIL_MODELS))
    logger.info("=" * 70)

    key_manager = GeminiApiKeyManager(
        keys_file=args.keys_file,
        calls_per_day=args.calls_per_day,
        rate_limit_delay=args.rate_limit_delay,
    )
    caller = GeminiCaller(key_manager, model=args.model)
    # Fail fast if the OpenRouter key is missing, before any answering call.
    openrouter_key = load_openrouter_key(args.openrouter_key_file)
    JUDGE_COUNCIL, JUDGE_CALLERS = build_judge_council(key_manager, openrouter_key)

    try:
        for dataset in args.datasets:
            qa_pairs = load_dataset(dataset)
            items = list(enumerate(qa_pairs))
            if args.limit is not None:
                items = items[: args.limit]

            store = ResultStore(results_dir, dataset, args.model)
            if args.resume:
                loaded = store.load()
                if loaded:
                    logger.info("[%s] resume: %d existing record(s), %d completed",
                                dataset, loaded, len(store.completed_ids()))

            pending = [i for i, _ in items if i not in store.completed_ids()]
            logger.info("=== %s: evaluating %d of %d instances (%d already done) ===",
                        dataset, len(pending), len(items), len(items) - len(pending))

            DATASET_EVALUATORS[dataset](dataset, items, caller, store)

            metrics = DATASET_METRICS[dataset](
                store.records, len(qa_pairs), dataset, args.model)
            write_json_atomic(results_dir / dataset / "metrics.json", metrics)
            logger.info("[%s] metrics saved: %s", dataset,
                        results_dir / dataset / "metrics.json")
            if args.calibration_dump and dataset in (*JUDGED_DATASETS,
                                                     "perturbation_detection"):
                write_calibration_dump(
                    store.records,
                    results_dir / dataset / "calibration_dump.csv",
                    args.calibration_dump, dataset)
    except KeyboardInterrupt:
        logger.warning("Interrupted — progress is checkpointed; re-run with "
                       "--resume (default) to continue.")
    finally:
        key_manager.save_key_status_report(results_dir / "api_key_status.txt")
        write_summary(results_dir, args.model, caller, key_manager,
                      judge_callers=JUDGE_CALLERS)
        logger.info("Run stats: %d API calls, %d error events, %d key "
                    "rotations. Judge calls: %s.",
                    caller.api_calls, caller.api_failures,
                    caller.rotation_count,
                    ", ".join(f"{c.model}={c.api_calls}"
                              for c in JUDGE_CALLERS) or "none")


if __name__ == "__main__":
    main()
