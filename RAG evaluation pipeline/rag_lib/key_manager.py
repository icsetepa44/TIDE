"""API-key loading (from .env / environment) + rate-limit-aware key rotation.

Two responsibilities:
  1. load_section_keys: read a section's API keys from environment variables. A
     .env file at config.DOTENV_FILE is loaded into the environment at import
     (python-dotenv; real environment variables take precedence). No key files.
  2. KeyManager: hold the keys in memory, hand a live google-genai (or OpenRouter)
     client to a caller, and rotate to the next key when one hits a quota / 429 error
     (with a cooldown so exhausted keys are retried later instead of hammered).
"""
from __future__ import annotations

import os
import re
import sys as _sys
from pathlib import Path as _Path

# Make sibling rag_lib modules (config) and parent OpenRouter/ modules importable
# no matter how this module is loaded.
for _p in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import time

import config
from dotenv import load_dotenv

# Load OpenRouter/.env into the environment once, at import. override=False so a
# real environment variable always wins over a value in the .env file.
load_dotenv(config.DOTENV_FILE, override=False)


# --------------------------------------------------------------------------- #
# Key loading (from environment; .env already loaded above)                   #
# --------------------------------------------------------------------------- #
def _split_keys(blob: str) -> list[str]:
    """Split a comma/newline/whitespace-separated key blob into a clean list."""
    parts = re.split(r"[,\s]+", blob.strip())
    return [p for p in (s.strip() for s in parts) if p]


def load_section_keys(section: str) -> list[str]:
    """Load one section's API keys from the environment (populated from .env).

    Per section, in order: the multi-key var (config.KEY_ENV_MULTI, comma/newline
    separated), then numbered vars NAME_1, NAME_2, ..., then the single-key var
    (config.KEY_ENV_SINGLE). Multiple keys enable rotation on rate-limit/quota.
    """
    multi = os.environ.get(config.KEY_ENV_MULTI.get(section, ""), "").strip()
    if multi:
        keys = _split_keys(multi)
        if keys:
            return keys

    single_name = config.KEY_ENV_SINGLE.get(section, "")
    if single_name:
        numbered: list[str] = []
        i = 1
        while True:
            v = os.environ.get(f"{single_name}_{i}", "").strip()
            if not v:
                break
            numbered.append(v)
            i += 1
        if numbered:
            return numbered
        v = os.environ.get(single_name, "").strip()
        if v:
            return [v]

    multi_name = config.KEY_ENV_MULTI.get(section)
    raise FileNotFoundError(
        f"No '{section}' API keys found. Copy .env.example to .env and set "
        f"{multi_name} (comma-separated for multiple keys), or export {multi_name} / "
        f"${single_name} in your environment."
    )


# --------------------------------------------------------------------------- #
# Error classification                                                        #
# --------------------------------------------------------------------------- #
class TooLongError(Exception):
    """Input exceeded the model's token limit — caller should skip and log the item."""


class AllKeysExhausted(Exception):
    """Every key is cooling down and the wait budget ran out."""


def classify_error(exc: Exception) -> str:
    """Bucket an SDK exception into: 'quota' | 'too_long' | 'server' | 'other'."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if code == 429 or "resource_exhausted" in msg or "quota" in msg or "rate limit" in msg:
        return "quota"
    if code == 400 and any(
        s in msg for s in ("token", "too long", "exceeds", "maximum", "input is too large")
    ):
        return "too_long"
    if code in (500, 503) or any(
        s in msg for s in ("internal error", "unavailable", "overloaded", "deadline")
    ):
        return "server"
    return "other"


# --------------------------------------------------------------------------- #
# KeyManager                                                                  #
# --------------------------------------------------------------------------- #
class KeyManager:
    """Round-robin over API keys, resting each key after a quota error."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("No API keys supplied to KeyManager.")
        self.keys = keys
        self.idx = 0
        self.cooldown_until = [0.0] * len(keys)   # epoch seconds a key is rested until
        self._clients: dict[int, object] = {}     # lazily-created genai clients per key
        self.last_key_index = 0
        self.rotations = 0                         # times a key was rested + rotated off

    # -- lifecycle ----------------------------------------------------------- #
    @classmethod
    def load(cls, section: str = "gemini", **_ignored) -> "KeyManager":
        """Load one section's keys from the environment (.env) into a KeyManager.

        `section` is "gemini" (embedding + the judge council's Gemini seat) or
        "openrouter" (models under test + the council's OpenRouter seats). Keys
        come from the section's environment variables — see `load_section_keys`.
        Extra kwargs from old vault call sites (`password=`, `vault_file=`) are
        accepted and ignored for backward compatibility.
        """
        return cls(load_section_keys(section))

    def _client(self, i: int):
        client = self._clients.get(i)
        if client is None:
            from google import genai  # imported lazily so crypto-only use needs no SDK
            client = genai.Client(api_key=self.keys[i])
            self._clients[i] = client
        return client

    # -- scheduling ---------------------------------------------------------- #
    def _pick_index(self, max_wait: float = 900.0) -> int:
        """Return the index of a healthy key, sleeping if all are cooling down."""
        n = len(self.keys)
        waited = 0.0
        while True:
            now = time.time()
            for step in range(n):
                i = (self.idx + step) % n
                if self.cooldown_until[i] <= now:
                    self.idx = i
                    return i
            # All keys are resting — sleep until the soonest one frees up.
            soonest = min(self.cooldown_until)
            nap = max(1.0, min(soonest - now, 30.0))
            if waited + nap > max_wait:
                raise AllKeysExhausted(
                    f"All {n} keys are rate-limited; waited {int(waited)}s. "
                    "Re-run later — progress is saved."
                )
            print(f"    [keys] all {n} keys cooling down; sleeping {int(nap)}s...")
            time.sleep(nap)
            waited += nap

    def _rest(self, i: int, seconds: float) -> None:
        self.cooldown_until[i] = time.time() + seconds
        self.idx = (i + 1) % len(self.keys)  # advance so the next call tries a fresh key
        self.rotations += 1

    # -- the one entry point callers use ------------------------------------ #
    def call(self, fn):
        """Run fn(client) with automatic key rotation.

        - quota/429  -> rest this key, rotate, retry
        - 5xx/server -> short backoff, retry same key a few times, then rotate
        - too_long   -> raise TooLongError (caller skips the item)
        - other      -> retry once on a different key, then re-raise
        """
        server_tries = 0
        other_tries = 0
        while True:
            i = self._pick_index()
            self.last_key_index = i
            try:
                return fn(self._client(i))
            except Exception as exc:  # noqa: BLE001 - classified below
                kind = classify_error(exc)
                if kind == "too_long":
                    raise TooLongError(str(exc)) from exc
                if kind == "quota":
                    self._rest(i, config.QUOTA_COOLDOWN_SECONDS)
                    continue
                if kind == "server":
                    server_tries += 1
                    if server_tries >= config.SERVER_MAX_RETRIES:
                        self._rest(i, 5)  # brief rest, move on
                        server_tries = 0
                        continue
                    time.sleep(min(2 ** server_tries, 20))
                    continue
                # unknown error: try a different key once, then give up on this item
                other_tries += 1
                if other_tries >= 2:
                    raise
                self.idx = (i + 1) % len(self.keys)

    def status_line(self) -> str:
        now = time.time()
        resting = sum(1 for t in self.cooldown_until if t > now)
        return f"key #{self.last_key_index + 1}/{len(self.keys)} ({resting} resting)"

    def get_usage_stats(self) -> dict:
        """Provider-agnostic key-pool stats for run summaries."""
        now = time.time()
        return {
            "n_keys": len(self.keys),
            "rotations": self.rotations,
            "keys_resting": sum(1 for t in self.cooldown_until if t > now),
            "last_key_index": self.last_key_index,
        }
