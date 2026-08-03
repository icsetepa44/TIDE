"""
OpenRouter chat-completions caller for the LLM-council judges.

Mirrors PK.GeminiCaller's interface:
    call(prompt, response_schema=None, max_output_tokens=None) -> str | None
None means this judge is unavailable for the call; llm_council.judge_atoms
absorbs single-judge outages via its 2-of-3 quorum.

JSON output is prompt-driven: the judge prompts already demand strict JSON and
llm_council's parsing is defensive (fence stripping, brace matching, one
repair retry), so `response_schema` is accepted for interface compatibility
but NOT translated into OpenRouter's response_format — Anthropic- and
OpenAI-hosted models have inconsistent structured-output support behind
OpenRouter, and a silent provider-side rejection would be harder to debug
than a parse retry.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger("openrouter")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_FILE = "openrouter_api.txt"
MAX_ATTEMPTS = 4
RATE_LIMIT_DELAY = 1.0  # OpenRouter is paid; no free-tier pacing needed
BACKOFF_CAP = 30.0


def load_openrouter_key(keys_file: str = OPENROUTER_KEY_FILE) -> str:
    """OPENROUTER_API_KEY env var, else the first non-blank line of keys_file."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    p = Path(keys_file)
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return line.strip()
    raise ValueError(
        f"No OpenRouter API key found: set the OPENROUTER_API_KEY environment "
        f"variable or put the key in {keys_file}"
    )


class OpenRouterCaller:
    """One OpenRouter judge model, temperature=0, with retry/backoff.

    Error policy: 429 -> honor Retry-After (min. current backoff) and retry;
    timeouts / 5xx / malformed 200 bodies -> exponential backoff retry;
    401/402/403 (bad key, out of credit) -> the caller goes `dead` and every
    future call returns None instantly, so a revoked key costs nothing per
    instance and the council degrades to the surviving judges; other 4xx
    (bad request / unknown model id) -> None for this call.
    """

    def __init__(self, model: str, api_key: str, temperature: float = 0.0,
                 max_output_tokens: int | None = None,
                 rate_limit_delay: float = RATE_LIMIT_DELAY):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.rate_limit_delay = rate_limit_delay
        self.api_calls = 0
        self.api_failures = 0
        self.dead = False  # permanent auth/credit failure: fail fast forever
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=httpx.Timeout(180.0, connect=15.0))

    def call(self, prompt: str, response_schema: dict | None = None,
             max_output_tokens: int | None = None) -> str | None:
        if self.dead:
            return None
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        mot = max_output_tokens or self.max_output_tokens
        if mot:
            body["max_tokens"] = mot
        # response_schema intentionally unused (see module docstring).

        backoff = 2.0
        for attempt in range(MAX_ATTEMPTS):
            time.sleep(self.rate_limit_delay)
            self.api_calls += 1
            try:
                resp = self._client.post(OPENROUTER_ENDPOINT,
                                         headers=self._headers, json=body)
            except httpx.HTTPError as e:  # timeout / connection: transient
                self.api_failures += 1
                logger.warning("[%s] transient HTTP error (attempt %d/%d): %s",
                               self.model, attempt + 1, MAX_ATTEMPTS, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_CAP)
                continue

            if resp.status_code == 429:
                self.api_failures += 1
                try:
                    retry_after = float(resp.headers.get("Retry-After") or 0)
                except ValueError:
                    retry_after = 0.0
                wait = max(retry_after, backoff)
                logger.warning("[%s] 429 rate-limited; sleeping %.0fs",
                               self.model, wait)
                time.sleep(wait)
                backoff = min(backoff * 2, BACKOFF_CAP)
                continue
            if resp.status_code in (401, 402, 403):
                self.api_failures += 1
                self.dead = True
                logger.error(
                    "[%s] PERMANENT OpenRouter failure (HTTP %d): %.300s — "
                    "this judge is disabled for the rest of the run",
                    self.model, resp.status_code, resp.text)
                return None
            if 400 <= resp.status_code < 500:
                self.api_failures += 1
                logger.error("[%s] OpenRouter HTTP %d: %.300s",
                             self.model, resp.status_code, resp.text)
                return None
            if resp.status_code >= 500:
                self.api_failures += 1
                logger.warning("[%s] OpenRouter HTTP %d (attempt %d/%d)",
                               self.model, resp.status_code,
                               attempt + 1, MAX_ATTEMPTS)
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_CAP)
                continue

            try:  # 200: the body may still carry an error object
                data = resp.json()
                if "error" in data:
                    raise ValueError(data["error"])
                text = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as e:
                self.api_failures += 1
                logger.warning("[%s] malformed 200 body (attempt %d/%d): %s",
                               self.model, attempt + 1, MAX_ATTEMPTS, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_CAP)
                continue
            # "" propagates upstream and triggers the judge's repair retry,
            # matching GeminiCaller's empty-response behavior.
            return text.strip() if text else ""

        logger.error("[%s] OpenRouter call failed after %d attempts",
                     self.model, MAX_ATTEMPTS)
        return None
