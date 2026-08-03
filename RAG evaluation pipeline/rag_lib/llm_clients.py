"""llm_clients.py — generation callers for the RAG evaluation.

Two callers, both honoring the single contract the evaluators/metrics depend on:

    caller.call(prompt, response_schema=None, max_output_tokens=None) -> str | None
        returns stripped text ("" if the model returned nothing), or None when
        all keys are exhausted / the call failed (recorded as api_failed and
        retried on --resume).
    caller.api_calls / api_failures / rotation_count      run counters
    caller.key_manager.get_usage_stats()                  key-pool stats

- OpenRouterCaller  — the models UNDER TEST (and the council's two OpenRouter
  judge seats). OpenAI-compatible Chat Completions API. Structured output is left
  to the prompt's own JSON instructions + the lenient parsers in eval_lib (models
  on OpenRouter vary in response_format support), unless USE_JSON_RESPONSE_FORMAT
  is enabled.
- GeminiCaller      — the council's Gemini judge seat (and Gemini answering when
  USE_GEMINI_API). Native JSON schema via google.genai.

Both pull keys via KeyManager from plaintext (no vault): OpenRouter keys from the
`openrouter` section (env OPENROUTER_API_KEYS or openrouter_api.txt), Gemini keys
from the `gemini` section (env GEMINI_API_KEYS or gemini_api.txt). Pass a shared
`key_manager` to reuse one key pool across callers/judge seats.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from key_manager import AllKeysExhausted, KeyManager, TooLongError  # noqa: E402

logger = logging.getLogger("llm_clients")

# When True, ask OpenRouter for a JSON object (models that ignore/reject it may
# error → that instance becomes api_failed). Default off for max compatibility:
# the MCQ/verdict/event prompts already instruct exact JSON and eval_lib's
# parsers are lenient.
USE_JSON_RESPONSE_FORMAT = False


# ════════════════════════════════════════════════════════════════════════════
# OpenRouter (models under test)
# ════════════════════════════════════════════════════════════════════════════

class OpenRouterKeyManager(KeyManager):
    """KeyManager whose rotated client is an OpenAI SDK client pointed at
    OpenRouter. Inherits the generic rotate/rest/cooldown machinery; the base
    classify_error already buckets 429→quota and 5xx→server off the SDK
    exception's status_code."""

    def _client(self, i: int):
        client = self._clients.get(i)
        if client is None:
            from openai import OpenAI  # lazy import
            client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=self.keys[i])
            self._clients[i] = client
        return client


class OpenRouterCaller:
    """Prompt-in / text-out generation caller for one OpenRouter model."""

    def __init__(self, model: str, key_manager: OpenRouterKeyManager,
                 temperature: float = 0.0, max_output_tokens: int | None = None):
        self.model = model
        self.key_manager = key_manager
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_calls = 0
        self.api_failures = 0

    @property
    def rotation_count(self) -> int:
        return self.key_manager.rotations

    def call(self, prompt: str, response_schema: dict | None = None,
             max_output_tokens: int | None = None) -> str | None:
        self.api_calls += 1
        mot = max_output_tokens or self.max_output_tokens

        def _do(client):
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if mot:
                kwargs["max_tokens"] = mot
            if response_schema is not None and USE_JSON_RESPONSE_FORMAT:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content if resp.choices else ""
            return (content or "").strip()

        try:
            return self.key_manager.call(_do)
        except (AllKeysExhausted, TooLongError):
            self.api_failures += 1
            return None
        except Exception as exc:  # noqa: BLE001 — one bad instance must not kill the run
            self.api_failures += 1
            logger.warning("[%s] OpenRouter call failed: %s", self.model, exc)
            return None


# ════════════════════════════════════════════════════════════════════════════
# Gemini (the constant judge)
# ════════════════════════════════════════════════════════════════════════════

class GeminiCaller:
    """Judge caller: paid Gemini key from the vault, native JSON-schema output.
    Same call() contract as OpenRouterCaller so make_judge_call_fn / judge_atoms
    work unchanged."""

    def __init__(self, model: str = config.JUDGE_MODEL, temperature: float = 0.0,
                 max_output_tokens: int | None = None, key_manager=None):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # Share one Gemini key pool across judge seats by passing key_manager;
        # otherwise load the section's plaintext keys.
        self.key_manager = key_manager or KeyManager.load(
            section=config.SECTION_GEMINI)
        self.api_calls = 0
        self.api_failures = 0

    @property
    def rotation_count(self) -> int:
        return self.key_manager.rotations

    def call(self, prompt: str, response_schema: dict | None = None,
             max_output_tokens: int | None = None) -> str | None:
        from google.genai import types  # lazy import
        self.api_calls += 1
        mot = max_output_tokens or self.max_output_tokens

        def _do(client):
            cfg = {"temperature": self.temperature}
            if mot:
                cfg["max_output_tokens"] = mot
            if response_schema is not None:
                cfg["response_mime_type"] = "application/json"
                cfg["response_schema"] = response_schema
            resp = client.models.generate_content(
                model=self.model, contents=prompt,
                config=types.GenerateContentConfig(**cfg))
            return (resp.text or "").strip()

        try:
            return self.key_manager.call(_do)
        except (AllKeysExhausted, TooLongError):
            self.api_failures += 1
            return None
        except Exception as exc:  # noqa: BLE001
            self.api_failures += 1
            logger.warning("[judge %s] Gemini call failed: %s", self.model, exc)
            return None


# ════════════════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    km = OpenRouterKeyManager.load(section=config.SECTION_OPENROUTER)
    caller = OpenRouterCaller("openai/gpt-4o-mini", km, max_output_tokens=64)
    reply = caller.call("Reply with exactly the word: pong")
    print(f"OpenRouter reply: {reply!r}")
    print(f"api_calls={caller.api_calls} failures={caller.api_failures} "
          f"rotations={caller.rotation_count}")
