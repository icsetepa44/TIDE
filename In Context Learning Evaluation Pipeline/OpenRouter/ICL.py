"""
ICL.py — GROUNDED (document-as-context) evaluation of
DeepSeek V4 Flash (via OpenRouter) on the ICSETEP concept-QA benchmark.

Grounded twin of PK.py (which measures PARAMETRIC knowledge):
the evaluation logic — prompts, rubric-v2 council judging (llm_council.
judge_atoms: 3 heterogeneous judges with per-boolean majority voting +
deterministic scoring in rubric_scoring.py), metrics, temperature=0, API
budget, checkpointing — is identical; the ONLY structural change is that
the true source documents (resolved from each record's `clause_text_ids` via
document_resolver.py) are prepended to every answering prompt, and the
preamble's final sentence is swapped for a grounded instruction. Judge calls
never see the documents, so grading is identical to the parametric run.

Model split (this variant): ANSWERING calls go to `deepseek/deepseek-v4-flash`
through OpenRouter's OpenAI-compatible chat-completions REST API
(OpenRouterCaller below; single paid key read from openrouter_api.txt).
JUDGING is the LLM council (pk.build_judge_council): gemini-3.5-flash via
pk.GeminiCaller on the key(s) in gemini_paid_api.txt, plus
anthropic/claude-opus-4.6 and openai/gpt-5.1 via openrouter_caller.
OpenRouterCaller reusing the same OpenRouter key as answering.

PK.py is imported and shared (rubric v2 lives there and in
rubric_scoring.py/llm_council.py). The four evaluator loop bodies are copied
from it (context varies per instance, so injection must happen at in-loop
prompt construction); every copied function cites its source lines and fences
its only changes in `── GROUNDED-DIFF ──` blocks.

`context_misalignment` note: the grounded prompt contains BOTH the misaligned
context and the authoritative document(s). That condition is labelled
`context_conflict_resolution` in the metrics — it measures whether the model
trusts the official document over the misaligned claim, NOT parametric
misalignment robustness.

Outputs (never touches param_check_results/):
    context_check_results/<dataset>/results.json   per-instance records
        (parametric fields + resolved_documents, unresolved_ids,
        context_status, context_char_count, context_token_estimate,
        clause_refs, documents_sent, dropped_documents, context_truncated)
    context_check_results/<dataset>/metrics.json   parametric metrics + context_stats
    context_check_results/summary.json             combined + resolution stats
    context_check_results/comparison.json          parametric-vs-grounded (only
        if the parametric results dir exists)
    context_check_results/resolution_report.txt    unresolved-ID log
    context_check_results/context_check.log        run log

Usage:
    python ICL.py                     # all 8 datasets
    python ICL.py --datasets relative_time_qa --limit 2
    python ICL.py --calibration-dump 25
    python ICL.py --no-resume

Answering-side rate limits: the OpenRouter key is paid, so there is no key
rotation — OpenRouterCaller retries 429/5xx/timeouts with exponential backoff
(honouring Retry-After) and gives up on an instance after the retry budget
(status=api_failed, retried on --resume). `--openrouter-delay` paces the
answering calls; `--rate-limit-delay` paces only the Gemini judge calls.

Context cap (--max-context-chars, default 1,200,000; 0 = unlimited):
DeepSeek V4 Flash's context window is 1M tokens; an over-long document block
would be rejected on every retry, so the cap is kept as a safety margin.
Bengali runs ~1.5-2 chars/token, so 1.2M chars ≈ 600-800k tokens; the
default keeps every instance inside the window while leaving room for the
question and answer — in practice no instance should be truncated.
The cap drops whole documents (never mid-document, keeping first-occurrence
order) until the block fits; truncated instances are flagged with
context_truncated=True and dropped_documents so they can be sliced out of
any analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

import llm_council
import PK as pk
import rubric_scoring
from document_resolver import DocumentResolver, ResolutionResult, ResolvedDocument

logger = logging.getLogger("context_check")

RESULTS_DIR = "context_check_results"
SCRIPT_DIR = Path(__file__).resolve().parent
# Largest document block sent per request. DeepSeek V4 Flash's context window is
# 1M tokens and an oversized request fails on every retry; Bengali runs
# ~1.5-2 chars/token, so 1.2M chars ≈ 600-800k tokens — inside the window
# with room left for the question and the answer.
MAX_CONTEXT_CHARS_DEFAULT = 1_200_000

# ── Answering side: DeepSeek V4 Flash via OpenRouter (single paid key) ──
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_DEFAULT = "deepseek/deepseek-v4-flash"
OPENROUTER_KEY_FILE_DEFAULT = "openrouter_api.txt"
OPENROUTER_RATE_LIMIT_DELAY = 2.5
OPENROUTER_TIMEOUT_S = 180.0
OPENROUTER_MAX_ATTEMPTS = 6

# ── Judge side: Gemini on the paid key (pk.GeminiCaller unchanged) ──
JUDGE_KEYS_FILE_DEFAULT = "gemini_paid_api.txt"
JUDGE_RATE_LIMIT_DELAY = 2.0
# The 500/day default in PK.py is a free-tier budget; the single
# paid judge key must never be marked "exhausted" mid-run.
JUDGE_CALLS_PER_DAY = 1_000_000


# ════════════════════════════════════════════════════════════════════════════
# OpenRouter answering caller
# ════════════════════════════════════════════════════════════════════════════

def load_openrouter_key(path: str) -> str:
    """First non-empty line of `path`, falling back to $OPENROUTER_API_KEY."""
    p = Path(path)
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                return line
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    raise SystemExit(f"No OpenRouter API key: '{path}' is missing or empty "
                     "and OPENROUTER_API_KEY is not set.")


def to_openai_schema(schema: dict) -> dict:
    """Gemini-style plain-dict schema -> OpenAI strict-mode JSON schema:
    drop `propertyOrdering` (a Gemini extension) and pin every object node
    with additionalProperties=false (required by strict structured output)."""
    def convert(node):
        if isinstance(node, dict):
            out = {k: convert(v) for k, v in node.items()
                   if k != "propertyOrdering"}
            if out.get("type") == "object":
                out.setdefault("additionalProperties", False)
            return out
        if isinstance(node, list):
            return [convert(v) for v in node]
        return node
    return convert(schema)


class OpenRouterCaller:
    """Answering-side twin of pk.GeminiCaller with the same call() contract —
    stripped text, "" on an empty answer, None on unrecoverable failure (the
    record becomes status=api_failed and is retried on --resume) — backed by
    OpenRouter's OpenAI-compatible chat-completions REST API. Single paid
    key, so no rotation: rotation_count stays 0 (pk.write_summary reads the
    same api_calls / api_failures / rotation_count counters)."""

    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL_DEFAULT,
                 temperature: float = 0.0,
                 rate_limit_delay: float = OPENROUTER_RATE_LIMIT_DELAY,
                 timeout: float = OPENROUTER_TIMEOUT_S):
        self.model = model
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.api_calls = 0
        self.api_failures = 0
        self.rotation_count = 0
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    @staticmethod
    def _backoff(resp: requests.Response | None, attempt: int) -> float:
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), 120.0)
                except ValueError:
                    pass
        return min(2.0 ** attempt, 30.0)

    def call(self, prompt: str, response_schema: dict | None = None,
             max_output_tokens: int | None = None) -> str | None:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True,
                                "schema": to_openai_schema(response_schema)},
            }

        for attempt in range(1, OPENROUTER_MAX_ATTEMPTS + 1):
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            self.api_calls += 1
            try:
                resp = self._session.post(OPENROUTER_URL, json=payload,
                                          timeout=self.timeout)
            except requests.RequestException as e:
                self.api_failures += 1
                logger.warning("OpenRouter request error (attempt %d/%d): %s",
                               attempt, OPENROUTER_MAX_ATTEMPTS, e)
                time.sleep(self._backoff(None, attempt))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                self.api_failures += 1
                wait = self._backoff(resp, attempt)
                logger.warning("OpenRouter HTTP %d (attempt %d/%d) — retrying "
                               "in %.1fs: %.200s", resp.status_code, attempt,
                               OPENROUTER_MAX_ATTEMPTS, wait, resp.text)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                # Non-retryable 4xx (bad request, context overflow, auth) —
                # retrying cannot fix it.
                self.api_failures += 1
                logger.error("OpenRouter HTTP %d — not retryable: %.500s",
                             resp.status_code, resp.text)
                return None

            try:
                data = resp.json()
            except ValueError:
                self.api_failures += 1
                logger.warning("OpenRouter returned non-JSON 200 body "
                               "(attempt %d/%d): %.200s", attempt,
                               OPENROUTER_MAX_ATTEMPTS, resp.text)
                time.sleep(self._backoff(None, attempt))
                continue
            if "error" in data:
                # OpenRouter occasionally reports errors inside a 200 body.
                self.api_failures += 1
                err = data["error"]
                code = err.get("code") if isinstance(err, dict) else None
                logger.warning("OpenRouter error in 200 body (attempt %d/%d): "
                               "%s", attempt, OPENROUTER_MAX_ATTEMPTS, err)
                if code in (400, 401, 402, 403, 404):
                    return None
                time.sleep(self._backoff(resp, attempt))
                continue
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                self.api_failures += 1
                logger.warning("OpenRouter response missing content "
                               "(attempt %d/%d): %.300s", attempt,
                               OPENROUTER_MAX_ATTEMPTS, json.dumps(data)[:300])
                time.sleep(self._backoff(None, attempt))
                continue
            return text.strip() if text else ""

        logger.error("OpenRouter call failed after %d attempts — giving up "
                     "on this instance.", OPENROUTER_MAX_ATTEMPTS)
        return None

# ════════════════════════════════════════════════════════════════════════════
# Grounded preamble + context block
# ════════════════════════════════════════════════════════════════════════════

# Derived from the parametric preamble, not retyped: sentence 1 (domain
# framing) stays byte-identical; only the closed-book instruction is replaced.
_PARAMETRIC_FINAL_SENTENCE = ("Answer from your own knowledge; no reference "
                              "documents are provided.")
_GROUNDED_FINAL_SENTENCE = ("Answer using the official source document(s) "
                            "provided above; they are authoritative.")
GROUNDED_PREAMBLE = pk.PARAMETRIC_PREAMBLE.replace(
    _PARAMETRIC_FINAL_SENTENCE, _GROUNDED_FINAL_SENTENCE)
assert GROUNDED_PREAMBLE != pk.PARAMETRIC_PREAMBLE, \
    "PARAMETRIC_PREAMBLE changed upstream — update the grounded derivation"
assert _GROUNDED_FINAL_SENTENCE in GROUNDED_PREAMBLE

CONTEXT_HEADER = ("You are given the following official source document(s). "
                  "Use them to answer.")
EMPTY_TEXT_PLACEHOLDER = "[This document's OCR text is empty.]"

CONDITION_NOTE = (
    "Grounded run places BOTH the misaligned context and the authoritative "
    "source document(s) in the prompt; this measures whether the model "
    "resolves the conflict in favor of the official document "
    "(context_conflict_resolution), NOT parametric misalignment robustness. "
    "Do not conflate with the parametric context_misalignment result.")


def build_context_block(docs: list[ResolvedDocument]) -> str:
    """Delimited document block prepended to every answering prompt."""
    parts = [CONTEXT_HEADER, ""]
    for i, d in enumerate(docs, 1):
        parts.append(f"--- DOCUMENT {i}: {d.stem} ---")
        parts.append(d.text if d.text else EMPTY_TEXT_PLACEHOLDER)
        parts.append("")
    return "\n".join(parts) + "\n"


def build_capped_context(docs: list[ResolvedDocument],
                         max_chars: int) -> tuple[str | None, list[str]]:
    """Doc-boundary context cap: keep each document (first-occurrence order)
    whose inclusion keeps the block within max_chars; drop the rest whole —
    never cut inside a document. Returns (block or None, dropped stems).
    max_chars <= 0 disables the cap."""
    if not docs:
        return None, []
    if not max_chars or max_chars <= 0:
        return build_context_block(docs), []
    sent: list[ResolvedDocument] = []
    dropped: list[str] = []
    for d in docs:
        if len(build_context_block(sent + [d])) <= max_chars:
            sent.append(d)
        else:
            dropped.append(d.stem)
    if not sent:
        logger.warning("Context cap %d chars is smaller than the smallest "
                       "document (%s, %d chars) — sending no documents",
                       max_chars, docs[0].stem, docs[0].char_count)
        return None, dropped
    return build_context_block(sent), dropped


def to_grounded_prompt(parametric_prompt: str, context_block: str) -> str:
    """Prepend the context block and swap the preamble's final sentence.

    Prompts are always built exactly as PK.py builds them, then
    transformed here — everything below the preamble is byte-identical to the
    parametric prompt by construction.
    """
    assert parametric_prompt.startswith(pk.PARAMETRIC_PREAMBLE), \
        "answering prompt does not start with PARAMETRIC_PREAMBLE"
    return context_block + GROUNDED_PREAMBLE + parametric_prompt[len(pk.PARAMETRIC_PREAMBLE):]


def context_fields(res: ResolutionResult, context_block: str | None,
                   dropped_stems: list[str]) -> dict:
    """Extra per-record fields for the grounded run (additive — the record
    stays compatible with the parametric schema)."""
    docs = []
    for d in res.resolved_docs:
        try:
            path = str(d.path.relative_to(SCRIPT_DIR))
        except ValueError:
            path = str(d.path)
        docs.append({"stem": d.stem, "path": path, "category": d.category,
                     "char_count": d.char_count})
    n_chars = len(context_block) if context_block else 0
    return {
        "resolved_documents": docs,
        "unresolved_ids": res.unresolved_ids,
        "context_status": res.context_status,
        "context_char_count": n_chars,
        "context_token_estimate": n_chars // 4,
        "clause_refs": res.clause_refs,
        "documents_sent": len(res.resolved_docs) - len(dropped_stems),
        "dropped_documents": dropped_stems,
        "context_truncated": bool(dropped_stems),
    }


def _log_context(dataset: str, iid: int, base: dict) -> None:
    logger.info("[%s] #%d context: %d doc(s) resolved, %d sent%s, %d chars "
                "(~%d tokens), status=%s",
                dataset, iid, len(base["resolved_documents"]),
                base["documents_sent"],
                f" ({len(base['dropped_documents'])} dropped by cap)"
                if base["context_truncated"] else "",
                base["context_char_count"], base["context_token_estimate"],
                base["context_status"])


def _resolve_and_ground(resolver: DocumentResolver, rec: dict, prompt: str,
                        max_context_chars: int,
                        ) -> tuple[ResolutionResult, str | None, str, list[str]]:
    """Shared GROUNDED-DIFF body: resolve the record's clause_text_ids and
    prepend the document block (doc-boundary capped at max_context_chars).
    Instances with nothing resolved (none_resolved / no_ids) keep the
    untransformed parametric prompt — a grounded preamble with zero documents
    would be incoherent; they are flagged via context_status and excludable
    from the comparison."""
    res = resolver.resolve_instance(rec.get("clause_text_ids") or [])
    ctx_block, dropped = build_capped_context(res.resolved_docs, max_context_chars)
    if ctx_block is not None:
        prompt = to_grounded_prompt(prompt, ctx_block)
    return res, ctx_block, prompt, dropped


# ════════════════════════════════════════════════════════════════════════════
# Grounded evaluators — copied from PK.py, GROUNDED-DIFF fenced
# ════════════════════════════════════════════════════════════════════════════

def eval_llm_judged_grounded(dataset: str, items: list[tuple[int, dict]],
                             answer_caller: OpenRouterCaller,
                             store: pk.ResultStore,
                             resolver: DocumentResolver,
                             max_context_chars: int = MAX_CONTEXT_CHARS_DEFAULT) -> None:
    """Copied verbatim from PK.eval_llm_judged (lines 1127-1251,
    rubric v2); GROUNDED-DIFF blocks and the answer/judge caller split are
    the only changes. Answering goes to DeepSeek V4 Flash via answer_caller
    (OpenRouter); judging is the 3-judge council (pk.JUDGE_COUNCIL). 1 answer
    call + 3 council judge calls per instance (2-of-3 quorum, per-boolean
    majority); the judges see the same context as in the parametric run
    (scenario / misaligned_context / anchor), never the documents."""
    build_prompt = pk.FREE_TEXT_PROMPTS[dataset]
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        prompt, context = build_prompt(rec)

        # ── GROUNDED-DIFF: resolve source documents, prepend them (capped) ──
        res, ctx_block, prompt, dropped = _resolve_and_ground(
            resolver, rec, prompt, max_context_chars)
        # ── /GROUNDED-DIFF ──

        # Deterministic GT-side date extraction (the judge never discovers GT
        # dates) + per-item rubric flags, recorded before any API call.
        plan = rubric_scoring.prepare_gt(dataset, rec)

        base = pk._base_record(iid, rec)
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

        # ── GROUNDED-DIFF: context fields recorded before the call so
        # api_failed records carry resolution info too ──
        base.update(context_fields(res, ctx_block, dropped))
        _log_context(dataset, iid, base)
        # ── /GROUNDED-DIFF ──

        raw = answer_caller.call(prompt)
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
                pk.JUDGE_COUNCIL,
                question=rec["question"],
                ground_truth=rec["answer"],
                model_answer=raw,
                task_type=dataset,
                gt_dates=plan.gt_dates,
                context=context,
                anchor=pk._judge_anchor(dataset, rec, plan),
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
                        "models": list(pk.JUDGE_COUNCIL_MODELS),
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


def eval_temporal_mcq_grounded(dataset: str, items: list[tuple[int, dict]],
                               answer_caller: OpenRouterCaller,
                               store: pk.ResultStore,
                               resolver: DocumentResolver,
                               max_context_chars: int = MAX_CONTEXT_CHARS_DEFAULT) -> None:
    """Copied verbatim from PK.eval_temporal_mcq (lines
    1140-1186); GROUNDED-DIFF blocks and the answering-caller swap are
    the only changes. Deterministic MCQ scoring, 1 answering call per
    instance (DeepSeek V4 Flash via OpenRouter), no LLM judge."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        options = rec["options"]
        options_block = "\n".join(f"Option {i}: {opt}" for i, opt in enumerate(options))
        prompt = pk.MCQ_PROMPT_TEMPLATE.format(
            preamble=pk.PARAMETRIC_PREAMBLE, question=rec["question"],
            options_block=options_block,
        )

        # ── GROUNDED-DIFF: resolve source documents, prepend them (capped) ──
        res, ctx_block, prompt, dropped = _resolve_and_ground(
            resolver, rec, prompt, max_context_chars)
        # ── /GROUNDED-DIFF ──

        base = pk._base_record(iid, rec)
        base["options"] = options
        base["answer_index"] = rec["answer_index"]
        base["answer_text"] = rec.get("answer")

        # ── GROUNDED-DIFF ──
        base.update(context_fields(res, ctx_block, dropped))
        _log_context(dataset, iid, base)
        # ── /GROUNDED-DIFF ──

        raw = answer_caller.call(prompt, response_schema=pk.MCQ_RESPONSE_SCHEMA)
        if raw is None:
            base.update(status="api_failed", model_answer=None,
                        predicted_index=None, format_failure=False, correct=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        choice = pk.parse_choice_index(raw, len(options))
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


def eval_event_sorting_grounded(dataset: str, items: list[tuple[int, dict]],
                                answer_caller: OpenRouterCaller,
                                store: pk.ResultStore,
                                resolver: DocumentResolver,
                                max_context_chars: int = MAX_CONTEXT_CHARS_DEFAULT) -> None:
    """Copied verbatim from PK.eval_event_sorting (lines
    1189-1259); GROUNDED-DIFF blocks and the answering-caller swap are
    the only changes. Answering goes to DeepSeek V4 Flash via answer_caller.
    gt_mapping_failed instances short-circuit BEFORE any prompt exists, so
    they get no document resolution and no context fields (metrics guards
    tolerate that). The judge-assisted extraction fallback is a parsing call,
    stays ungrounded and single-Gemini (not a council verdict), and runs on
    the council's Gemini seat (its schema is Gemini-style)."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        events = rec["events_shuffled"]
        gt_order = pk.map_ground_truth_order(events, rec["answer_ordered"])

        base = pk._base_record(iid, rec)
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
        prompt = pk.EVENT_SORTING_PROMPT_TEMPLATE.format(
            preamble=pk.PARAMETRIC_PREAMBLE, question=rec["question"],
            events_block=events_block,
        )

        # ── GROUNDED-DIFF: resolve source documents, prepend them (capped) ──
        res, ctx_block, prompt, dropped = _resolve_and_ground(
            resolver, rec, prompt, max_context_chars)
        base.update(context_fields(res, ctx_block, dropped))
        _log_context(dataset, iid, base)
        # ── /GROUNDED-DIFF ──

        raw = answer_caller.call(prompt, response_schema=pk.EVENT_ORDER_RESPONSE_SCHEMA)
        if raw is None:
            base.update(status="api_failed", model_answer=None,
                        predicted_order_indices=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        pred, method = pk.parse_event_order(raw, events)
        if pred is None:
            # Judge-assisted parsing fallback (exactly one extra call) on the
            # council's Gemini seat (JUDGE_CALLERS[0] = GeminiCaller).
            logger.warning("[%s] #%d deterministic parsing failed — trying "
                           "judge-assisted extraction", dataset, iid)
            pred = llm_council.extract_event_order(
                pk.JUDGE_CALLERS[0].call, events, raw)
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
            tau, pairwise = pk.order_agreement_metrics(pred, gt_order)
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


def eval_perturbation_grounded(dataset: str, items: list[tuple[int, dict]],
                               answer_caller: OpenRouterCaller,
                               store: pk.ResultStore,
                               resolver: DocumentResolver,
                               max_context_chars: int = MAX_CONTEXT_CHARS_DEFAULT) -> None:
    """Copied verbatim from PK.eval_perturbation (lines
    1376-1548, rubric v2); GROUNDED-DIFF blocks and the answer/judge caller
    split are the only changes. Verdict and explanation prompts go to
    DeepSeek V4 Flash via answer_caller; the explanation is judged by the
    3-judge council (pk.JUDGE_COUNCIL, 2-of-3 quorum). Documents are resolved
    once per instance and the same block is prepended to BOTH the verdict and
    the explanation prompts (context_char_count is counted once). The
    explanation judging (context=statement, gated on perturbation_localized)
    and the false-positive auto-fail branch are copied unchanged."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        statement = rec["statement"]

        base = pk._base_record(iid, rec)
        base.update(
            statement=statement,
            is_perturbed=rec["is_perturbed"],
            perturbed_fact=rec.get("perturbed_fact"),
            correct_fact=rec.get("correct_fact"),
            ground_truth_answer=rec["answer"],
            rubric_version=rubric_scoring.RUBRIC_VERSION,
        )

        # ── GROUNDED-DIFF: resolve once per instance (capped block shared by
        # both prompts) ──
        res = resolver.resolve_instance(rec.get("clause_text_ids") or [])
        ctx_block, dropped = build_capped_context(res.resolved_docs, max_context_chars)
        base.update(context_fields(res, ctx_block, dropped))
        _log_context(dataset, iid, base)
        # ── /GROUNDED-DIFF ──

        # ── Step A: verdict ──
        verdict_prompt = pk.PERTURB_VERDICT_PROMPT_TEMPLATE.format(
            preamble=pk.PARAMETRIC_PREAMBLE, statement=statement)
        # ── GROUNDED-DIFF ──
        if ctx_block is not None:
            verdict_prompt = to_grounded_prompt(verdict_prompt, ctx_block)
        # ── /GROUNDED-DIFF ──
        raw_verdict = answer_caller.call(verdict_prompt, response_schema=pk.VERDICT_RESPONSE_SCHEMA)
        if raw_verdict is None:
            base.update(status="api_failed", api_failed_stage="verdict")
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED (verdict) — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        verdict = pk.parse_verdict(raw_verdict)
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
            explain_prompt = pk.PERTURB_EXPLAIN_PROMPT_TEMPLATE.format(
                preamble=pk.PARAMETRIC_PREAMBLE, statement=statement)
            # ── GROUNDED-DIFF: same document block on the explanation prompt ──
            if ctx_block is not None:
                explain_prompt = to_grounded_prompt(explain_prompt, ctx_block)
            # ── /GROUNDED-DIFF ──
            raw_expl = answer_caller.call(explain_prompt)
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
                        pk.JUDGE_COUNCIL,
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
                                "models": list(pk.JUDGE_COUNCIL_MODELS),
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


GROUNDED_EVALUATORS = {
    "context_misalignment": eval_llm_judged_grounded,
    "event_sorting": eval_event_sorting_grounded,
    "open_ended": eval_llm_judged_grounded,
    "perturbation_detection": eval_perturbation_grounded,
    "relative_time_qa": eval_llm_judged_grounded,
    "scenario_short_answer": eval_llm_judged_grounded,
    "temporal_mcq": eval_temporal_mcq_grounded,
    "temporal_scenario_short_answer": eval_llm_judged_grounded,
}
# Drift tripwire: a dataset added/removed upstream must be mirrored here.
assert set(GROUNDED_EVALUATORS) == set(pk.DATASET_EVALUATORS)


# ════════════════════════════════════════════════════════════════════════════
# Metrics / summary / comparison
# ════════════════════════════════════════════════════════════════════════════

def _dist_stats(values: list) -> dict | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return {
        "min": min(vals),
        "median": statistics.median(vals),
        "max": max(vals),
        "mean": round(statistics.mean(vals), 1),
    }


def compute_grounded_metrics(dataset: str, records: dict, n_dataset_total: int,
                             model: str) -> dict:
    """Parametric metrics (unchanged, via PK.DATASET_METRICS) plus
    grounded-run context statistics."""
    m = pk.DATASET_METRICS[dataset](records, n_dataset_total, dataset, model)
    recs = list(records.values())
    # isinstance guards: gt_mapping_failed records (and any legacy records)
    # carry no context fields.
    sizes = [r["context_char_count"] for r in recs
             if isinstance(r.get("context_char_count"), int)]
    m["evaluation_mode"] = "grounded_document_context"
    m["context_stats"] = {
        "n_records_with_context_fields": len(sizes),
        "by_context_status": dict(Counter(
            r.get("context_status") for r in recs if r.get("context_status"))),
        "n_records_with_unresolved_ids": sum(1 for r in recs if r.get("unresolved_ids")),
        "docs_per_instance": _dist_stats(
            [len(r["resolved_documents"]) for r in recs
             if isinstance(r.get("resolved_documents"), list)]),
        "context_char_count": _dist_stats(sizes),
        "context_token_estimate": _dist_stats([s // 4 for s in sizes]),
        "token_estimate_note": "chars//4 heuristic; underestimates Bengali-script "
                               "token counts.",
        "n_records_truncated": sum(1 for r in recs if r.get("context_truncated")),
        "n_documents_dropped_total": sum(
            len(r["dropped_documents"]) for r in recs
            if isinstance(r.get("dropped_documents"), list)),
    }
    if dataset == "context_misalignment":
        # Conflict resolution, not misalignment robustness — see CONDITION_NOTE.
        m["condition"] = "context_conflict_resolution"
        m["condition_note"] = CONDITION_NOTE
    return m


def write_grounded_summary(results_dir: Path, model: str,
                           answer_caller: OpenRouterCaller,
                           key_manager: pk.GeminiApiKeyManager,
                           resolver: DocumentResolver) -> None:
    """pk.write_summary works unchanged (it reads <results_dir>/<d>/metrics.json);
    its caller counters come from the OpenRouter answering caller and the
    per-judge council counters land in run_stats.judge_api_calls. Augment its
    output with the grounded-run resolution stats and the answer/judge
    backend split."""
    pk.write_summary(results_dir, model, answer_caller, key_manager,
                     judge_callers=pk.JUDGE_CALLERS)
    path = results_dir / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not augment %s: %s", path, e)
        return
    summary["evaluation_mode"] = "grounded_document_context"
    summary["answer_backend"] = "openrouter"
    summary["answer_model"] = answer_caller.model
    summary["judge_backend"] = "council"
    summary["judge_council"] = list(pk.JUDGE_COUNCIL_MODELS)
    summary["judge_api_calls_this_run"] = sum(
        c.api_calls for c in pk.JUDGE_CALLERS)
    summary["judge_api_error_events_this_run"] = sum(
        c.api_failures for c in pk.JUDGE_CALLERS)
    summary["judge_key_rotations_this_run"] = sum(
        getattr(c, "rotation_count", 0) for c in pk.JUDGE_CALLERS)
    summary["resolution"] = resolver.resolution_summary()
    pk.write_json_atomic(path, summary)


# Dotted paths into each dataset's metrics.json, per dataset type (rubric v2:
# judged datasets compare mean scores, not judge-verdict accuracy; pre-v2
# metrics.json files lack these paths, so their deltas are skipped — pre-v2
# runs are incomparable by design).
_JUDGED_COMPARISON = ["mean_final_score", "mean_date_score",
                      "meaning_correct_rate"]
COMPARISON_METRICS: dict[str, list[str]] = {
    "context_misalignment": _JUDGED_COMPARISON + [
        "context_adherence_rate", "conflict_acknowledgement_rate"],
    "open_ended": _JUDGED_COMPARISON,
    "relative_time_qa": _JUDGED_COMPARISON,
    "scenario_short_answer": _JUDGED_COMPARISON,
    "temporal_scenario_short_answer": _JUDGED_COMPARISON,
    "temporal_mcq": ["accuracy", "f1_macro"],
    "event_sorting": ["exact_match_accuracy", "kendall_tau_mean",
                      "pairwise_order_accuracy"],
    "perturbation_detection": [
        "detection.accuracy", "detection.f1",
        "explanation.mean_explanation_final_score_over_detected",
        "explanation.mean_explanation_final_score_over_all_perturbed",
        "explanation.localization_rate"],
}


def _dig(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def write_comparison(results_dir: Path, parametric_dir: Path, model: str) -> None:
    """Side-by-side parametric vs grounded metrics. Reads the parametric dir
    (read-only) and this run's metrics; skipped entirely when the parametric
    results do not exist yet."""
    if not parametric_dir.is_dir():
        logger.info("Parametric results dir %s not found — skipping comparison.json",
                    parametric_dir)
        return
    datasets: dict[str, dict] = {}
    for name in sorted(pk.DATASET_EVALUATORS):
        p_path = parametric_dir / name / "metrics.json"
        g_path = results_dir / name / "metrics.json"
        if not p_path.is_file() or not g_path.is_file():
            logger.warning("comparison: skipping %s (no metrics at %s)",
                           name, p_path if not p_path.is_file() else g_path)
            continue
        try:
            pm = json.loads(p_path.read_text(encoding="utf-8"))
            gm = json.loads(g_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("comparison: skipping %s (%s)", name, e)
            continue
        entry: dict = {
            "parametric": {"n_with_records": pm.get("n_with_records"),
                           "n_dataset_total": pm.get("n_dataset_total")},
            "grounded": {"n_with_records": gm.get("n_with_records"),
                         "n_dataset_total": gm.get("n_dataset_total")},
            "delta": {},
        }
        for metric in COMPARISON_METRICS[name]:
            pv, gv = _dig(pm, metric), _dig(gm, metric)
            entry["parametric"][metric] = pv
            entry["grounded"][metric] = gv
            if isinstance(pv, (int, float)) and isinstance(gv, (int, float)):
                entry["delta"][metric] = round(gv - pv, 4)
        entry["coverage_mismatch"] = (
            pm.get("n_with_records") != gm.get("n_with_records"))
        entry["rubric_version"] = gm.get("rubric_version")
        if pm.get("rubric_version") != gm.get("rubric_version"):
            entry["rubric_version_mismatch"] = (
                f"parametric={pm.get('rubric_version')} vs "
                f"grounded={gm.get('rubric_version')} — pre-v2 runs are "
                "incomparable; missing metrics are skipped.")
        if name == "context_misalignment":
            entry["condition"] = "context_conflict_resolution"
            entry["condition_note"] = CONDITION_NOTE
        datasets[name] = entry
    payload = {
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parametric_dir": str(parametric_dir),
        "grounded_dir": str(results_dir),
        "note": "delta = grounded - parametric. Deltas are only meaningful when "
                "both runs cover the same instances (see coverage_mismatch).",
        "datasets": datasets,
    }
    pk.write_json_atomic(results_dir / "comparison.json", payload)
    logger.info("Saved parametric-vs-grounded comparison: %s",
                results_dir / "comparison.json")


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(results_dir: Path) -> None:
    """Copy of pk.setup_logging with the log file renamed to context_check.log."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = pk.TqdmLoggingHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(results_dir / "context_check.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    # Root-logger config also captures the imported modules' loggers
    # ("param_check": key rotation/backoff; "llm_council"; "openrouter";
    # "document_resolver"), so quota churn stays visible in this run's log.
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded (document-as-context) evaluation of DeepSeek V4 Flash "
                    "(via OpenRouter) on the ICSETEP concept-QA benchmark; "
                    "judging is the 3-judge LLM council.")
    parser.add_argument("--datasets", nargs="+",
                        choices=sorted(GROUNDED_EVALUATORS), metavar="NAME",
                        default=sorted(GROUNDED_EVALUATORS),
                        help="Datasets to evaluate (default: all 8). "
                             f"Choices: {', '.join(sorted(GROUNDED_EVALUATORS))}")
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
    parser.add_argument("--openrouter-key-file", default=OPENROUTER_KEY_FILE_DEFAULT,
                        help="File holding the OpenRouter API key, used for "
                             "the answering calls AND the two OpenRouter "
                             "council judges "
                             f"(default: {OPENROUTER_KEY_FILE_DEFAULT}; "
                             "falls back to $OPENROUTER_API_KEY).")
    parser.add_argument("--judge-keys-file", default=JUDGE_KEYS_FILE_DEFAULT,
                        help="Gemini API keys file for the council's Gemini "
                             f"judge seat (default: {JUDGE_KEYS_FILE_DEFAULT}).")
    parser.add_argument("--model", default=OPENROUTER_MODEL_DEFAULT,
                        help="OpenRouter model for ANSWERING "
                             f"(default: {OPENROUTER_MODEL_DEFAULT}). The "
                             "judge council is fixed: "
                             f"{', '.join(pk.JUDGE_COUNCIL_MODELS)}.")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help=f"Output directory (default: {RESULTS_DIR}).")
    parser.add_argument("--parametric-dir", default=pk.RESULTS_DIR,
                        help="Parametric results directory to compare against "
                             f"(read-only; default: {pk.RESULTS_DIR}). "
                             "comparison.json is skipped if it does not exist.")
    parser.add_argument("--max-context-chars", type=int,
                        default=MAX_CONTEXT_CHARS_DEFAULT,
                        help="Cap on the document block per request, dropping "
                             "whole documents from the end of the list "
                             f"(default: {MAX_CONTEXT_CHARS_DEFAULT}; 0 = "
                             "unlimited). DeepSeek V4 Flash's context window is "
                             "1M tokens — requests above it fail on every "
                             "retry, so the cap is kept as a safety margin.")
    parser.add_argument("--openrouter-delay", type=float,
                        default=OPENROUTER_RATE_LIMIT_DELAY,
                        help="Seconds to sleep before every OpenRouter "
                             "answering call "
                             f"(default: {OPENROUTER_RATE_LIMIT_DELAY}).")
    parser.add_argument("--rate-limit-delay", type=float, default=JUDGE_RATE_LIMIT_DELAY,
                        help="Seconds to sleep before every Gemini JUDGE call "
                             f"(default: {JUDGE_RATE_LIMIT_DELAY}).")
    parser.add_argument("--calls-per-day", type=int, default=JUDGE_CALLS_PER_DAY,
                        help="Max JUDGE calls per Gemini key per day "
                             f"(default: {JUDGE_CALLS_PER_DAY}; effectively "
                             "unlimited for the paid key).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    setup_logging(results_dir)

    logger.info("=" * 70)
    logger.info("Grounded (document-as-context) check | answering=%s (OpenRouter) "
                "| judge_council=%s | datasets=%s | limit=%s | resume=%s | "
                "max_context_chars=%s",
                args.model, "+".join(pk.JUDGE_COUNCIL_MODELS),
                ",".join(args.datasets), args.limit, args.resume,
                args.max_context_chars or "unlimited")
    logger.info("=" * 70)

    # Fail fast on a missing corpus before touching API keys.
    resolver = DocumentResolver()

    openrouter_key = load_openrouter_key(args.openrouter_key_file)
    answer_caller = OpenRouterCaller(
        api_key=openrouter_key,
        model=args.model,
        rate_limit_delay=args.openrouter_delay,
    )
    key_manager = pk.GeminiApiKeyManager(
        keys_file=args.judge_keys_file,
        calls_per_day=args.calls_per_day,
        rate_limit_delay=args.rate_limit_delay,
    )
    # The copied evaluator bodies read the council from PK's module-globals
    # (same objects for both drivers on purpose). The council's Gemini seat
    # rotates over --judge-keys-file; the Claude/GPT seats reuse the
    # answering OpenRouter key.
    pk.JUDGE_COUNCIL, pk.JUDGE_CALLERS = pk.build_judge_council(
        key_manager, openrouter_key)

    try:
        for dataset in args.datasets:
            qa_pairs = pk.load_dataset(dataset)
            items = list(enumerate(qa_pairs))
            if args.limit is not None:
                items = items[: args.limit]

            store = pk.ResultStore(results_dir, dataset, args.model)
            if args.resume:
                loaded = store.load()
                if loaded:
                    logger.info("[%s] resume: %d existing record(s), %d completed",
                                dataset, loaded, len(store.completed_ids()))

            pending = [i for i, _ in items if i not in store.completed_ids()]
            logger.info("=== %s: evaluating %d of %d instances (%d already done) ===",
                        dataset, len(pending), len(items), len(items) - len(pending))

            GROUNDED_EVALUATORS[dataset](dataset, items, answer_caller,
                                         store, resolver,
                                         args.max_context_chars)

            metrics = compute_grounded_metrics(
                dataset, store.records, len(qa_pairs), args.model)
            pk.write_json_atomic(results_dir / dataset / "metrics.json", metrics)
            logger.info("[%s] metrics saved: %s", dataset,
                        results_dir / dataset / "metrics.json")
            if args.calibration_dump and dataset in (*pk.JUDGED_DATASETS,
                                                     "perturbation_detection"):
                pk.write_calibration_dump(
                    store.records,
                    results_dir / dataset / "calibration_dump.csv",
                    args.calibration_dump, dataset)
            s = resolver.stats
            logger.info("[%s] resolution so far (this run): %d/%d ids resolved "
                        "(%d unresolved); instances full=%d partial=%d "
                        "none_resolved=%d no_ids=%d",
                        dataset, s.resolved_ids, s.total_ids, s.unresolved_ids,
                        s.instances_full, s.instances_partial,
                        s.instances_none, s.instances_no_ids)
    except KeyboardInterrupt:
        logger.warning("Interrupted — progress is checkpointed; re-run with "
                       "--resume (default) to continue.")
    finally:
        key_manager.save_key_status_report(results_dir / "api_key_status.txt")
        write_grounded_summary(results_dir, args.model, answer_caller,
                               key_manager, resolver)
        resolver.write_unresolved_log(results_dir / "resolution_report.txt")
        s = resolver.stats
        logger.info("Resolution totals (this run): %d ids -> %d resolved, "
                    "%d unresolved (%d distinct).",
                    s.total_ids, s.resolved_ids, s.unresolved_ids,
                    len(s.unresolved_id_set))
        write_comparison(results_dir, Path(args.parametric_dir), args.model)
        logger.info("Run stats: answering (OpenRouter/%s): %d API calls, "
                    "%d error events. Judge calls: %s.",
                    answer_caller.model, answer_caller.api_calls,
                    answer_caller.api_failures,
                    ", ".join(f"{c.model}={c.api_calls}"
                              for c in pk.JUDGE_CALLERS) or "none")


if __name__ == "__main__":
    main()
