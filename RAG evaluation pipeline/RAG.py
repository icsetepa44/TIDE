"""RAG.py — RAG evaluation of OpenRouter models on the ICSETEP customs-QA benchmark.

Self-contained driver:
(1) context is retrieved from our ChromaDB via RagRetriever (rag_lib/rag_retriever.py),
(2) the models under test answer on OpenRouter (or directly on the Gemini API when
    USE_GEMINI_API is True), and
(3) judging is the 3-model LLM council (llm_council.py: gemini-3.5-flash +
    anthropic/claude-opus-4.6 + openai/gpt-5.1, per-boolean majority, 2-of-3 quorum).

Scoring is the deterministic rubric v2 in rubric_scoring.py; the eight QA datasets live
in concept_qa_by_type/; all other logic lives in rag_lib/. (The evaluation rubric follows
the ICSETEP benchmark methodology.)

Configuration is via the module-level globals below — no CLI.

Run:
    # copy .env.example -> .env and set GEMINI_API_KEYS / OPENROUTER_API_KEYS
    python RAG.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

# The helper/config modules live in OpenRouter/rag_lib/; the shared judge modules
# (llm_council, rubric_scoring) live in OpenRouter/ itself. Put both on the path.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "rag_lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
import eval_lib as E  # noqa: E402
import llm_council  # noqa: E402
import rubric_scoring  # noqa: E402
from key_manager import KeyManager  # noqa: E402
from llm_clients import GeminiCaller, OpenRouterCaller, OpenRouterKeyManager  # noqa: E402
from rag_retriever import RagRetriever  # noqa: E402

logger = logging.getLogger("rag_check")

# ════════════════════════════════════════════════════════════════════════════
# Configuration (globals — no CLI)
# ════════════════════════════════════════════════════════════════════════════

OPENROUTER_MODELS = [
    "moonshotai/kimi-k2.5",
    "openai/gpt-oss-120b",
    "z-ai/glm-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
    "qwen/qwen3.6-35b-a3b",
    "google/gemma-4-31b-it",
    "deepseek/deepseek-v4-flash",
]

# True  -> run GEMINI_MODELS directly on the Gemini API (vault's Gemini keys).
# False -> run OPENROUTER_MODELS on OpenRouter (the default).
USE_GEMINI_API = False
GEMINI_MODELS = [
    "gemini-3.5-flash",
]

MODELS = GEMINI_MODELS if USE_GEMINI_API else OPENROUTER_MODELS
RESULTS_ROOT = config.RAG_RESULTS_ROOT           # OpenRouter/rag_check_results
TOP_K = 10
# MAX_CONTEXT_CHARS = E.MAX_CONTEXT_CHARS_DEFAULT  # 400_000 (0 = unlimited)
MAX_CONTEXT_CHARS = 0  # 400_000 (0 = unlimited)
LIMIT = 0                                     # int -> first N instances/dataset (smoke)
RESUME = True                                    # skip completed ids; retry api_failed

# ── LLM-judge council (3 seats, per-boolean majority, 2-of-3 quorum) ──
# Same trio as the PK/ICL settings so the RAG metric is directly comparable.
# Seat 1 is Gemini (native structured output); seats 2-3 answer on OpenRouter.
GEMINI_JUDGE_MODEL = "gemini-3.5-flash"
OPENROUTER_JUDGE_MODELS = ("anthropic/claude-opus-4.6", "openai/gpt-5.1")
JUDGE_COUNCIL_MODELS = (GEMINI_JUDGE_MODEL, *OPENROUTER_JUDGE_MODELS)

# Which concept_qa_by_type datasets to run, and how many samples from each:
#   True  -> all instances (or LIMIT, if set)
#   False -> skip this dataset
#   int N -> first N instances of this dataset (overrides LIMIT)
RUN_DATASET = {
    "context_misalignment": True,
    "event_sorting": True,
    "open_ended": True,
    "perturbation_detection": True,
    "relative_time_qa": True,
    "scenario_short_answer": True,
    "temporal_mcq": True,
    "temporal_scenario_short_answer": True,
}
assert set(RUN_DATASET) == set(E.DATASET_METRICS), "RUN_DATASET keys drifted"


def _dataset_limit(spec):
    """Map a RUN_DATASET value to a per-dataset instance cap (None = all)."""
    if isinstance(spec, bool):
        return LIMIT
    return spec


# ════════════════════════════════════════════════════════════════════════════
# Evaluators — teammate's grounded loops, retrieval swapped to RagRetriever
# ════════════════════════════════════════════════════════════════════════════

def eval_llm_judged(dataset, items, caller, judge_council, extract_call, store,
                    retriever, max_context_chars=MAX_CONTEXT_CHARS):
    """5 free-text datasets, rubric v2. 1 answer call + 3 council judge calls per
    instance (per-boolean majority, 2-of-3 quorum). The judges see the dataset's
    own scenario / misaligned_context / anchor as `context`, never the retrieved
    documents."""
    build_prompt = E.FREE_TEXT_PROMPTS[dataset]
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        prompt, context = build_prompt(rec)

        # ── RAG: retrieve source documents, prepend them (capped) ──
        res = retriever.retrieve(dataset, rec)
        ctx_block, dropped = E.build_capped_context(res.resolved_docs, max_context_chars)
        if ctx_block is not None:
            prompt = E.to_grounded_prompt(prompt, ctx_block)

        # Deterministic GT-side date extraction + per-item rubric flags.
        plan = rubric_scoring.prepare_gt(dataset, rec)

        base = E._base_record(iid, rec)
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

        base.update(E.context_fields(res, ctx_block, dropped))
        E._log_context(dataset, iid, base)

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
                judge_council,
                question=rec["question"],
                ground_truth=rec["answer"],
                model_answer=raw,
                task_type=dataset,
                gt_dates=plan.gt_dates,
                context=context,
                anchor=E._judge_anchor(dataset, rec, plan),
                required_granularity=plan.required_granularity,
                extra_gt=({"conflict": rec.get("conflict")}
                          if dataset == "context_misalignment" else None),
            )
            if ar.error == "judge_api_failed":
                # Council below quorum with an API outage contributing — retried
                # on resume. A single judge outage degrades to 2-of-3 instead.
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
                    base["context_atoms"] = {
                        k: ar.atoms.extras.get(k)
                        for k in ("followed_context", "followed_parametric",
                                  "conflict_acknowledged", "blended_sources")}

        base["elapsed_s"] = round(time.time() - t0, 2)
        store.upsert(base)
        logger.info("[%s] %d/%d (#%d) -> score=%s (%.1fs)",
                    dataset, pos, len(items), iid, base.get("final_score"),
                    base["elapsed_s"])


def eval_temporal_mcq(dataset, items, caller, judge_council, extract_call, store,
                      retriever, max_context_chars=MAX_CONTEXT_CHARS):
    """Deterministic MCQ scoring, 1 call per instance, no LLM judge."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        options = rec["options"]
        options_block = "\n".join(f"Option {i}: {opt}" for i, opt in enumerate(options))
        prompt = E.MCQ_PROMPT_TEMPLATE.format(
            preamble=E.PARAMETRIC_PREAMBLE, question=rec["question"],
            options_block=options_block,
        )

        # ── RAG ──
        res = retriever.retrieve(dataset, rec)
        ctx_block, dropped = E.build_capped_context(res.resolved_docs, max_context_chars)
        if ctx_block is not None:
            prompt = E.to_grounded_prompt(prompt, ctx_block)

        base = E._base_record(iid, rec)
        base["options"] = options
        base["answer_index"] = rec["answer_index"]
        base["answer_text"] = rec.get("answer")

        base.update(E.context_fields(res, ctx_block, dropped))
        E._log_context(dataset, iid, base)

        raw = caller.call(prompt, response_schema=E.MCQ_RESPONSE_SCHEMA)
        if raw is None:
            base.update(status="api_failed", model_answer=None,
                        predicted_index=None, format_failure=False, correct=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        choice = E.parse_choice_index(raw, len(options))
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


def eval_event_sorting(dataset, items, caller, judge_council, extract_call, store,
                       retriever, max_context_chars=MAX_CONTEXT_CHARS):
    """Deterministic ordering metrics. gt_mapping_failed instances short-circuit
    BEFORE any prompt exists (no retrieval, no context fields). The judge-assisted
    extraction fallback is a single-judge parsing call (the council's Gemini seat),
    not a council verdict, and stays ungrounded."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        events = rec["events_shuffled"]
        gt_order = E.map_ground_truth_order(events, rec["answer_ordered"])

        base = E._base_record(iid, rec)
        base["events_shuffled"] = events
        base["answer_ordered"] = rec["answer_ordered"]
        base["ground_truth_order_indices"] = gt_order

        if gt_order is None:
            base.update(status="gt_mapping_failed", model_answer=None,
                        predicted_order_indices=None)
            store.upsert(base)
            logger.error("[%s] #%d could not map answer_ordered onto "
                         "events_shuffled — instance skipped", dataset, iid)
            continue

        events_block = "\n".join(f"Event {i}: {e}" for i, e in enumerate(events))
        prompt = E.EVENT_SORTING_PROMPT_TEMPLATE.format(
            preamble=E.PARAMETRIC_PREAMBLE, question=rec["question"],
            events_block=events_block,
        )

        # ── RAG ──
        res = retriever.retrieve(dataset, rec)
        ctx_block, dropped = E.build_capped_context(res.resolved_docs, max_context_chars)
        if ctx_block is not None:
            prompt = E.to_grounded_prompt(prompt, ctx_block)
        base.update(E.context_fields(res, ctx_block, dropped))
        E._log_context(dataset, iid, base)

        raw = caller.call(prompt, response_schema=E.EVENT_ORDER_RESPONSE_SCHEMA)
        if raw is None:
            base.update(status="api_failed", model_answer=None,
                        predicted_order_indices=None)
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        pred, method = E.parse_event_order(raw, events)
        if pred is None:
            logger.warning("[%s] #%d deterministic parsing failed — trying "
                           "judge-assisted extraction", dataset, iid)
            pred = llm_council.extract_event_order(extract_call, events, raw)
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
            tau, pairwise = E.order_agreement_metrics(pred, gt_order)
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


def eval_perturbation(dataset, items, caller, judge_council, extract_call, store,
                      retriever, max_context_chars=MAX_CONTEXT_CHARS):
    """Two-step: CORRECT/INCORRECT verdict + council-judged explanation. Documents
    are retrieved once per instance and the same block is prepended to BOTH the
    verdict and the explanation prompts."""
    done = store.completed_ids()
    for pos, (iid, rec) in enumerate(tqdm(items, desc=dataset, unit="qa"), 1):
        if iid in done:
            continue
        t0 = time.time()
        statement = rec["statement"]

        base = E._base_record(iid, rec)
        base.update(
            statement=statement,
            is_perturbed=rec["is_perturbed"],
            perturbed_fact=rec.get("perturbed_fact"),
            correct_fact=rec.get("correct_fact"),
            ground_truth_answer=rec["answer"],
            rubric_version=rubric_scoring.RUBRIC_VERSION,
        )

        # ── RAG: retrieve once; the capped block is shared by both prompts ──
        res = retriever.retrieve(dataset, rec)
        ctx_block, dropped = E.build_capped_context(res.resolved_docs, max_context_chars)
        base.update(E.context_fields(res, ctx_block, dropped))
        E._log_context(dataset, iid, base)

        # ── Step A: verdict ──
        verdict_prompt = E.PERTURB_VERDICT_PROMPT_TEMPLATE.format(
            preamble=E.PARAMETRIC_PREAMBLE, statement=statement)
        if ctx_block is not None:
            verdict_prompt = E.to_grounded_prompt(verdict_prompt, ctx_block)
        raw_verdict = caller.call(verdict_prompt, response_schema=E.VERDICT_RESPONSE_SCHEMA)
        if raw_verdict is None:
            base.update(status="api_failed", api_failed_stage="verdict")
            store.upsert(base)
            logger.error("[%s] %d/%d (#%d) API FAILED (verdict) — will retry on resume",
                         dataset, pos, len(items), iid)
            continue

        verdict = E.parse_verdict(raw_verdict)
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
            explain_prompt = E.PERTURB_EXPLAIN_PROMPT_TEMPLATE.format(
                preamble=E.PARAMETRIC_PREAMBLE, statement=statement)
            if ctx_block is not None:
                explain_prompt = E.to_grounded_prompt(explain_prompt, ctx_block)
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
                        judge_council,
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


EVALUATORS = {
    "context_misalignment": eval_llm_judged,
    "event_sorting": eval_event_sorting,
    "open_ended": eval_llm_judged,
    "perturbation_detection": eval_perturbation,
    "relative_time_qa": eval_llm_judged,
    "scenario_short_answer": eval_llm_judged,
    "temporal_mcq": eval_temporal_mcq,
    "temporal_scenario_short_answer": eval_llm_judged,
}
# Drift tripwire: every metric-bearing dataset must have an evaluator.
assert set(EVALUATORS) == set(E.DATASET_METRICS)


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    E.setup_logging(RESULTS_ROOT)
    E.JUDGE_RUN_CONFIG.update(judge_council=list(JUDGE_COUNCIL_MODELS))

    datasets = [d for d, on in RUN_DATASET.items() if on]
    dataset_desc = ",".join(
        f"{d}[{_dataset_limit(RUN_DATASET[d]) or 'all'}]" for d in datasets)
    logger.info("=" * 70)
    logger.info("RAG check | models=%d | api=%s | judge_council=%s | datasets=%s | "
                "top_k=%d | limit=%s | resume=%s | max_context_chars=%s",
                len(MODELS), "gemini" if USE_GEMINI_API else "openrouter",
                "+".join(JUDGE_COUNCIL_MODELS), dataset_desc, TOP_K,
                LIMIT, RESUME, MAX_CONTEXT_CHARS or "unlimited")
    logger.info("=" * 70)

    # Shared key pools, loaded once from plaintext (no vault, no password prompt):
    #   gemini_km      -> query embedding + the council's Gemini judge seat
    #                     (+ Gemini answering when USE_GEMINI_API)
    #   openrouter_km  -> the models under test + the council's two OpenRouter seats
    gemini_km = KeyManager.load(section=config.SECTION_GEMINI)
    openrouter_km = OpenRouterKeyManager.load(section=config.SECTION_OPENROUTER)

    # Retriever + judge council are built once and reused across all models.
    retriever = RagRetriever(top_k=TOP_K, key_manager=gemini_km)
    gem_judge = GeminiCaller(model=GEMINI_JUDGE_MODEL, key_manager=gemini_km)
    or_judges = [OpenRouterCaller(m, openrouter_km) for m in OPENROUTER_JUDGE_MODELS]
    judge_callers = [gem_judge, *or_judges]
    judge_council = [(c.model, E.make_judge_call_fn(c)) for c in judge_callers]
    # Event-order parsing fallback is single-judge (the council's Gemini seat).
    extract_call = E.make_judge_call_fn(gem_judge)

    for model in MODELS:
        slug = model.replace("/", "__")
        results_dir = RESULTS_ROOT / slug
        logger.info("#" * 70)
        logger.info("MODEL %s -> %s", model, results_dir)
        logger.info("#" * 70)

        retriever.reset_stats()
        if USE_GEMINI_API:
            caller = GeminiCaller(model=model, key_manager=gemini_km)
        else:
            caller = OpenRouterCaller(model, openrouter_km)

        try:
            for dataset in datasets:
                if dataset not in EVALUATORS:
                    logger.warning("No evaluator for dataset %s — skipped", dataset)
                    continue
                qa_pairs = E.load_dataset(dataset)
                items = list(enumerate(qa_pairs))
                limit = _dataset_limit(RUN_DATASET[dataset])
                if limit is not None:
                    items = items[:limit]

                store = E.ResultStore(results_dir, dataset, model)
                if RESUME:
                    loaded = store.load()
                    if loaded:
                        logger.info("[%s] resume: %d existing record(s), %d completed",
                                    dataset, loaded, len(store.completed_ids()))

                pending = [i for i, _ in items if i not in store.completed_ids()]
                logger.info("=== %s: evaluating %d of %d instances (%d already done) ===",
                            dataset, len(pending), len(items), len(items) - len(pending))

                EVALUATORS[dataset](dataset, items, caller, judge_council,
                                    extract_call, store, retriever, MAX_CONTEXT_CHARS)

                metrics = E.compute_grounded_metrics(
                    dataset, store.records, len(qa_pairs), model)
                E.write_json_atomic(results_dir / dataset / "metrics.json", metrics)
                logger.info("[%s] metrics saved: %s", dataset,
                            results_dir / dataset / "metrics.json")
        except KeyboardInterrupt:
            logger.warning("Interrupted — progress is checkpointed; re-run to "
                           "continue (resume is on).")
            E.write_grounded_summary(results_dir, model, caller,
                                     caller.key_manager, retriever, judge_callers)
            raise
        E.write_grounded_summary(results_dir, model, caller,
                                 caller.key_manager, retriever, judge_callers)
        logger.info("[%s] run stats: %d API calls, %d error events, %d key "
                    "rotations. Judge council calls: %s.",
                    model, caller.api_calls, caller.api_failures,
                    caller.rotation_count,
                    ", ".join(f"{c.model}={c.api_calls}" for c in judge_callers))


if __name__ == "__main__":
    main()
