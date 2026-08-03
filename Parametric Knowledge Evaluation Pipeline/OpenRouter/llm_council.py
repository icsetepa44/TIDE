"""
LLM-council module for the parametric knowledge evaluation pipeline.

The grading judge is a COUNCIL of heterogeneous LLM judges (in production:
gemini-3.5-flash natively, plus anthropic/claude-opus-4.6 and openai/gpt-5.1
via OpenRouter) whose atomic boolean judgments are combined by per-boolean
strict majority voting.

Exposes:
  - judge_atoms(judges, *, question, ground_truth, model_answer, task_type,
                gt_dates, ...) -> AtomJudgeResult
        Rubric v2: each council judge emits atomic booleans as strict JSON
        (per-GT-date match, contradiction, meaning_correct, uncertain,
        per-type extras); booleans are combined per atom by strict majority
        over the judges that succeeded. Judges NEVER compute a numeric score —
        all scoring is deterministic Python in rubric_scoring.py.
        `judges` is an ordered list of (judge_name, call_fn) pairs. Quorum is
        a strict majority of the council (2 of 3): a single judge outage
        degrades to voting over the survivors (a 1-1 tie resolves to False);
        the instance is judge-failed only when a majority of judges fail.
  - judge(call_fn, question, ground_truth, model_answer, task_type, context=None)
        -> JudgeResult   (LEGACY v1 single-verdict single-judge, superseded by
        judge_atoms for scoring; kept for reference)
  - extract_event_order(call_fn, shuffled_events, model_answer)
        -> list[int] | None   (judge-assisted parsing fallback for event
        sorting; deliberately single-Gemini — it recovers an ordering the
        answer expresses, there is no verdict to vote on)

Design notes:
  - This module does NOT create its own API clients. The driver
    (PK.py) injects each judge's `call_fn` with signature
    `call_fn(prompt, response_schema=None) -> str | None`: the Gemini judge
    shares the driver's rotating-key client, the OpenRouter judges use
    openrouter_caller.OpenRouterCaller. All judges run at temperature=0.
  - The Gemini judge is forced into strict JSON via a Gemini response schema;
    the OpenRouter judges are prompt-driven JSON. Chain-of-thought reasoning
    is emitted BEFORE the final decision in every case.
  - Parsing is defensive: markdown fences are stripped, a regex fallback is
    attempted, and one full repair re-call is made per judge on malformed
    JSON before that judge's vote is discarded.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import rubric_scoring

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Response schemas (Gemini structured output)
# ════════════════════════════════════════════════════════════════════════════════

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "decision": {"type": "string", "enum": ["CORRECT", "INCORRECT"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["reasoning", "decision", "confidence"],
    "propertyOrdering": ["reasoning", "decision", "confidence"],
}

EXTRACT_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "ordered_indices": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["ordered_indices"],
    "propertyOrdering": ["reasoning", "ordered_indices"],
}

# ════════════════════════════════════════════════════════════════════════════════
# Judge prompt
# ════════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT_HEADER = """\
You are a strict but fair grading judge for a question-answering benchmark on
Bangladesh customs regulations (SROs — Statutory Regulatory Orders, duty rules,
baggage rules, and trade agreements).

You will be given a question, the authoritative GROUND-TRUTH ANSWER, and a
MODEL ANSWER. Decide whether the MODEL ANSWER agrees with the GROUND-TRUTH
ANSWER on the substance of what the question asks.

Grading rules:
1. The GROUND-TRUTH ANSWER is authoritative and correct BY DEFINITION. Do not
   use your own knowledge or opinion to second-guess it. Even if you believe
   the ground truth is wrong, you must grade against it.
2. Grade ONLY semantic equivalence and factual agreement. Ignore writing
   quality, verbosity, tone, formatting, and style.
3. Mark CORRECT when the model answer conveys the same key fact(s) as the
   ground truth, including:
   - paraphrases and synonyms;
   - different date formats (e.g., "2011-06-09" vs "9 June 2011" vs "June 9, 2011");
   - equivalent numeric/unit expressions (e.g., "5%" vs "five percent");
   - extra detail that is consistent with the ground truth (additional correct
     information does not make an answer wrong).
4. Mark INCORRECT when the model answer:
   - misses the key fact the question asks for;
   - contradicts the ground truth on any key fact (wrong value, wrong date,
     wrong rate, wrong instrument/SRO, wrong direction of change);
   - gives multiple conflicting values for the same fact without clearly
     committing to one (self-contradiction);
   - hedges or refuses without committing to an answer (e.g., "it may be X or
     Y", "I don't have enough information to say");
   - answers a different question than the one asked.
5. If the ground truth contains several facts, the model must agree with the
   fact(s) the question actually asks about; omitting peripheral detail is
   acceptable, omitting or contradicting the core fact is not.
6. Distinguish omission from contradiction for specific identifiers (SRO
   numbers, H.S. codes, dates, rates): OMITTING an identifier the question
   does not ask about is acceptable, but ASSERTING a different value than the
   one the ground truth states (e.g., citing a different SRO number for the
   same instrument) is a contradiction — mark INCORRECT. Confabulated
   specifics must not be rewarded.

Worked examples:

Example 1
Question: When did the D-8 Preferential Trade Agreement become effective?
Ground-truth answer: The agreement actually became effective on August 25, 2011.
Model answer: It entered into force on 25/08/2011.
Correct verdict: {"reasoning": "The model gives the same date in a different format and 'entered into force' is a paraphrase of 'became effective'. The key fact agrees.", "decision": "CORRECT", "confidence": 95}

Example 2
Question: What is the applicable Regulatory Duty rate for copper bars under the 2015 regulation?
Ground-truth answer: The applicable Regulatory Duty rate is 15%.
Model answer: Under SRO 353 of 2015, copper bars attract a Regulatory Duty of 5%.
Correct verdict: {"reasoning": "The model states 5% while the ground truth says 15%. The key fact (the rate) is contradicted, even though the surrounding detail sounds plausible.", "decision": "INCORRECT", "confidence": 98}

Example 3
Question: Which SRO first introduced the tiered PSI fee structure?
Ground-truth answer: SRO 156/2013 formalized the tiered PSI fee structure.
Model answer: Several SROs issued between 2010 and 2014 dealt with PSI fees, so it could have been any of them; I cannot say definitively which one introduced the tiered structure.
Correct verdict: {"reasoning": "The model hedges and never commits to a specific SRO, so it fails to state the key fact the question asks for.", "decision": "INCORRECT", "confidence": 92}

Example 4
Question: What Regulatory Duty rate applied to cereal flours under the 2018 amendment?
Ground-truth answer: The Regulatory Duty rate for cereal flours was set at 10%.
Model answer: The 2018 amendment set the Regulatory Duty on cereal flours at 10%; it did so by replacing Table-II of the earlier 2017 notification.
Correct verdict: {"reasoning": "The model states the same 10% rate. The extra detail about replacing Table-II is consistent with the ground truth and does not change the key fact.", "decision": "CORRECT", "confidence": 93}

Example 5
Question: How did the duty treatment of polyester yarn change under the 2017-06-01 SRO?
Ground-truth answer: The 2017-06-01 SRO (135-AIN) introduced a 0% Customs Duty rate for polyester yarn imported by VAT-registered weaving mills.
Model answer: Under SRO 166-Ain/2017, dated 1 June 2017, polyester yarn imported by VAT-registered weaving mills became fully exempt from Customs Duty.
Correct verdict: {"reasoning": "The described change (duty-free treatment for VAT-registered weaving mills, 1 June 2017) matches, but the model asserts SRO 166-Ain where the ground truth says 135-AIN. Asserting a different identifier is a contradiction under rule 6, not a harmless omission.", "decision": "INCORRECT", "confidence": 85}
"""

JUDGE_PROMPT_FOOTER = """\
First think through the comparison step by step in the "reasoning" field, then
give the final "decision", then rate your "confidence" that your decision is
right, as an integer from 0 to 100 (90-100 = clear-cut case, 70-89 = fairly
confident, 50-69 = borderline judgment call, below 50 = essentially guessing).
Respond with strict JSON only, exactly this shape:
{"reasoning": "<step-by-step comparison>", "decision": "CORRECT" or "INCORRECT", "confidence": <integer 0-100>}
"""

# Extra guidance injected per task type (optional).
TASK_TYPE_NOTES = {
    "context_misalignment": (
        "Note: the context shown to the model deliberately contains misaligned "
        "or incorrect information. The ground-truth answer below is the "
        "authority; if the model followed the misleading context and thereby "
        "contradicts the ground truth, it is INCORRECT."
    ),
    "perturbation_explanation": (
        "Note: the model was asked WHY and WHERE a given statement is wrong. "
        "Mark CORRECT only if the model identifies the same error (and, where "
        "the ground truth gives one, the same correction) as the ground-truth "
        "answer. Spotting that something is wrong but naming a different "
        "error is INCORRECT."
    ),
}


@dataclass
class JudgeResult:
    """Outcome of one LLM-judge evaluation."""

    decision: str  # "CORRECT" | "INCORRECT" | "JUDGE_FAILED"
    reasoning: str
    raw_response: str | None
    error: str | None = None
    confidence: float | None = None  # judge's self-reported confidence, 0-100

    @property
    def ok(self) -> bool:
        """True if the judge produced a usable verdict."""
        return self.decision in ("CORRECT", "INCORRECT")

    @property
    def is_correct(self) -> bool:
        return self.decision == "CORRECT"


# ════════════════════════════════════════════════════════════════════════════════
# Defensive JSON parsing
# ════════════════════════════════════════════════════════════════════════════════

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from a response."""
    return _FENCE_RE.sub("", text.strip()).strip()


def _loads_or_none(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_confidence(value) -> float | None:
    """Coerce a confidence value to a float clamped to [0, 100], else None."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return None
    if c != c:  # NaN
        return None
    return max(0.0, min(100.0, c))


def parse_judge_response(raw: str) -> tuple[str, str, float | None] | None:
    """
    Parse the judge's raw response into (decision, reasoning, confidence).
    Confidence is None when it cannot be recovered; the whole result is None
    if no usable decision can be recovered.
    """
    cleaned = strip_json_fences(raw)

    data = _loads_or_none(cleaned)
    if data is None:
        # Sometimes valid JSON is embedded in surrounding prose — try to
        # extract the outermost object by brace matching.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            data = _loads_or_none(cleaned[start : end + 1])

    if isinstance(data, dict):
        decision = str(data.get("decision", "")).strip().upper()
        if decision in ("CORRECT", "INCORRECT"):
            return (decision, str(data.get("reasoning", "")).strip(),
                    _coerce_confidence(data.get("confidence")))

    # Regex fallback on the raw text.
    m = re.search(r'"decision"\s*:\s*"(CORRECT|INCORRECT)"', raw, re.IGNORECASE)
    if m:
        reasoning = ""
        rm = re.search(r'"reasoning"\s*:\s*"(.*?)"\s*[,}]', raw, re.DOTALL)
        if rm:
            reasoning = rm.group(1)
        confidence = None
        cm = re.search(r'"confidence"\s*:\s*"?([\d.]+)"?', raw)
        if cm:
            confidence = _coerce_confidence(cm.group(1))
        return m.group(1).upper(), reasoning.strip(), confidence

    # Last resort: a bare verdict word. INCORRECT is checked first; the \b
    # boundary prevents matching the CORRECT inside INCORRECT.
    if re.search(r"\bINCORRECT\b", raw):
        return "INCORRECT", "", None
    if re.search(r"\bCORRECT\b", raw):
        return "CORRECT", "", None

    return None


# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

def build_judge_prompt(
    question: str,
    ground_truth: str,
    model_answer: str,
    task_type: str,
    context: str | None = None,
) -> str:
    """Assemble the full judge prompt for one instance."""
    parts = [JUDGE_PROMPT_HEADER]

    note = TASK_TYPE_NOTES.get(task_type)
    if note:
        parts.append(note + "\n")

    parts.append("Now grade the following case.\n")
    if context:
        if task_type == "perturbation_explanation":
            label = (
                "Statement under evaluation (the model judged it wrong and "
                "explained why; grade that explanation against the ground truth):"
            )
        else:
            label = (
                "Context/scenario that was shown to the model (for reference "
                "only — the ground truth below remains the authority):"
            )
        parts.append(f"{label}\n{context}\n")
    parts.append(f"Question: {question}\n")
    parts.append(f"Ground-truth answer: {ground_truth}\n")
    parts.append(f"Model answer: {model_answer}\n")
    parts.append(JUDGE_PROMPT_FOOTER)
    return "\n".join(parts)


def judge(
    call_fn,
    question: str,
    ground_truth: str,
    model_answer: str,
    task_type: str,
    context: str | None = None,
) -> JudgeResult:
    """
    LEGACY v1 judge: grade `model_answer` against `ground_truth` with a single
    CORRECT/INCORRECT verdict. Superseded by `judge_atoms` (rubric v2, atomic
    booleans + deterministic scoring); kept for reference and comparability
    with pre-v2 result archives.

    `call_fn(prompt, response_schema=None) -> str | None` must be the driver's
    shared rotating Gemini wrapper (temperature=0). One re-call is made if the
    first response is malformed; after that the instance is a judge-failure.
    """
    prompt = build_judge_prompt(question, ground_truth, model_answer, task_type, context)

    raw = None
    for attempt in range(2):  # initial call + one retry on malformed JSON
        raw = call_fn(prompt, response_schema=JUDGE_RESPONSE_SCHEMA)
        if raw is None:
            return JudgeResult(
                decision="JUDGE_FAILED", reasoning="", raw_response=None,
                error="judge_api_failed",
            )
        parsed = parse_judge_response(raw)
        if parsed is not None:
            decision, reasoning, confidence = parsed
            return JudgeResult(decision=decision, reasoning=reasoning,
                               raw_response=raw, confidence=confidence)
        logger.warning(
            "Judge returned malformed JSON (attempt %d/2): %.200s", attempt + 1, raw
        )

    return JudgeResult(
        decision="JUDGE_FAILED", reasoning="", raw_response=raw,
        error="judge_malformed_json",
    )


def extract_event_order(call_fn, shuffled_events: list[str], model_answer: str) -> list[int] | None:
    """
    Judge-assisted parsing fallback for event sorting: given the shuffled event
    list and the model's raw answer, extract the chronological ordering the
    answer intends, as 0-based indices into `shuffled_events`.

    Returns a valid permutation of range(len(shuffled_events)), or None if the
    ordering cannot be recovered.
    """
    n = len(shuffled_events)
    events_block = "\n".join(f"Event {i}: {e}" for i, e in enumerate(shuffled_events))
    prompt = (
        "You are a strict parsing assistant. A model was asked to arrange the "
        "numbered events below in chronological order (earliest first). Your job "
        "is ONLY to extract the ordering that the model's answer expresses — do "
        "NOT judge whether the ordering is chronologically right, and do NOT "
        "produce your own ordering.\n\n"
        f"Events (numbered):\n{events_block}\n\n"
        f"Model's answer:\n{model_answer}\n\n"
        "The answer may reference events by their numbers or by their text; map "
        "text references back to the corresponding event numbers.\n"
        "Return strict JSON only:\n"
        '{"reasoning": "<how you mapped the answer to event numbers>", '
        '"ordered_indices": [<each event number exactly once, in the order the '
        "model's answer intends>]}\n"
        'If the answer does not clearly express an ordering of these events, '
        'return {"reasoning": "...", "ordered_indices": []}.'
    )

    raw = call_fn(prompt, response_schema=EXTRACT_ORDER_SCHEMA)
    if raw is None:
        return None

    cleaned = strip_json_fences(raw)
    data = _loads_or_none(cleaned)
    if not isinstance(data, dict):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            data = _loads_or_none(cleaned[start : end + 1])
    if not isinstance(data, dict):
        logger.warning("extract_event_order: malformed JSON: %.200s", raw)
        return None

    indices = data.get("ordered_indices")
    if (
        isinstance(indices, list)
        and len(indices) == n
        and all(isinstance(i, int) for i in indices)
        and sorted(indices) == list(range(n))
    ):
        return indices

    logger.warning("extract_event_order: not a valid permutation: %s", indices)
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Rubric v2: atomic judge (judge_atoms)
#
# The judge answers a fixed set of yes/no questions as strict JSON and NEVER
# produces a numeric score; every score is computed deterministically in
# rubric_scoring.py. GT dates are extracted in Python and handed TO the judge
# as a canonical list — the judge only decides whether the model answer
# commits to each of them.
# ════════════════════════════════════════════════════════════════════════════════

ATOM_SCHEMA_BASE = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "date_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gt_index": {"type": "integer"},
                    "gt_date": {"type": "string"},
                    "matched": {"type": "boolean"},
                    "model_expression": {"type": "string"},
                },
                "required": ["gt_index", "gt_date", "matched", "model_expression"],
                "propertyOrdering": ["gt_index", "gt_date", "matched",
                                     "model_expression"],
            },
        },
        "contradicting_dates_in_answer": {"type": "boolean"},
        "meaning_correct": {"type": "boolean"},
        "meaning_rationale": {"type": "string"},
        "uncertain": {"type": "boolean"},
        # Self-reported certainty; logged and averaged per run, never scored.
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["reasoning", "date_matches", "contradicting_dates_in_answer",
                 "meaning_correct", "meaning_rationale", "uncertain",
                 "confidence"],
    "propertyOrdering": ["reasoning", "date_matches",
                         "contradicting_dates_in_answer", "meaning_correct",
                         "meaning_rationale", "uncertain", "confidence"],
}

# Extra boolean/string atoms per task type, merged into the base schema.
ATOM_SCHEMA_EXTENSIONS: dict[str, dict[str, dict]] = {
    "perturbation_explanation": {
        "perturbation_flagged": {"type": "boolean"},
        "perturbation_localized": {"type": "boolean"},
        "localized_element": {"type": "string"},
        "correction_given": {"type": "boolean"},
        "correction_correct": {"type": "boolean"},
    },
    "context_misalignment": {
        "followed_context": {"type": "boolean"},
        "followed_parametric": {"type": "boolean"},
        "conflict_acknowledged": {"type": "boolean"},
        "blended_sources": {"type": "boolean"},
    },
}

# Extension booleans whose absence makes a response malformed (repair retry).
ATOM_REQUIRED_EXTRAS: dict[str, tuple[str, ...]] = {
    "perturbation_explanation": ("perturbation_flagged", "perturbation_localized",
                                 "correction_given", "correction_correct"),
    "context_misalignment": ("followed_context", "followed_parametric",
                             "conflict_acknowledged", "blended_sources"),
}


def _atom_schema(task_type: str) -> dict:
    """Base atom schema, extended with the task type's extra fields."""
    ext = ATOM_SCHEMA_EXTENSIONS.get(task_type)
    if not ext:
        return ATOM_SCHEMA_BASE
    schema = json.loads(json.dumps(ATOM_SCHEMA_BASE))  # deep copy
    schema["properties"].update(ext)
    schema["required"].extend(ext)
    schema["propertyOrdering"].extend(ext)
    return schema


# Embedded VERBATIM in every atom prompt (rubric v2 meaning checklist).
MEANING_CHECKLIST = """\
meaning_correct is TRUE only if ALL of the following hold, judged against the
ground-truth answer; otherwise FALSE. Binary — no partial credit.
- The model answer entails the ground truth's key facts and contradicts none
  of them.
- Correct legal instrument / entity / section (the right SRO, Act, or clause).
- Correct action AND direction of change — increased vs decreased, inserted vs
  omitted, effective vs repealed. Direction errors are automatically FALSE: in
  amendment QA, "raised to 15%" vs "lowered to 15%" is the difference between
  correct and catastrophically wrong.
- Correct values (rates, amounts, thresholds) when the ground truth states them.
- Paraphrase and language switching (Bangla <-> English) are acceptable; extra
  correct detail is acceptable.
Judge meaning_correct on the NON-DATE substance of the answer; date agreement
is captured separately by date_matches and must not double-count here."""

ATOM_PROMPT_HEADER = """\
You are a grading assistant for a question-answering benchmark on Bangladesh
customs regulations (SROs — Statutory Regulatory Orders, duty rules, baggage
rules, and trade agreements).

You do NOT produce a grade, score, or verdict. You answer a fixed set of
atomic yes/no questions about the MODEL ANSWER; a program computes the score
from your booleans.

Rules:
1. The GROUND-TRUTH ANSWER is authoritative and correct BY DEFINITION. Do not
   use your own knowledge or opinion to second-guess it.
2. The ground-truth dates listed below were extracted programmatically from
   the ground-truth answer. Decide ONLY whether the model answer commits to
   each of them; never introduce ground-truth dates of your own.
3. For each listed ground-truth date: matched = true iff the MODEL ANSWER
   commits to that date — in any language, script, or format (Bengali
   numerals, Bangla month names, "9 June 2011", "09/06/2011", spelled-out
   ordinals, or a relative expression that resolves to it) — supplying at
   least the required components with matching values. Copy the exact span of
   the model answer that expresses the date into model_expression (empty
   string if not matched).
4. contradicting_dates_in_answer = true iff the model answer asserts a
   DIFFERENT date, at the required granularity, for a fact the ground truth
   dates — or gives multiple conflicting dates for the same fact without
   committing to one. Extra dates about other facts that do not conflict with
   the ground truth are NOT contradictions.
5. {meaning_checklist}
6. If a judgment is genuinely borderline under the checklist, still commit to
   a boolean but set uncertain = true and say why in meaning_rationale (such
   items are routed to human review; they are never soft-scored)."""

ATOM_PROMPT_FOOTER = """\
First think through the case step by step in the "reasoning" field, then fill
every other field. Finally rate your "confidence" that your judgments are
right, as an integer from 0 to 100 (90-100 = clear-cut case, 70-89 = fairly
confident, 50-69 = borderline judgment call, below 50 = essentially guessing);
confidence is recorded for analysis and does not affect scoring.
Respond with strict JSON only, exactly this shape:
{"reasoning": "<step-by-step analysis>", "date_matches": [{"gt_index": <int>, "gt_date": "<canonical date being judged>", "matched": true/false, "model_expression": "<verbatim span from the model answer, or empty>"}], "contradicting_dates_in_answer": true/false, "meaning_correct": true/false, "meaning_rationale": "<1-2 sentences>", "uncertain": true/false, "confidence": <integer 0-100>}
Include one date_matches entry per listed ground-truth date (an empty list if
none are listed), plus any additional fields required by the task note above."""

ATOM_REPAIR_MESSAGE = """\

Your previous response was not valid JSON matching the required schema.
Previous response (truncated): {previous}
Return ONLY valid JSON matching the schema — no prose outside the JSON object."""

_GRANULARITY_LABELS = {
    "year": ("the YEAR only — a different month or day than the ground truth "
             "is NEITHER a mismatch NOR a contradiction"),
    "month": ("the YEAR and MONTH — a different day than the ground truth is "
              "NEITHER a mismatch NOR a contradiction"),
    "day": "the full date: YEAR, MONTH and DAY",
}


def _describe_gt_dates(gt_dates: list["rubric_scoring.CanonicalDate"],
                       required_granularity: str | None) -> str:
    """The canonical GT date list handed to the judge, one line per date with
    its required match granularity spelled out."""
    if not gt_dates:
        return ("Ground-truth dates: none were extracted from the ground-truth "
                "answer. Return an empty date_matches list.")
    rank = {"year": 1, "month": 2, "day": 3}
    lines = ["Ground-truth dates (canonical, extracted programmatically from "
             "the ground-truth answer):"]
    for i, d in enumerate(gt_dates):
        effective = d.granularity
        if required_granularity is not None and \
                rank[required_granularity] < rank[effective]:
            effective = required_granularity
        line = (f"  [{i}] {d.canonical()} — as written in the ground truth: "
                f"\"{d.source_span}\" — required match: "
                f"{_GRANULARITY_LABELS[effective]}")
        if d.calendar == "bangla":
            line += (" [Bangla-calendar date — decide the match semantically; "
                     "a program could not compare calendars]")
        if d.range_role:
            line += f" [{d.range_role} of a date range]"
        lines.append(line)
    return "\n".join(lines)


def _atom_task_note(task_type: str, anchor: str | None,
                    extra_gt: dict | None) -> str | None:
    extra_gt = extra_gt or {}
    if task_type == "context_misalignment":
        note = (
            "Task note (context_misalignment): the context shown to the model "
            "deliberately contains misaligned or incorrect information; the "
            "ground-truth answer remains the authority. Additionally report "
            "these booleans about the model's behavior (they are logged, not "
            "scored):\n"
            "- followed_context: the answer adopts the misaligned context's "
            "wrong value/claim.\n"
            "- followed_parametric: the answer gives the authoritative "
            "(ground-truth) value.\n"
            "- conflict_acknowledged: the answer explicitly notes a "
            "discrepancy between the given context and the actual regulation.\n"
            "- blended_sources: the answer mixes the misaligned and "
            "authoritative values.")
        if extra_gt.get("conflict"):
            note += f"\nDocumented conflict for this item: {extra_gt['conflict']}"
        return note
    if task_type == "perturbation_explanation":
        note = (
            "Task note (perturbation explanation): the statement below "
            "contains a deliberately perturbed element, and the model was "
            "asked to explain WHY and WHERE the statement is wrong. The "
            "specific perturbed element is:\n"
            f"- perturbed (wrong) value in the statement: "
            f"{extra_gt.get('perturbed_fact')}\n"
            f"- correct value per the regulation: {extra_gt.get('correct_fact')}\n"
            "Additionally report:\n"
            "- perturbation_flagged: the explanation asserts the statement is "
            "wrong.\n"
            "- perturbation_localized: the explanation identifies THAT "
            "SPECIFIC element above as the error. Saying something is off "
            "while naming a DIFFERENT element is false.\n"
            "- localized_element: the element the model identified (empty "
            "string if none).\n"
            "- correction_given: the model states what the correct fact "
            "should be.\n"
            "- correction_correct: the model's stated correction matches the "
            "correct value above.")
        return note
    if task_type == "relative_time_qa":
        if anchor:
            return (
                "Task note (relative-time QA): the question situates the "
                f"target relative to an anchor. Anchor: {anchor}.\n"
                "The model may express the answer date absolutely or relative "
                "to the anchor; a relative expression counts as committing to "
                "a date only if it resolves to the listed ground-truth date "
                "at its required granularity.")
        return (
            "Task note (relative-time QA): no anchor date is available for "
            "this item. Decide each date match by semantic equivalence "
            "between the model answer and the listed ground-truth date "
            "(booleans only).")
    return None


def build_atom_prompt(
    question: str,
    ground_truth: str,
    model_answer: str,
    task_type: str,
    gt_dates: list["rubric_scoring.CanonicalDate"],
    context: str | None = None,
    anchor: str | None = None,
    required_granularity: str | None = None,
    extra_gt: dict | None = None,
) -> str:
    """Assemble the full atomic-judge prompt for one instance."""
    parts = [ATOM_PROMPT_HEADER.format(meaning_checklist=MEANING_CHECKLIST)]

    note = _atom_task_note(task_type, anchor, extra_gt)
    if note:
        parts.append(note)

    parts.append("Now analyze the following case.")
    if context:
        if task_type == "perturbation_explanation":
            label = ("Statement under evaluation (the model judged it wrong "
                     "and explained why; analyze that explanation against the "
                     "ground truth):")
        else:
            label = ("Context/scenario that was shown to the model (for "
                     "reference only — the ground truth below remains the "
                     "authority):")
        parts.append(f"{label}\n{context}")
    parts.append(f"Question: {question}")
    parts.append(f"Ground-truth answer: {ground_truth}")
    parts.append(_describe_gt_dates(gt_dates, required_granularity))
    parts.append(f"Model answer: {model_answer}")
    parts.append(ATOM_PROMPT_FOOTER)
    return "\n\n".join(parts)


# Contract every judge's call_fn must satisfy:
# call_fn(prompt, response_schema=None) -> str | None (None = judge unavailable).
CallFn = Callable[..., "str | None"]


@dataclass
class AtomJudgeResult:
    """Outcome of one council evaluation (majority-voted over the judges)."""

    atoms: "rubric_scoring.JudgeAtoms | None"   # council-voted atoms; None on error
    raw_response: str | None    # first surviving judge's raw response
    error: str | None = None    # "judge_api_failed" | "judge_malformed_json"
    n_votes: int = 1            # judges polled
    repair_retry_used: bool = False   # any judge used its repair retry
    # Per-judge attribution, keyed by judge_name:
    raw_responses: dict[str, str | None] = field(default_factory=dict)
    judge_errors: dict[str, str] = field(default_factory=dict)  # failed judges only
    per_judge_atoms: dict[str, dict] = field(default_factory=dict)  # surviving judges


def _parse_atoms_or_none(raw: str, n_gt_dates: int,
                         required_extras: tuple[str, ...]):
    cleaned = strip_json_fences(raw)
    data = _loads_or_none(cleaned)
    if data is None:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            data = _loads_or_none(cleaned[start:end + 1])
    if data is None:
        return None
    try:
        return rubric_scoring.JudgeAtoms.from_json(
            data, n_gt_dates, required_extras=required_extras)
    except ValueError as e:
        logger.warning("Judge atoms failed validation: %s", e)
        return None


def _judge_atoms_once(call_fn, prompt: str, schema: dict, n_gt_dates: int,
                      required_extras: tuple[str, ...]) -> AtomJudgeResult:
    """One judge call with ONE repair retry on malformed/invalid JSON."""
    raw = call_fn(prompt, response_schema=schema)
    if raw is None:
        return AtomJudgeResult(atoms=None, raw_response=None,
                               error="judge_api_failed")
    atoms = _parse_atoms_or_none(raw, n_gt_dates, required_extras)
    if atoms is not None:
        return AtomJudgeResult(atoms=atoms, raw_response=raw)

    logger.warning("Atom judge returned malformed JSON, sending repair "
                   "retry: %.200s", raw)
    repair_prompt = prompt + ATOM_REPAIR_MESSAGE.format(previous=raw[:500])
    raw2 = call_fn(repair_prompt, response_schema=schema)
    if raw2 is None:
        return AtomJudgeResult(atoms=None, raw_response=raw,
                               error="judge_api_failed", repair_retry_used=True)
    atoms = _parse_atoms_or_none(raw2, n_gt_dates, required_extras)
    if atoms is not None:
        return AtomJudgeResult(atoms=atoms, raw_response=raw2,
                               repair_retry_used=True)
    return AtomJudgeResult(atoms=None, raw_response=raw2,
                           error="judge_malformed_json", repair_retry_used=True)


def _majority(values: list[bool]) -> bool:
    return sum(values) * 2 > len(values)


def _atoms_summary(atoms: "rubric_scoring.JudgeAtoms") -> dict:
    """JSON-serializable snapshot of one judge's atoms, for per-judge
    attribution in the results file (disagreement/calibration analysis)."""
    return {
        "date_matched": list(atoms.date_matched),
        "contradicting_dates_in_answer": atoms.contradicting_dates_in_answer,
        "meaning_correct": atoms.meaning_correct,
        "uncertain": atoms.uncertain,
        "confidence": atoms.confidence,
        "extras": dict(atoms.extras),
    }


def _vote(named: list[tuple[str, AtomJudgeResult]],
          n_gt_dates: int) -> AtomJudgeResult:
    """Majority vote per boolean atom over the judges that succeeded; strings
    come from the first surviving judge (council order is fixed, so this is
    deterministic). With an even number of survivors a tie is NOT a majority
    (sum*2 > len), so a 1-1 split resolves to False — conservative.
    vote_splits records true-votes/total per atom; per-judge booleans are
    preserved in per_judge_atoms."""
    ok = [(name, r) for name, r in named if r.atoms is not None]
    first = ok[0][1].atoms
    n = len(ok)
    splits: dict[str, str] = {}

    def voted(key: str, values: list[bool]) -> bool:
        splits[key] = f"{sum(values)}/{n}"
        return _majority(values)

    date_matched = [
        voted(f"date_matches.{i}.matched",
              [r.atoms.date_matched[i] for _, r in ok])
        for i in range(n_gt_dates)
    ]
    extras = dict(first.extras)
    for key, val in first.extras.items():
        if isinstance(val, bool):
            extras[key] = voted(
                f"extras.{key}",
                [bool(r.atoms.extras.get(key)) for _, r in ok])

    # Confidence: mean over the surviving judges (logged-only); the individual
    # per-judge values are kept in vote_splits for calibration analysis.
    confidences = [r.atoms.confidence for _, r in ok
                   if r.atoms.confidence is not None]
    if confidences:
        splits["confidence_votes"] = ",".join(f"{c:g}" for c in confidences)

    atoms = rubric_scoring.JudgeAtoms(
        date_matched=date_matched,
        model_expressions=list(first.model_expressions),
        contradicting_dates_in_answer=voted(
            "contradicting_dates_in_answer",
            [r.atoms.contradicting_dates_in_answer for _, r in ok]),
        meaning_correct=voted("meaning_correct",
                              [r.atoms.meaning_correct for _, r in ok]),
        meaning_rationale=first.meaning_rationale,
        uncertain=voted("uncertain", [r.atoms.uncertain for _, r in ok]),
        confidence=(round(sum(confidences) / len(confidences), 1)
                    if confidences else None),
        extras=extras,
        vote_splits=splits,
    )
    return AtomJudgeResult(
        atoms=atoms, raw_response=ok[0][1].raw_response, n_votes=len(named),
        repair_retry_used=any(r.repair_retry_used for _, r in named),
        raw_responses={name: r.raw_response for name, r in named},
        judge_errors={name: r.error for name, r in named
                      if r.atoms is None and r.error},
        per_judge_atoms={name: _atoms_summary(r.atoms) for name, r in ok},
    )


def judge_atoms(
    judges: list[tuple[str, CallFn]],
    *,
    question: str,
    ground_truth: str,
    model_answer: str,
    task_type: str,
    gt_dates: list["rubric_scoring.CanonicalDate"],
    context: str | None = None,
    anchor: str | None = None,
    required_granularity: str | None = None,
    extra_gt: dict | None = None,
) -> AtomJudgeResult:
    """Rubric-v2 council call: atomic booleans in, no numeric score out.

    `judges` is the ordered council — (judge_name, call_fn) pairs, every
    judge receiving the identical prompt and voting independently.
    `gt_dates` is the deterministic canonical GT date list (rubric_scoring.
    prepare_gt); each judge only decides whether the model answer commits to
    each. One repair retry per judge on malformed JSON; boolean atoms are
    combined by per-atom strict majority over the judges that succeeded.
    Quorum is a strict majority of the council (2 of 3): a single judge
    failure degrades to voting over the survivors; below quorum the instance
    is judge-failed — "judge_api_failed" (retried on --resume) if any judge
    failure was an API outage, else "judge_malformed_json".
    """
    prompt = build_atom_prompt(question, ground_truth, model_answer, task_type,
                               gt_dates, context=context, anchor=anchor,
                               required_granularity=required_granularity,
                               extra_gt=extra_gt)
    schema = _atom_schema(task_type)
    required_extras = ATOM_REQUIRED_EXTRAS.get(task_type, ())
    n = len(gt_dates)

    named = [(name, _judge_atoms_once(fn, prompt, schema, n, required_extras))
             for name, fn in judges]

    ok = [(name, r) for name, r in named if r.atoms is not None]
    quorum = len(judges) // 2 + 1
    if len(ok) < quorum:
        failed = [(name, r) for name, r in named if r.atoms is None]
        error = ("judge_api_failed"
                 if any(r.error == "judge_api_failed" for _, r in failed)
                 else "judge_malformed_json")
        logger.warning("Judge council below quorum: only %d/%d judges "
                       "succeeded (%s)", len(ok), len(judges),
                       ", ".join(f"{name}: {r.error}" for name, r in failed))
        return AtomJudgeResult(
            atoms=None, raw_response=failed[0][1].raw_response, error=error,
            n_votes=len(judges),
            repair_retry_used=any(r.repair_retry_used for _, r in named),
            raw_responses={name: r.raw_response for name, r in named},
            judge_errors={name: r.error for name, r in failed if r.error},
        )
    return _vote(named, n)
