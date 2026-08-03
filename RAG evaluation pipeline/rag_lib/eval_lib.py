"""eval_lib.py — provider-agnostic evaluation machinery for the RAG check.

Self-contained copy of the ICSETEP benchmark's prompts, response parsers,
result store, rubric-v2 metrics, and document-context assembly — lifted from the
teammate's PK_check_gemini.py (evaluation logic) and context_check_gemini.py
(context assembly) so the new RAG driver depends on NEITHER of those files (both
are deleted). The scoring metric is therefore identical to the teammate's run;
only the source of the context (RAG retrieval) and the answering provider
(OpenRouter) differ.

Imports only stdlib + the shared rubric_scoring module +
scipy/sklearn/tqdm. No argparse — configuration lives in the driver's globals.
"""
from __future__ import annotations

import csv
import difflib
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from scipy.stats import kendalltau
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# rubric_scoring is shared, one dir up (OpenRouter/).
for _p in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rubric_scoring

logger = logging.getLogger("rag_eval")

# Strip markdown code fences (```json ... ```) from a model/judge response.
# (Formerly llm_judge.strip_json_fences; inlined so the runtime lib carries no
# dependency on the retired single-judge module.)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from a response."""
    return _FENCE_RE.sub("", text.strip()).strip()

# ════════════════════════════════════════════════════════════════════════════
# Configuration constants (copied from PK_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL = "gemini-3.1-flash-lite"   # default judge model (driver overrides)
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 12.0
CALLS_PER_DAY = 500
RESULTS_DIR = "rag_check_results"

# eval_lib.py lives in OpenRouter/rag_lib/; the datasets are one dir up.
SCRIPT_DIR = Path(__file__).resolve().parent
QA_DIR = SCRIPT_DIR.parent / "concept_qa_by_type"

# Fixed decoding budget for judge calls (the atom JSON is small; 4096 leaves
# room for the reasoning field).
JUDGE_MAX_OUTPUT_TOKENS = 4096

# Judge decoding parameters recorded in every metrics.json/summary.json. The
# driver overwrites judge_council / judge_quorum at startup (the judge is the
# 3-model LLM council, not a single self-consistency-voted model).
JUDGE_RUN_CONFIG = {
    "judge_backend": "council",
    "judge_council": [],       # driver fills in the 3 seat model names
    "judge_quorum": 2,         # 2-of-3 surviving-judge quorum
    "judge_max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    "temperature": 0.0,
}


def make_judge_call_fn(caller):
    """Wrap a caller as a call_fn for llm_council.judge_atoms: same
    caller/rotation as answering calls, but with the fixed judge decoding budget
    (JUDGE_MAX_OUTPUT_TOKENS)."""
    def judge_call(prompt: str, response_schema: dict | None = None) -> str | None:
        return caller.call(prompt, response_schema=response_schema,
                           max_output_tokens=JUDGE_RUN_CONFIG["judge_max_output_tokens"])
    return judge_call


# ════════════════════════════════════════════════════════════════════════════
# Prompt templates (copied from PK_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

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

# Structured-output schemas (plain dicts)
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


# ════════════════════════════════════════════════════════════════════════════
# Document-context assembly (adapted from context_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

# Largest document block sent per request. Chosen from the teammate's evidence:
# 342k chars was deliverable, 642k rejected by every key. With chunk-level RAG
# and top_k=10 this cap is rarely reached, but it guards pathological cases.
MAX_CONTEXT_CHARS_DEFAULT = 400_000

# Derived from the parametric preamble, not retyped: sentence 1 (domain framing)
# stays byte-identical; only the closed-book instruction is replaced.
_PARAMETRIC_FINAL_SENTENCE = ("Answer from your own knowledge; no reference "
                              "documents are provided.")
_GROUNDED_FINAL_SENTENCE = ("Answer using the official source document(s) "
                            "provided above; they are authoritative.")
GROUNDED_PREAMBLE = PARAMETRIC_PREAMBLE.replace(
    _PARAMETRIC_FINAL_SENTENCE, _GROUNDED_FINAL_SENTENCE)
assert GROUNDED_PREAMBLE != PARAMETRIC_PREAMBLE, \
    "PARAMETRIC_PREAMBLE changed — update the grounded derivation"
assert _GROUNDED_FINAL_SENTENCE in GROUNDED_PREAMBLE

CONTEXT_HEADER = ("You are given the following official source document(s). "
                  "Use them to answer.")
EMPTY_TEXT_PLACEHOLDER = "[This document's OCR text is empty.]"

CONDITION_NOTE = (
    "RAG run places BOTH the misaligned context and the retrieved source "
    "document(s) in the prompt; this measures whether the model resolves the "
    "conflict in favor of the retrieved official document "
    "(context_conflict_resolution), NOT parametric misalignment robustness. "
    "Do not conflate with the parametric context_misalignment result.")


def build_context_block(docs) -> str:
    """Delimited document block prepended to every answering prompt."""
    parts = [CONTEXT_HEADER, ""]
    for i, d in enumerate(docs, 1):
        parts.append(f"--- DOCUMENT {i}: {d.stem} ---")
        parts.append(d.text if d.text else EMPTY_TEXT_PLACEHOLDER)
        parts.append("")
    return "\n".join(parts) + "\n"


def build_capped_context(docs, max_chars: int) -> tuple[str | None, list[str]]:
    """Doc-boundary context cap: keep each document (first-occurrence order)
    whose inclusion keeps the block within max_chars; drop the rest whole —
    never cut inside a document. Returns (block or None, dropped stems).
    max_chars <= 0 disables the cap."""
    if not docs:
        return None, []
    if not max_chars or max_chars <= 0:
        return build_context_block(docs), []
    sent = []
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
    Everything below the preamble is byte-identical to the parametric prompt."""
    assert parametric_prompt.startswith(PARAMETRIC_PREAMBLE), \
        "answering prompt does not start with PARAMETRIC_PREAMBLE"
    return context_block + GROUNDED_PREAMBLE + parametric_prompt[len(PARAMETRIC_PREAMBLE):]


def context_fields(res, context_block: str | None, dropped_stems: list[str]) -> dict:
    """Extra per-record fields for the RAG run (additive — the record stays
    compatible with the parametric schema). `res` is a RetrievedContext."""
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
    logger.info("[%s] #%d context: %d doc(s) retrieved, %d sent%s, %d chars "
                "(~%d tokens), status=%s",
                dataset, iid, len(base["resolved_documents"]),
                base["documents_sent"],
                f" ({len(base['dropped_documents'])} dropped by cap)"
                if base["context_truncated"] else "",
                base["context_char_count"], base["context_token_estimate"],
                base["context_status"])


# ════════════════════════════════════════════════════════════════════════════
# Defensive response parsing (copied from PK_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

def _load_json_defensive(raw: str):
    """Parse JSON from a model response, tolerating fences and surrounding prose."""
    cleaned = strip_json_fences(raw)
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
    m = re.fullmatch(r"\s*\[?\s*(-?\d+)\s*\]?\s*", strip_json_fences(raw))
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


# ── Event-sorting parsing / matching ────────────────────────────────────────

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


# ════════════════════════════════════════════════════════════════════════════
# Checkpointed result store (copied from PK_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

class ResultStore:
    """
    Incremental per-dataset result storage: results.json is atomically
    rewritten after EVERY instance, so an interruption never loses progress.
    Instances with a terminal status are skipped on resume; 'api_failed'
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


# ════════════════════════════════════════════════════════════════════════════
# Metrics (copied from PK_check_gemini.py + context_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

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
    legacy = [r for r in non_failed if "rubric_version" not in r]
    if legacy:
        logger.warning(
            "[%s] %d record(s) predate rubric v2 and are EXCLUDED from "
            "aggregates — re-run with a fresh results-dir for a clean v2 run.",
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


# ── RAG-run metrics: parametric metrics + retrieval/context stats ────────────

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
    """Parametric metrics (unchanged, via DATASET_METRICS) plus RAG-run context
    statistics."""
    m = DATASET_METRICS[dataset](records, n_dataset_total, dataset, model)
    recs = list(records.values())
    sizes = [r["context_char_count"] for r in recs
             if isinstance(r.get("context_char_count"), int)]
    m["evaluation_mode"] = "rag_document_context"
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
        m["condition"] = "context_conflict_resolution"
        m["condition_note"] = CONDITION_NOTE
    return m


# ════════════════════════════════════════════════════════════════════════════
# Logging / dataset IO / summary (copied from PK_check_gemini.py)
# ════════════════════════════════════════════════════════════════════════════

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
    file_handler = logging.FileHandler(results_dir / "rag_check.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("httpx", "google_genai", "google_genai.models", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_dataset(name: str) -> list[dict]:
    path = QA_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["qa_pairs"]


def write_summary(results_dir: Path, model: str, caller, key_manager) -> None:
    datasets = {}
    for name in DATASET_METRICS:
        metrics_path = results_dir / name / "metrics.json"
        if metrics_path.exists():
            try:
                datasets[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not include %s in summary: %s", metrics_path, e)
    score_by_dataset: dict[str, dict] = {}
    counters = {"items_scored": 0, "judge_error": 0, "uncertain": 0,
                "calendar_mismatch": 0, "anchor_missing": 0,
                "offset_granularity_unparsed": 0}
    judge_config = None
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
        },
    }
    write_json_atomic(results_dir / "summary.json", summary)
    logger.info("Saved combined summary: %s", results_dir / "summary.json")


def write_grounded_summary(results_dir: Path, model: str, caller, key_manager,
                           retriever, judge_callers=None) -> None:
    """write_summary + the RAG retrieval stats block + the judge-council stats.

    `judge_callers` is the list of judge seat callers (Gemini + the two
    OpenRouter seats); their per-seat api_calls/api_failures/rotations are
    recorded so the council's cost is visible alongside the answering model's.
    """
    write_summary(results_dir, model, caller, key_manager)
    path = results_dir / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not augment %s: %s", path, e)
        return
    summary["evaluation_mode"] = "rag_document_context"
    summary["retrieval"] = retriever.resolution_summary()
    if judge_callers:
        summary["judge_backend"] = "council"
        summary["judge_council"] = [c.model for c in judge_callers]
        summary["judge_quorum"] = JUDGE_RUN_CONFIG.get("judge_quorum", 2)
        summary["judge_run_stats"] = {
            c.model: {"api_calls": c.api_calls,
                      "api_error_events": c.api_failures,
                      "key_rotations": getattr(c, "rotation_count", 0)}
            for c in judge_callers
        }
        summary["judge_api_calls_this_run"] = sum(c.api_calls for c in judge_callers)
    write_json_atomic(path, summary)


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
