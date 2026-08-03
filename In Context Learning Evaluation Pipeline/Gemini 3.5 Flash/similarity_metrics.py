"""
similarity_metrics.py — post-hoc BLEU + BERTScore over a finished evaluation
run, as a sanity check on the LLM judge.

Reads <results-dir>/<dataset>/results.json (parametric or grounded — same
schema), computes reference-similarity metrics for every item that PASSED the
hard gates (i.e., whose meaning the judge actually evaluated), writes the
per-item values back into results.json, and appends aggregates to
<dataset>/metrics.json under "similarity_metrics" (plus a compact "similarity"
block in summary.json when present).

Why: the judge's meaning_correct decision should correlate with surface /
semantic similarity to the ground truth — items the judge marks correct should
score higher on BLEU and BERTScore. The aggregates therefore report the means
split by meaning_correct and a point-biserial correlation, so judge quality
can be eyeballed directly. These metrics are DIAGNOSTIC ONLY — they never
touch final_score.

Eligibility (mirrors when the meaning component is live in
rubric_scoring.score_base): status=="ok", not an auto-scored empty answer,
judge atoms present, and n_matched == n_gt_dates (n=0 items are meaning-only
and trivially eligible; n=1 requires the date matched; n>=2 requires all
matched — Decision A's all-or-nothing gate).

Metrics:
  - BLEU: custom implementation in rubric_scoring.bleu_score (no NLTK).
    Primary = cumulative BLEU-2 (see rationale there); BLEU-1..4 stored.
  - BERTScore: the `bert-score` package directly (NOT the HF `evaluate`
    wrapper — evaluate wraps this same package with identical numbers but
    extra dependencies and less config control). Model: xlm-roberta-large,
    the package authors' recommended multilingual model — needed because the
    corpus mixes Bangla and English. rescale_with_baseline=False (baselines
    are per-language; raw scores are consistent within-run, which is all the
    judge-agreement analysis needs). Per item: precision, recall, F1;
    aggregates: average precision / recall / F1 over eligible items.

Usage:
    python similarity_metrics.py --results-dir param_check_results
    python similarity_metrics.py --results-dir context_check_results
    python similarity_metrics.py --results-dir param_check_results --recompute

torch / bert-score are imported lazily inside compute_bertscore() so this
module stays importable (and unit-testable) without them. First run downloads
xlm-roberta-large (~2.3 GB) into the HuggingFace cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from scipy.stats import pointbiserialr

import rubric_scoring

logger = logging.getLogger("similarity_metrics")

# The 3 datasets this analysis targets (free-text, judge-scored, no anchor
# machinery). Extendable via --datasets.
DEFAULT_DATASETS = ("open_ended", "scenario_short_answer",
                    "temporal_scenario_short_answer")
DEFAULT_MODEL_TYPE = "xlm-roberta-large"
DEFAULT_BATCH_SIZE = 32


def write_json_atomic(path: Path, payload: dict) -> None:
    """Local copy of the drivers' atomic writer — this script deliberately
    avoids importing PK (and with it the google SDK)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════════
# Eligibility (the hard-gate filter)
# ════════════════════════════════════════════════════════════════════════════════

def similarity_eligible(rec: dict) -> bool:
    """True iff the item's meaning was actually evaluated by the judge —
    i.e., it passed every hard gate. Only then is comparing the model answer
    to the ground truth a meaningful judge-sanity signal."""
    if rec.get("status") != "ok":
        return False
    if rec.get("answered_empty"):
        return False
    if rec.get("judge_atoms") is None:
        return False
    n_gt = rec.get("n_gt_dates")
    n_matched = rec.get("n_matched")
    if not isinstance(n_gt, int) or not isinstance(n_matched, int):
        return False
    return n_matched == n_gt


# ════════════════════════════════════════════════════════════════════════════════
# BERTScore (lazy torch import)
# ════════════════════════════════════════════════════════════════════════════════

def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def compute_bertscore(candidates: list[str], references: list[str],
                      model_type: str, batch_size: int, device: str) -> dict:
    """One batched bert-score call. Returns per-pair lists plus the config
    actually used (recorded in metrics for reproducibility)."""
    import bert_score
    t0 = time.time()
    P, R, F = bert_score.score(
        cands=candidates, refs=references,
        model_type=model_type, batch_size=batch_size, device=device,
        rescale_with_baseline=False, verbose=False,
    )
    try:
        from bert_score.utils import model2layers
        num_layers = model2layers.get(model_type)
    except ImportError:  # internal layout differs across bert-score versions
        num_layers = None
    logger.info("BERTScore: %d pair(s) scored in %.1fs on %s",
                len(candidates), time.time() - t0, device)
    return {
        "precision": [round(float(x), 4) for x in P],
        "recall": [round(float(x), 4) for x in R],
        "f1": [round(float(x), 4) for x in F],
        "config": {
            "bertscore_model": model_type,
            "num_layers": num_layers,
            "rescale_with_baseline": False,
            "bert_score_version": bert_score.__version__,
            "device": device,
            "batch_size": batch_size,
        },
    }


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation
# ════════════════════════════════════════════════════════════════════════════════

def _mean(vals: list) -> float | None:
    nums = [float(v) for v in vals
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(nums) / len(nums), 4) if nums else None


def _metric_means(recs: list[dict]) -> dict:
    return {
        "n": len(recs),
        "bleu_mean": _mean([r.get("bleu") for r in recs]),
        "bertscore_precision_mean": _mean([r.get("bertscore_precision") for r in recs]),
        "bertscore_recall_mean": _mean([r.get("bertscore_recall") for r in recs]),
        "bertscore_f1_mean": _mean([r.get("bertscore_f1") for r in recs]),
    }


def _judge_agreement(eligible: list[dict]) -> dict:
    """Point-biserial correlation between the judge's binary meaning_correct
    and each similarity metric — the direct test of 'higher judge alignment
    should mean higher BLEU/BERTScore'. Needs both classes present."""
    out: dict = {}
    labels = [1 if r.get("meaning_correct") else 0 for r in eligible]
    if len(set(labels)) < 2 or len(labels) < 3:
        out["note"] = ("correlation not computed: needs >=3 eligible items "
                       "with both meaning_correct outcomes present")
        return out
    for key in ("bleu", "bertscore_f1"):
        vals = [r.get(key) for r in eligible]
        if any(not isinstance(v, (int, float)) for v in vals):
            continue
        r_pb, p = pointbiserialr(labels, [float(v) for v in vals])
        out[key] = {"point_biserial_r": round(float(r_pb), 4),
                    "p_value": round(float(p), 6)}
    out["note"] = ("point-biserial correlation of meaning_correct (0/1) vs "
                   "the metric over eligible items; positive r supports the "
                   "judge (judge-correct items are more similar to the GT)")
    return out


def build_aggregates(records: list[dict], bert_config: dict | None) -> dict:
    eligible = [r for r in records if r.get("similarity_eligible")]
    scored_ok = [r for r in records
                 if r.get("status") == "ok" and r.get("judge_atoms") is not None
                 and not r.get("answered_empty")]
    agg = {
        "n_records": len(records),
        "n_eligible": len(eligible),
        "n_skipped_hard_gate": len(scored_ok) - len(eligible),
        "n_skipped_unscored": len(records) - len(scored_ok),
        **{k: v for k, v in _metric_means(eligible).items() if k != "n"},
        "by_meaning_correct": {
            "true": _metric_means([r for r in eligible if r.get("meaning_correct")]),
            "false": _metric_means([r for r in eligible if not r.get("meaning_correct")]),
        },
        "judge_agreement": _judge_agreement(eligible),
        "eligibility_rule": "status==ok AND not answered_empty AND judge_atoms "
                            "present AND n_matched == n_gt_dates (meaning "
                            "component live in score_base)",
        "config": {
            "bleu_variant": f"custom cumulative BLEU-{rubric_scoring.BLEU_PRIMARY_N}, "
                            "epsilon-smoothed, brevity penalty, Bengali-aware "
                            "tokenizer (rubric_scoring.bleu_score)",
            **(bert_config or {}),
        },
        "note": "Diagnostic judge-sanity metrics over hard-gate-passing items "
                "only; never part of final_score.",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return agg


# ════════════════════════════════════════════════════════════════════════════════
# Per-dataset processing
# ════════════════════════════════════════════════════════════════════════════════

def process_dataset(results_dir: Path, dataset: str, model_type: str,
                    batch_size: int, device: str, recompute: bool) -> dict | None:
    results_path = results_dir / dataset / "results.json"
    if not results_path.is_file():
        logger.warning("[%s] no results at %s — skipped", dataset, results_path)
        return None
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    records = payload.get("results", [])
    if not records:
        logger.warning("[%s] empty results — skipped", dataset)
        return None

    # Eligibility + BLEU (cheap, always recomputed so rule changes propagate).
    for r in records:
        r["similarity_eligible"] = similarity_eligible(r)
        if r["similarity_eligible"]:
            b = rubric_scoring.bleu_score(r["ground_truth_answer"],
                                          r.get("model_answer") or "")
            r["bleu"] = b["bleu"]
            r["bleu_by_n"] = b["bleu_by_n"]
            r["bleu_brevity_penalty"] = b["brevity_penalty"]
        else:
            for k in ("bleu", "bleu_by_n", "bleu_brevity_penalty",
                      "bertscore_precision", "bertscore_recall", "bertscore_f1"):
                r.pop(k, None)

    # BERTScore: one batched call over the pairs that still need it.
    pending = [r for r in records if r["similarity_eligible"]
               and (recompute or not isinstance(r.get("bertscore_f1"), float))]
    bert_config = None
    if pending:
        result = compute_bertscore(
            candidates=[r.get("model_answer") or "" for r in pending],
            references=[r["ground_truth_answer"] for r in pending],
            model_type=model_type, batch_size=batch_size, device=device,
        )
        bert_config = result["config"]
        for r, p, rc, f in zip(pending, result["precision"],
                               result["recall"], result["f1"]):
            r["bertscore_precision"] = p
            r["bertscore_recall"] = rc
            r["bertscore_f1"] = f
    else:
        logger.info("[%s] BERTScore already present on all eligible items "
                    "(use --recompute to redo)", dataset)

    write_json_atomic(results_path, payload)

    agg = build_aggregates(records, bert_config)
    metrics_path = results_dir / dataset / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["similarity_metrics"] = agg
        write_json_atomic(metrics_path, metrics)
    else:
        logger.warning("[%s] no metrics.json — aggregates not persisted there",
                       dataset)
    logger.info("[%s] eligible=%d/%d | bleu_mean=%s | bertscore P/R/F1 = %s/%s/%s",
                dataset, agg["n_eligible"], agg["n_records"], agg["bleu_mean"],
                agg["bertscore_precision_mean"], agg["bertscore_recall_mean"],
                agg["bertscore_f1_mean"])
    return agg


def update_summary(results_dir: Path, per_dataset: dict[str, dict]) -> None:
    path = results_dir / "summary.json"
    if not path.is_file() or not per_dataset:
        return
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not augment %s: %s", path, e)
        return
    summary["similarity"] = {
        name: {k: agg.get(k) for k in
               ("n_eligible", "bleu_mean", "bertscore_precision_mean",
                "bertscore_recall_mean", "bertscore_f1_mean")}
        for name, agg in per_dataset.items()
    }
    write_json_atomic(path, summary)
    logger.info("summary.json augmented with similarity block")


# ════════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc BLEU + BERTScore (judge-sanity metrics) over a "
                    "finished evaluation run.")
    parser.add_argument("--results-dir", required=True,
                        help="Finished run to augment (e.g. param_check_results "
                             "or context_check_results).")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        metavar="NAME",
                        help=f"Datasets to process (default: "
                             f"{' '.join(DEFAULT_DATASETS)}).")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE,
                        help=f"BERTScore model (default: {DEFAULT_MODEL_TYPE}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"BERTScore batch size (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cpu", "cuda"],
                        help="Torch device (default: auto = mps if available).")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute BERTScore even where already present.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        logger.error("Results dir not found: %s", results_dir)
        sys.exit(1)
    device = resolve_device(args.device)
    logger.info("Similarity metrics | results=%s | datasets=%s | model=%s | "
                "device=%s", results_dir, ",".join(args.datasets),
                args.model_type, device)
    per_dataset: dict[str, dict] = {}
    for dataset in args.datasets:
        agg = process_dataset(results_dir, dataset, args.model_type,
                              args.batch_size, device, args.recompute)
        if agg:
            per_dataset[dataset] = agg
    update_summary(results_dir, per_dataset)


if __name__ == "__main__":
    main()
