"""Independently recount the aligned relation-mean encoder comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data" / "reference" / "A1_reference.npz"
DEFAULT_RUNS = ROOT / "experiments" / "runs" / "mean_encoder_p2_20261119"
DEFAULT_RECOUNT = ROOT / "experiments" / "reports" / "mean_encoder_independent_recount.json"
DEFAULT_SUMMARY = ROOT / "experiments" / "reports" / "mean_encoder_summary.json"
ARMS = ("Astar_mean", "B0_mean", "B0_mean_perm")
SEEDS = (0, 1, 42)


def rare_f1(predicted: np.ndarray, truth: np.ndarray, *, ref_limit: float, pred_limit: float) -> float:
    n_spots = truth.size
    truth_counts = Counter(truth.tolist())
    predicted_counts = Counter(predicted.tolist())
    rare_truth = {label for label, count in truth_counts.items() if count / n_spots <= ref_limit}
    rare_predicted = {label for label, count in predicted_counts.items() if count / n_spots <= pred_limit}
    true_mask = np.isin(truth, list(rare_truth))
    predicted_mask = np.isin(predicted, list(rare_predicted))
    return float(f1_score(true_mask, predicted_mask, zero_division=0))


def metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "values_by_seed": values,
    }


def load_run(run_dir: Path, truth: np.ndarray, seed: int, arm: str) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    predicted = np.load(run_dir / "pred.npy", allow_pickle=False).reshape(-1)
    if config.get("arm") != arm or config.get("seed") != seed:
        raise ValueError(f"Run metadata mismatch in {run_dir}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_dir}")
    if predicted.shape != truth.shape:
        raise ValueError(f"Prediction length mismatch in {run_dir}")

    metrics = {
        "ari": float(adjusted_rand_score(truth, predicted)),
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "rare_f1_primary_le_5pct": rare_f1(predicted, truth, ref_limit=0.05, pred_limit=0.05),
        "rare_f1_secondary_lt_4_5pct": _strict_rare_f1(predicted, truth),
    }
    saved_metrics = {
        "ari": float(saved["ari"]),
        "nmi": float(saved["nmi"]),
        "rare_f1_primary_le_5pct": float(saved["rare_primary"]["f1"]),
        "rare_f1_secondary_lt_4_5pct": float(saved["rare_secondary"]["f1"]),
    }
    matches = all(np.isclose(metrics[key], saved_metrics[key], rtol=0.0, atol=1e-12) for key in metrics)
    return {
        "arm": arm,
        "seed": seed,
        "n_clusters": int(np.unique(predicted).size),
        **metrics,
        "saved_evaluation_match": bool(matches),
    }


def _strict_rare_f1(predicted: np.ndarray, truth: np.ndarray) -> float:
    n_spots = truth.size
    truth_counts = Counter(truth.tolist())
    predicted_counts = Counter(predicted.tolist())
    rare_truth = {label for label, count in truth_counts.items() if count / n_spots < 0.05}
    rare_predicted = {label for label, count in predicted_counts.items() if count / n_spots < 0.045}
    true_mask = np.isin(truth, list(rare_truth))
    predicted_mask = np.isin(predicted, list(rare_predicted))
    return float(f1_score(true_mask, predicted_mask, zero_division=0))


def paired_summary(runs: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_by_seed = {run["seed"]: run for run in runs if run["arm"] == left}
    right_by_seed = {run["seed"]: run for run in runs if run["arm"] == right}
    if set(left_by_seed) != set(SEEDS) or set(right_by_seed) != set(SEEDS):
        raise ValueError("Each comparison arm must contain seeds 0, 1 and 42 exactly once")
    metrics = ("ari", "nmi", "rare_f1_primary_le_5pct")
    return {
        metric: metric_summary([right_by_seed[seed][metric] - left_by_seed[seed][metric] for seed in SEEDS])
        for metric in metrics
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-recount", type=Path, default=DEFAULT_RECOUNT)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    truth = np.load(args.reference_npz, allow_pickle=False)["labels"].astype(str)
    runs = [
        load_run(args.runs_dir / f"{arm}_seed{seed}", truth, seed, arm)
        for arm in ARMS
        for seed in SEEDS
    ]
    recount = {
        "comparison": "A1 aligned LightweightHeteroEncoder relation-mean comparison",
        "n_runs": len(runs),
        "all_manifests_complete": True,
        "all_saved_evaluations_match": all(run["saved_evaluation_match"] for run in runs),
        "runs": runs,
    }
    summaries = {}
    for arm in ARMS:
        selected = [run for run in runs if run["arm"] == arm]
        summaries[arm] = {
            metric: metric_summary([run[metric] for run in selected])
            for metric in ("ari", "nmi", "rare_f1_primary_le_5pct")
        }
    summary = {
        "comparison": "A1 aligned LightweightHeteroEncoder relation-mean comparison",
        "n_seeds": len(SEEDS),
        "scope": "Descriptive single-section comparison conditional on supplied labels.",
        "arms": summaries,
        "paired_changes": {
            "B0_mean_minus_Astar_mean": paired_summary(runs, "Astar_mean", "B0_mean"),
            "B0_mean_perm_minus_Astar_mean": paired_summary(runs, "Astar_mean", "B0_mean_perm"),
        },
        "independent_recount_passed": recount["all_saved_evaluations_match"],
    }

    for path, payload in ((args.out_recount, recount), (args.out_summary, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for run in runs:
        print(
            f"{run['arm']:14s} seed={run['seed']:>2} ARI={run['ari']:.4f} "
            f"NMI={run['nmi']:.4f} rare-F1={run['rare_f1_primary_le_5pct']:.4f} "
            f"match={run['saved_evaluation_match']}"
        )
    print(f"recount matched: {recount['all_saved_evaluations_match']}")
    print(f"wrote {args.out_summary}")
    return 0 if recount["all_saved_evaluations_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
