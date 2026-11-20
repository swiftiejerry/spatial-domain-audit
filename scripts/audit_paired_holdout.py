"""Recount whole-section and spatial-holdout scores from saved partitions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference" / "A1_reference.npz"
RUNS = ROOT / "experiments" / "runs" / "p2_aligned_v1"
REPORT = ROOT / "experiments" / "reports" / "paired_holdout_recount.json"


def _rare_masks(
    truth: np.ndarray, prediction: np.ndarray, full_truth: np.ndarray, full_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth_share = {
        str(label): count / len(full_truth)
        for label, count in zip(*np.unique(full_truth.astype(str), return_counts=True))
    }
    prediction_share = {
        int(label): count / len(full_prediction)
        for label, count in zip(*np.unique(full_prediction, return_counts=True))
    }
    rare_truth = {label for label, share in truth_share.items() if share <= 0.05}
    rare_prediction = {label for label, share in prediction_share.items() if share <= 0.05}
    return (
        np.isin(truth.astype(str), list(rare_truth)),
        np.isin(prediction, list(rare_prediction)),
    )


def _score_region(
    truth: np.ndarray,
    prediction: np.ndarray,
    full_truth: np.ndarray,
    full_prediction: np.ndarray,
) -> dict[str, float | int]:
    rare_truth, rare_prediction = _rare_masks(truth, prediction, full_truth, full_prediction)
    return {
        "n_spots": int(len(truth)),
        "ari": float(adjusted_rand_score(truth.astype(str), prediction)),
        "nmi": float(normalized_mutual_info_score(truth.astype(str), prediction)),
        "rare_f1": float(f1_score(rare_truth, rare_prediction, zero_division=0)),
        "reference_rare_spots": int(rare_truth.sum()),
        "predicted_rare_spots": int(rare_prediction.sum()),
    }


def _summarize_seeds(seed_results: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for region in ("whole_section", "leftmost_20_percent"):
        summary[region] = {}
        for metric in ("ari", "nmi", "rare_f1"):
            values = np.asarray(
                [seed_results[str(seed)][region][metric] for seed in (0, 1, 42)],
                dtype=np.float64,
            )
            summary[region][metric] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            }
    return summary


def main() -> int:
    reference = np.load(REFERENCE, allow_pickle=False)
    truth = reference["labels"].astype(str)
    coordinates = np.asarray(reference["coordinates"], dtype=np.float64)
    holdout = np.lexsort((np.arange(len(coordinates)), coordinates[:, 0]))[
        : math.ceil(0.20 * len(coordinates))
    ]
    results: dict[str, dict[str, object]] = {}

    for arm in ("Astar", "B0", "B0_holdout"):
        seed_results = {}
        for seed in (0, 1, 42):
            prediction = np.load(RUNS / f"{arm}_seed{seed}" / "pred.npy", allow_pickle=False)
            if prediction.shape != truth.shape:
                raise ValueError(f"{arm} seed {seed} prediction length differs from the reference")
            seed_results[str(seed)] = {
                "whole_section": _score_region(truth, prediction, truth, prediction),
                "leftmost_20_percent": _score_region(
                    truth[holdout], prediction[holdout], truth, prediction
                ),
            }
        results[arm] = seed_results

    payload = {
        "scope": "Within-section spatial holdout, conditional on the supplied reference labels",
        "holdout_rule": "Leftmost ceil(20% of spots) by x-coordinate; row index breaks coordinate ties",
        "rare_rule": "Reference classes and predicted clusters are defined by their whole-section share <= 5%, then scored within each region",
        "n_spots": int(len(truth)),
        "n_holdout_spots": int(len(holdout)),
        "arms": results,
        "mean_and_sample_sd": {
            arm: _summarize_seeds(seed_results)
            for arm, seed_results in results.items()
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
