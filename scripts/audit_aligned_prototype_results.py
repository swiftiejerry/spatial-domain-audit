from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


SEEDS = (0, 1, 42)
EXPECTED_ARMS = (
    "C1_default",
    "C1_center_quantile",
    "C1_confident_quantile",
    "D0_center_quantile",
)


def _rare_metrics(predicted: np.ndarray, truth: np.ndarray, *, primary: bool) -> dict[str, Any]:
    n_spots = truth.size
    truth_counts = Counter(truth.tolist())
    prediction_counts = Counter(predicted.tolist())

    if primary:
        rare_types = {label for label, count in truth_counts.items() if count * 20 <= n_spots}
        rare_clusters = {label for label, count in prediction_counts.items() if count * 20 <= n_spots}
    else:
        rare_types = {label for label, count in truth_counts.items() if count * 20 < n_spots}
        rare_clusters = {label for label, count in prediction_counts.items() if count * 200 < 9 * n_spots}

    true_rare = np.isin(truth, list(rare_types))
    predicted_rare = np.isin(predicted, list(rare_clusters))
    result: dict[str, Any] = {
        "f1": float(f1_score(true_rare, predicted_rare, zero_division=0)),
        "n_true_rare": int(true_rare.sum()),
        "n_predicted_rare": int(predicted_rare.sum()),
        "n_predicted_rare_clusters": len(rare_clusters),
        "per_type": {},
    }

    for label in sorted(rare_types):
        true_type = truth == label
        true_positive = int(np.count_nonzero(true_type & predicted_rare))
        false_positive = int(np.count_nonzero(~true_type & predicted_rare))
        false_negative = int(np.count_nonzero(true_type & ~predicted_rare))
        precision = (
            None
            if true_positive + false_positive == 0
            else true_positive / (true_positive + false_positive)
        )
        recall = true_positive / int(true_type.sum())
        f1 = (
            0.0
            if precision is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        cluster_enrichment = []
        for cluster in rare_clusters:
            cluster_mask = predicted == cluster
            cluster_enrichment.append(
                float(np.mean(true_type[cluster_mask]) / np.mean(true_type))
            )
        result["per_type"][label] = {
            "reference_spots": int(true_type.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": float(recall),
            "f1": float(f1),
            "maximum_eligible_cluster_enrichment": (
                None if not cluster_enrichment else max(cluster_enrichment)
            ),
        }
    return result


def _summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else 0.0,
        "values_by_seed": values,
    }


def _load_run(run_dir: Path, truth: np.ndarray) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    prediction_path = run_dir / "pred.npy"
    initial_path = run_dir / "warmup_init_labels.npy"
    if not metrics_path.is_file() or not prediction_path.is_file() or not initial_path.is_file():
        raise FileNotFoundError(f"Required saved result is missing in {run_dir.name}")

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    predicted = np.load(prediction_path, allow_pickle=False).reshape(-1)
    initial = np.load(initial_path, allow_pickle=False).reshape(-1)
    if predicted.shape != truth.shape or initial.shape != truth.shape:
        raise ValueError(f"Prediction length mismatch in {run_dir.name}")

    dynamic = saved["dynamic"]
    saved_final = saved["scformer"]
    saved_initial = dynamic["initial_partition"]
    final_primary = _rare_metrics(predicted, truth, primary=True)
    final_secondary = _rare_metrics(predicted, truth, primary=False)
    initial_primary = _rare_metrics(initial, truth, primary=True)
    initial_secondary = _rare_metrics(initial, truth, primary=False)
    final_ari = float(adjusted_rand_score(truth, predicted))
    final_nmi = float(normalized_mutual_info_score(truth, predicted))
    initial_ari = float(adjusted_rand_score(truth, initial))
    initial_nmi = float(normalized_mutual_info_score(truth, initial))

    matches = {
        "final_ari": bool(np.isclose(final_ari, saved_final["ari"], atol=1e-12, rtol=0)),
        "final_nmi": bool(np.isclose(final_nmi, saved_final["nmi"], atol=1e-12, rtol=0)),
        "final_secondary_rare_f1": bool(
            np.isclose(final_secondary["f1"], saved_final["rare_f1"], atol=1e-12, rtol=0)
        ),
        "initial_ari": bool(np.isclose(initial_ari, saved_initial["ari"], atol=1e-12, rtol=0)),
        "initial_nmi": bool(np.isclose(initial_nmi, saved_initial["nmi"], atol=1e-12, rtol=0)),
        "initial_secondary_rare_f1": bool(
            np.isclose(initial_secondary["f1"], saved_initial["rare_f1"], atol=1e-12, rtol=0)
        ),
    }

    run_id = saved["run_id"]
    arm = next((candidate for candidate in EXPECTED_ARMS if run_id.endswith("_" + candidate)), None)
    if arm is None:
        raise ValueError(f"Unrecognized run arm in {run_dir.name}")

    return {
        "run_id": run_id,
        "arm": arm,
        "seed": int(saved["seed"]),
        "n_spots": int(truth.size),
        "n_clusters": int(np.unique(predicted).size),
        "configuration": {
            "center_embedding": bool(dynamic["center_embedding"]),
            "delta_mode": dynamic["delta_mode"],
            "delta": dynamic.get("delta"),
            "delta_quantile": dynamic.get("delta_quantile"),
            "confident_only_refresh": bool(dynamic.get("confident_only_refresh", False)),
            "spatial_relation_enabled": bool(saved["spatial"]["relation_enabled"]),
            "spatial_k": int(saved["spatial"]["k"]),
        },
        "initial": {
            "ari": initial_ari,
            "nmi": initial_nmi,
            "rare_primary": initial_primary,
            "rare_secondary": initial_secondary,
        },
        "final": {
            "ari": final_ari,
            "nmi": final_nmi,
            "rare_primary": final_primary,
            "rare_secondary": final_secondary,
        },
        "refresh_trace": [
            {
                key: event.get(key)
                for key in (
                    "epoch",
                    "confident_fraction",
                    "n_nonempty_before_reseed",
                    "n_nonempty",
                    "n_reseeded",
                    "changed_fraction",
                    "delta_effective",
                )
            }
            for event in dynamic.get("refresh_trace", [])
        ],
        "saved_evaluation_matches_independent_recount": matches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-npz", type=Path, default=Path("data/reference/A1_reference.npz"))
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/reports/aligned_prototype_independent_recount.json"),
    )
    args = parser.parse_args()

    truth = np.load(args.reference_npz, allow_pickle=False)["labels"].astype(str)
    runs = [
        _load_run(path, truth)
        for path in sorted(args.runs_dir.iterdir())
        if path.is_dir() and (path / "metrics.json").is_file()
    ]
    arm_seed_pairs = {(run["arm"], run["seed"]) for run in runs}
    expected = {(arm, seed) for arm in EXPECTED_ARMS for seed in SEEDS}
    if arm_seed_pairs != expected or len(runs) != len(expected):
        missing = sorted(expected - arm_seed_pairs)
        extra = sorted(arm_seed_pairs - expected)
        raise ValueError(f"Run set differs from preregistration. missing={missing}, extra={extra}")

    checks = [
        value
        for run in runs
        for value in run["saved_evaluation_matches_independent_recount"].values()
    ]
    summaries: dict[str, Any] = {}
    for arm in EXPECTED_ARMS:
        selected = sorted((run for run in runs if run["arm"] == arm), key=lambda item: item["seed"])
        summaries[arm] = {
            metric: _summarize([float(run["final"][metric]) for run in selected])
            for metric in ("ari", "nmi")
        }
        summaries[arm]["rare_f1_primary"] = _summarize(
            [float(run["final"]["rare_primary"]["f1"]) for run in selected]
        )
        summaries[arm]["rare_f1_secondary"] = _summarize(
            [float(run["final"]["rare_secondary"]["f1"]) for run in selected]
        )

    payload = {
        "scope": "Descriptive single-section evaluation conditional on the supplied final_annot labels.",
        "n_runs": len(runs),
        "all_expected_runs_present": True,
        "all_saved_evaluations_match_independent_recount": all(checks),
        "rare_primary_rule": "reference class share <= 5%; predicted cluster share <= 5%",
        "rare_secondary_rule": "reference class share < 5%; predicted cluster share < 4.5%",
        "runs": runs,
        "summary_by_arm": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Independent recount: {len(runs)} runs, "
        f"all saved metrics match={payload['all_saved_evaluations_match_independent_recount']}, "
        f"output={args.out.as_posix()}"
    )
    for arm, result in summaries.items():
        print(
            f"{arm}: ARI={result['ari']['mean']:.4f} ± {result['ari']['sample_sd']:.4f}; "
            f"NMI={result['nmi']['mean']:.4f} ± {result['nmi']['sample_sd']:.4f}; "
            f"rare-F1<=5%={result['rare_f1_primary']['mean']:.4f} ± "
            f"{result['rare_f1_primary']['sample_sd']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
