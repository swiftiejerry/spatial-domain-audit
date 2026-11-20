from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def binary_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(truth & prediction))
    false_positive = int(np.count_nonzero(~truth & prediction))
    false_negative = int(np.count_nonzero(truth & ~prediction))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def evaluate_threshold(
    reference: np.ndarray,
    prediction: np.ndarray,
    rare_labels: list[str],
    threshold_percent: float,
) -> dict[str, object]:
    spot_count = len(reference)
    _, cluster_ids, cluster_sizes = np.unique(
        prediction, return_inverse=True, return_counts=True
    )
    cluster_shares = cluster_sizes / spot_count
    eligible_ids = np.flatnonzero(cluster_shares < threshold_percent / 100.0)
    predicted_rare = np.isin(cluster_ids, eligible_ids)
    true_rare = np.isin(reference, rare_labels)

    per_type: dict[str, dict[str, float | int | None]] = {}
    for label in rare_labels:
        true_type = reference == label
        true_positive = int(np.count_nonzero(true_type & predicted_rare))
        type_size = int(np.count_nonzero(true_type))
        predicted_size = int(np.count_nonzero(predicted_rare))
        precision = None if predicted_size == 0 else true_positive / predicted_size
        recall = 0.0 if type_size == 0 else true_positive / type_size
        type_f1 = 0.0 if precision is None or precision + recall == 0 else (
            2 * precision * recall / (precision + recall)
        )

        eligible_cluster_enrichment: list[float] = []
        for cluster_id in eligible_ids:
            cluster_mask = cluster_ids == cluster_id
            cluster_type_share = float(np.mean(true_type[cluster_mask]))
            global_type_share = type_size / spot_count
            eligible_cluster_enrichment.append(cluster_type_share / global_type_share)

        per_type[label] = {
            "reference_spots": type_size,
            "precision": precision,
            "recall": recall,
            "f1": type_f1,
            "maximum_eligible_cluster_enrichment": (
                None if not eligible_cluster_enrichment else max(eligible_cluster_enrichment)
            ),
        }

    return {
        "prediction_threshold_percent_strictly_below": threshold_percent,
        "eligible_predicted_clusters": len(eligible_ids),
        "predicted_rare_spots": int(np.count_nonzero(predicted_rare)),
        "predicted_rare_share": float(np.mean(predicted_rare)),
        "binary_rare_f1": binary_f1(true_rare, predicted_rare),
        "per_rare_type": per_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recount rare-class recall, F1, enrichment, and predicted-size threshold "
            "sensitivity from saved partitions."
        )
    )
    parser.add_argument(
        "--reference-npz",
        type=Path,
        default=Path("data/reference/A1_reference.npz"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/reports/rare_class_threshold_sensitivity.json"),
    )
    args = parser.parse_args()

    reference_bundle = np.load(args.reference_npz, allow_pickle=False)
    reference = reference_bundle["labels"].astype(str)
    labels, label_counts = np.unique(reference, return_counts=True)
    rare_labels = labels[label_counts / len(reference) < 0.05].tolist()
    thresholds = np.arange(1.0, 12.0 + 0.25, 0.5).round(1).tolist()

    run_results: list[dict[str, object]] = []
    for prediction_path in sorted(args.runs_root.rglob("pred.npy")):
        prediction = np.load(prediction_path, allow_pickle=False).reshape(-1)
        if prediction.size != reference.size:
            continue
        metrics_path = prediction_path.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        run_results.append(
            {
                "run_id": metrics.get("run_id", prediction_path.parent.name),
                "relative_run_path": prediction_path.parent.relative_to(args.runs_root).as_posix(),
                "arm": metrics.get("arm"),
                "seed": metrics.get("seed"),
                "thresholds": [
                    evaluate_threshold(reference, prediction.astype(str), rare_labels, value)
                    for value in thresholds
                ],
            }
        )

    result = {
        "schema_version": 1,
        "reference_spots": int(reference.size),
        "reference_rare_rule": "class share strictly below 5%",
        "reference_rare_labels": [
            {"label": label, "spots": int(count), "share": float(count / len(reference))}
            for label, count in zip(labels, label_counts, strict=True)
            if count / len(reference) < 0.05
        ],
        "prediction_threshold_rule": "cluster share strictly below the listed threshold",
        "threshold_grid_percent": thresholds,
        "run_count": len(run_results),
        "runs": run_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Recounted {len(run_results)} A1 predictions across "
        f"{len(thresholds)} thresholds; rare classes={len(rare_labels)}; "
        f"output={args.output.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
