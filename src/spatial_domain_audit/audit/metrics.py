"""CPU-only construct-validity and clustering evaluation metrics.

This module evaluates externally supplied predicted partitions, pseudo-labels,
and embeddings against supplied reference classes without tuning partitions or
hyperparameters against ground-truth labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)

from ..graph.spatial_knn import build_spatial_graph


METRIC_DEFINITIONS = {
    "partition": "Externally supplied partition; no truth-informed clustering occurs here.",
    "rare_primary": "truth class share <= 5%; predicted cluster share <= 5%",
    "rare_secondary": "truth class share < 5%; predicted cluster share < 4.5%",
    "binary": "precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2TP/(2TP+FP+FN); zero denominator -> 0",
    "enrichment": "true-rare fraction among predicted-rare spots / whole-slide true-rare fraction",
    "hungarian": "Maximum one-to-one overlap alignment between cluster IDs and reference labels.",
    "boundary": "Edge-level F1 between (y_i != y_j) and (z_hat_i != z_hat_j) on deduplicated coordinate k-NN edges.",
    "boundary_mixing": "Boundary erasure FN/(TP+FN) and within-cluster truth crossing FN/(TN+FN).",
    "connectivity": "Connected components of each cluster induced subgraph on spatial edges.",
    "churn": "1 - maximum one-to-one overlap(predicted, initial)/N via Hungarian assignment.",
    "effective_rank": "Participation ratio (sum s^2)^2 / sum s^4 of mean-centered singular values s.",
    "embedding_knn": "Cosine k-NN purity excluding self; imbalanced-class chance is sum_c (n_c/N)^2.",
    "summary": "Arithmetic mean and sample SD (ddof=1); descriptive paired contrasts across matched seeds.",
}


def _codes(values: object, name: str) -> np.ndarray:
    codes = np.asarray(values)
    if codes.ndim != 1 or not codes.size:
        raise ValueError(f"{name} must be a nonempty 1-D array")
    if codes.dtype.kind not in "iu" or np.any(codes < 0):
        raise ValueError(f"{name} must contain nonnegative integer codes")
    return codes.astype(np.int64, copy=False)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
    }


def _alignment(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    pred_ids, pred_inverse = np.unique(predicted, return_inverse=True)
    ref_ids, ref_inverse = np.unique(reference, return_inverse=True)
    overlap = np.zeros((pred_ids.size, ref_ids.size), dtype=np.int64)
    np.add.at(overlap, (pred_inverse, ref_inverse), 1)
    rows, columns = linear_sum_assignment(-overlap)
    aligned_reference_index = np.full(predicted.size, -1, dtype=np.int64)
    for row, column in zip(rows, columns):
        aligned_reference_index[pred_inverse == row] = column
    matched = int(overlap[rows, columns].sum())
    return {
        "mapping": [
            {
                "predicted_label": int(pred_ids[row]),
                "reference_label": int(ref_ids[column]),
                "overlap": int(overlap[row, column]),
            }
            for row, column in zip(rows, columns)
        ],
        "unmatched_predicted_labels": [int(pred_ids[i]) for i in range(pred_ids.size) if i not in set(rows)],
        "matched_count": matched,
        "aligned_reference_index": aligned_reference_index,
        "reference_ids": ref_ids,
        "reference_inverse": ref_inverse,
    }


def align_labels_to_reference(predicted: object, reference: object) -> np.ndarray:
    """Relabel ``predicted`` integer IDs into ``reference`` label space via Hungarian matching."""
    pred = _codes(predicted, "predicted")
    ref = _codes(reference, "reference")
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference must have identical lengths")
    matched = _alignment(pred, ref)
    mapping = {row["predicted_label"]: row["reference_label"] for row in matched["mapping"]}
    next_id = int(ref.max()) + 1
    out = np.empty_like(pred)
    for label in np.unique(pred):
        target = mapping.get(int(label))
        if target is None:
            target = next_id
            next_id += 1
        out[pred == label] = target
    return out


def assignment_churn(predicted: object, initial_labels: object) -> dict[str, Any]:
    """Compute spot deviations after maximum-overlap cluster-ID alignment."""
    pred, initial = _codes(predicted, "predicted"), _codes(initial_labels, "initial_labels")
    if pred.shape != initial.shape:
        raise ValueError("predicted and initial_labels must have identical lengths")
    matched = _alignment(pred, initial)
    changed = pred.size - matched["matched_count"]
    return {
        "changed_count": int(changed),
        "fraction": float(changed / pred.size),
        "n_spots": int(pred.size),
        "ari": float(adjusted_rand_score(initial, pred)),
        "mapping": matched["mapping"],
        "unmatched_predicted_labels": matched["unmatched_predicted_labels"],
    }


def binary_rare_scores(
    pred: np.ndarray,
    truth: pd.Series | np.ndarray,
    ref_thr: float = 0.05,
    pred_thr: float = 0.045,
) -> dict[str, Any]:
    """Compute binary rare-vs-non-rare precision, recall, and F1 over string/code labels."""
    truth_s = pd.Series(np.asarray(truth).astype(str))
    truth_share = truth_s.value_counts(normalize=True)
    rare_types = truth_share[truth_share < ref_thr].index
    y_true = truth_s.isin(rare_types).astype(int).to_numpy()

    pred_s = pd.Series(pred)
    cluster_share = pred_s.value_counts(normalize=True)
    rare_clusters = cluster_share[cluster_share < pred_thr].index
    y_pred = pred_s.isin(rare_clusters).astype(int).to_numpy()

    return {
        "rare_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "rare_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "rare_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_true_rare": int(y_true.sum()),
        "n_pred_rare": int(y_pred.sum()),
        "rare_types": [str(t) for t in rare_types],
        "n_rare_clusters": int(len(rare_clusters)),
    }


def per_rare_class_capture(
    pred: np.ndarray,
    truth: pd.Series | np.ndarray,
    ref_thr: float = 0.05,
    pred_thr: float = 0.045,
) -> dict[str, dict[str, float | int]]:
    """Per-domain spot capture inside sub-threshold predicted clusters."""
    truth_s = pd.Series(np.asarray(truth).astype(str))
    truth_share = truth_s.value_counts(normalize=True)
    rare_types = truth_share[truth_share < ref_thr].index
    pred_s = pd.Series(pred)
    cluster_share = pred_s.value_counts(normalize=True)
    rare_clusters = set(cluster_share[cluster_share < pred_thr].index)
    truth_values = truth_s.to_numpy()
    result: dict[str, dict[str, float | int]] = {}
    for rare_type in rare_types:
        mask = truth_values == str(rare_type)
        total = int(mask.sum())
        hit = int(np.isin(pred[mask], list(rare_clusters)).sum()) if rare_clusters else 0
        result[str(rare_type)] = {
            "n_cells": total,
            "n_in_rare_cluster": hit,
            "recall": (hit / total) if total else 0.0,
        }
    return result


def cluster_summary_table(pred: np.ndarray, truth: pd.Series | np.ndarray) -> pd.DataFrame:
    """Summarize cluster sizes, dominant reference class, and purity per cluster."""
    frame = pd.DataFrame({"cluster": pred, "truth": np.asarray(truth).astype(str)})
    rows = []
    for cluster, sub in frame.groupby("cluster"):
        counts = sub["truth"].value_counts()
        rows.append(
            {
                "cluster": int(cluster),
                "n_cells": len(sub),
                "pct": round(100 * len(sub) / len(frame), 2),
                "top_truth": counts.index[0],
                "top_n": int(counts.iloc[0]),
                "purity": round(counts.iloc[0] / len(sub), 3),
                "n_truth_types": int(sub["truth"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)


def _class_name(label: int, names: Mapping[int, str] | Sequence[str] | None) -> str:
    if names is None:
        return str(label)
    try:
        return str(names[label])
    except (KeyError, IndexError) as exc:
        raise ValueError(f"class_names has no entry for truth code {label}") from exc


def _rare_metrics(
    pred: np.ndarray, truth: np.ndarray, *, primary: bool
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    truth_ids, truth_inverse, truth_sizes = np.unique(truth, return_inverse=True, return_counts=True)
    pred_ids, pred_inverse, pred_sizes = np.unique(pred, return_inverse=True, return_counts=True)
    n = int(pred.size)
    if primary:
        true_rare_classes = truth_sizes * 20 <= n
        pred_rare_clusters = pred_sizes * 20 <= n
    else:
        true_rare_classes = truth_sizes * 20 < n
        pred_rare_clusters = pred_sizes * 200 < 9 * n
    true_rare, pred_rare = true_rare_classes[truth_inverse], pred_rare_clusters[pred_inverse]
    scores = _binary_metrics(true_rare, pred_rare)
    true_count, predicted_count = int(true_rare.sum()), int(pred_rare.sum())
    captured_fraction = _ratio(scores["tp"], predicted_count)
    prevalence = true_count / n
    per_class_hits = []
    for label in truth_ids[true_rare_classes]:
        mask = truth == label
        count = int(np.count_nonzero(mask & pred_rare))
        per_class_hits.append(
            {
                "truth_label": int(label),
                "support": int(mask.sum()),
                "hits_in_predicted_rare_clusters": count,
                "capture_recall": float(count / mask.sum()),
            }
        )
    scores.update(
        {
            "definition": METRIC_DEFINITIONS["rare_primary" if primary else "rare_secondary"],
            "true_rare_labels": truth_ids[true_rare_classes].astype(int).tolist(),
            "predicted_rare_clusters": pred_ids[pred_rare_clusters].astype(int).tolist(),
            "true_rare_count": true_count,
            "predicted_rare_count": predicted_count,
            "rare_absolute_hits": scores["tp"],
            "whole_slide_rare_fraction": prevalence,
            "rare_fraction_in_predicted_rare_clusters": captured_fraction,
            "enrichment": _ratio(captured_fraction, prevalence) if captured_fraction is not None else None,
            "per_true_rare_class_capture": per_class_hits,
        }
    )
    return scores, true_rare, pred_rare


def _undirected_edges(edge_index: object, n_spots: int) -> np.ndarray:
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("spatial_edges must be an integer array of shape (2, E)")
    if np.any(edges < 0) or np.any(edges >= n_spots):
        raise ValueError("spatial_edges contain out-of-range spot indices")
    pairs = np.sort(edges.astype(np.int64, copy=False).T, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


def _spatial_metrics(pred: np.ndarray, truth: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    pairs = _undirected_edges(edges, pred.size)
    left, right = pairs.T
    true_boundary = truth[left] != truth[right]
    pred_boundary = pred[left] != pred[right]
    boundary = _binary_metrics(true_boundary, pred_boundary)
    boundary["truth_positive_edges"] = int(true_boundary.sum())
    boundary["predicted_positive_edges"] = int(pred_boundary.sum())
    parent = np.arange(pred.size)
    degree = np.zeros(pred.size, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for source, target in pairs[~pred_boundary]:
        root_source, root_target = find(int(source)), find(int(target))
        if root_source != root_target:
            parent[root_target] = root_source
        degree[source] += 1
        degree[target] += 1
    roots = np.asarray([find(i) for i in range(pred.size)])
    connectivity = []
    for label in np.unique(pred):
        mask = pred == label
        _, sizes = np.unique(roots[mask], return_counts=True)
        connectivity.append(
            {
                "label": int(label),
                "size": int(mask.sum()),
                "n_components": int(sizes.size),
                "component_sizes_descending": sorted(sizes.astype(int).tolist(), reverse=True),
                "largest_component_fraction": float(sizes.max() / mask.sum()),
                "isolated_spot_count": int(np.count_nonzero(degree[mask] == 0)),
            }
        )
    return {
        "n_undirected_edges": int(pairs.shape[0]),
        "boundary": boundary,
        "boundary_mixing": {
            "truth_cross_fraction": _ratio(int(true_boundary.sum()), pairs.shape[0]),
            "predicted_cross_fraction": _ratio(int(pred_boundary.sum()), pairs.shape[0]),
            "erased_true_boundary_fraction": _ratio(boundary["fn"], boundary["tp"] + boundary["fn"]),
            "truth_crossing_within_predicted_cluster_fraction": _ratio(
                boundary["fn"], boundary["tn"] + boundary["fn"]
            ),
            "false_predicted_boundary_fraction": _ratio(boundary["fp"], boundary["tp"] + boundary["fp"]),
        },
        "connectivity": connectivity,
        "size_weighted_largest_component_fraction": float(
            sum(r["component_sizes_descending"][0] for r in connectivity) / pred.size
        ),
        "isolated_spot_fraction_within_predicted_clusters": float(np.mean(degree == 0)),
    }


def embedding_diagnostics(
    embedding: object,
    truth: object,
    *,
    knn_k: int = 10,
    spatial_edges: object | None = None,
) -> dict[str, Any]:
    """Measure embedding rank, cosine class separation, and k-NN purity."""
    labels = _codes(truth, "truth")
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != labels.size or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("embedding must be finite, nonempty, 2-D and aligned with labels")
    if isinstance(knn_k, bool) or not isinstance(knn_k, (int, np.integer)) or knn_k < 1:
        raise ValueError("knn_k must be a positive integer")
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    rank = float(total**2 / np.sum(energy**2)) if total else 0.0
    norms = np.linalg.norm(values, axis=1)
    unit = np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] != 0)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    group_sums = np.zeros((counts.size, values.shape[1]), dtype=np.float64)
    np.add.at(group_sums, inverse, unit)
    group_dot = float(np.sum(group_sums**2))
    norm_dot = float(np.sum(unit**2))
    same_pairs = int(np.sum(counts * (counts - 1)))
    different_pairs = int(labels.size**2 - np.sum(counts**2))
    same_cosine = _ratio(group_dot - norm_dot, same_pairs)
    different_cosine = _ratio(float(np.sum(unit.sum(axis=0) ** 2)) - group_dot, different_pairs)
    k = min(int(knn_k), labels.size - 1)
    hits = 0
    if k:
        for start in range(0, labels.size, 256):
            stop = min(start + 256, labels.size)
            similarities = unit[start:stop] @ unit.T
            similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
            hits += int(np.count_nonzero(labels[neighbors] == labels[start:stop, None]))
    scores: dict[str, Any] = {
        "n_spots": int(labels.size),
        "dimension": int(values.shape[1]),
        "effective_rank": rank,
        "centered_total_sum_squares": total,
        "zero_norm_spot_count": int(np.count_nonzero(norms == 0)),
        "knn_k": k,
        "knn_purity": _ratio(hits, labels.size * k),
        "chance": float(np.sum((counts / labels.size) ** 2)),
        "chance_without_self": _ratio(same_pairs, labels.size * (labels.size - 1)),
        "same_class_cosine": same_cosine,
        "different_class_cosine": different_cosine,
        "class_separation": (
            same_cosine - different_cosine
            if same_cosine is not None and different_cosine is not None
            else None
        ),
    }
    if spatial_edges is not None:
        pairs = _undirected_edges(spatial_edges, labels.size)
        left, right = pairs.T
        scores["spatial_neighbor_cosine_mean"] = (
            float(np.mean(np.sum(unit[left] * unit[right], axis=1))) if pairs.size else None
        )
        scores["spatial_neighbor_squared_distance_mean"] = (
            float(np.mean(np.sum((values[left] - values[right]) ** 2, axis=1))) if pairs.size else None
        )
    return scores


def evaluate_partition(
    predicted: object,
    truth: object,
    *,
    coordinates: object | None = None,
    spatial_edges: object | None = None,
    initial_labels: object | None = None,
    embedding: object | None = None,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    spatial_k: int = 10,
    knn_k: int = 10,
) -> dict[str, Any]:
    """Compute complete clustering, rare-domain, boundary, and movement diagnostics."""
    pred, labels = _codes(predicted, "predicted"), _codes(truth, "truth")
    if pred.shape != labels.shape:
        raise ValueError("predicted and truth must have identical lengths")
    primary, primary_truth_mask, _ = _rare_metrics(pred, labels, primary=True)
    secondary, secondary_truth_mask, _ = _rare_metrics(pred, labels, primary=False)
    aligned = _alignment(pred, labels)
    per_class = []
    for index, label in enumerate(aligned["reference_ids"]):
        mask = labels == label
        assigned = aligned["aligned_reference_index"] == index
        scores = _binary_metrics(mask, assigned)
        scores.update(
            {
                "truth_label": int(label),
                "name": _class_name(int(label), class_names),
                "support": int(mask.sum()),
                "predicted_count": int(assigned.sum()),
                "is_rare_primary": bool(primary_truth_mask[mask][0]),
                "is_rare_secondary": bool(secondary_truth_mask[mask][0]),
            }
        )
        per_class.append(scores)
    cluster_sizes = []
    for label, size in zip(*np.unique(pred, return_counts=True)):
        mask = pred == label
        primary_hits = int(np.count_nonzero(mask & primary_truth_mask))
        secondary_hits = int(np.count_nonzero(mask & secondary_truth_mask))
        cluster_sizes.append(
            {
                "label": int(label),
                "size": int(size),
                "fraction": float(size / pred.size),
                "is_rare_primary": bool(size * 20 <= pred.size),
                "is_rare_secondary": bool(size * 200 < 9 * pred.size),
                "true_rare_count_primary": primary_hits,
                "true_rare_fraction_primary": float(primary_hits / size),
                "rare_enrichment_primary": _ratio(primary_hits / size, primary["whole_slide_rare_fraction"]),
                "true_rare_count_secondary": secondary_hits,
                "true_rare_fraction_secondary": float(secondary_hits / size),
                "rare_enrichment_secondary": _ratio(secondary_hits / size, primary["whole_slide_rare_fraction"]),
            }
        )
    result: dict[str, Any] = {
        "n_spots": int(pred.size),
        "n_clusters": int(np.unique(pred).size),
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred, average_method="arithmetic")),
        "rare_primary": primary,
        "rare_secondary": secondary,
        "cluster_sizes": cluster_sizes,
        "hungarian": {
            "mapping": aligned["mapping"],
            "unmatched_predicted_labels": aligned["unmatched_predicted_labels"],
            "accuracy": float(aligned["matched_count"] / pred.size),
            "macro_f1": float(np.mean([r["f1"] for r in per_class])),
            "per_class": per_class,
        },
    }
    edges = spatial_edges
    if coordinates is not None:
        coords = np.asarray(coordinates)
        if coords.ndim != 2 or coords.shape[0] != pred.size or not coords.shape[1] or not np.isfinite(coords).all():
            raise ValueError("coordinates must be finite, 2-D and aligned with labels")
        if edges is None:
            edges = build_spatial_graph(coords, mode="knn", k=spatial_k, weighted=False).edge_index
    if edges is not None:
        result["spatial"] = _spatial_metrics(pred, labels, edges)
        result["spatial"]["graph_source"] = (
            "provided_common_edge_index" if spatial_edges is not None else "coordinate_knn_undirected_union"
        )
        result["spatial"]["k"] = None if spatial_edges is not None else spatial_k
    if initial_labels is not None:
        result["assignment_churn"] = assignment_churn(pred, initial_labels)
    if embedding is not None:
        result["embedding"] = embedding_diagnostics(embedding, labels, knn_k=knn_k, spatial_edges=edges)
    return result


def mean_sample_sd(values: Sequence[float]) -> dict[str, Any]:
    """Compute arithmetic mean, sample SD (ddof=1), and min/max across finite values."""
    numbers = np.asarray(values, dtype=np.float64)
    if numbers.ndim != 1 or not numbers.size or not np.isfinite(numbers).all():
        raise ValueError("values must be a nonempty finite 1-D sequence")
    return {
        "n": int(numbers.size),
        "values": numbers.tolist(),
        "mean": float(numbers.mean()),
        "sd": float(numbers.std(ddof=1)) if numbers.size >= 2 else None,
        "min": float(numbers.min()),
        "max": float(numbers.max()),
    }


def paired_difference_summary(
    reference_by_seed: Mapping[int, float], treatment_by_seed: Mapping[int, float]
) -> dict[str, Any]:
    """Summarize paired treatment-minus-reference differences over identical seed sets."""
    if not reference_by_seed or set(reference_by_seed) != set(treatment_by_seed):
        raise ValueError("paired comparison requires identical nonempty seed sets")
    seeds = sorted(reference_by_seed)
    reference = mean_sample_sd([reference_by_seed[s] for s in seeds])
    treatment = mean_sample_sd([treatment_by_seed[s] for s in seeds])
    delta = np.asarray(treatment["values"]) - np.asarray(reference["values"])
    result = mean_sample_sd(delta)
    result.update(
        {
            "seeds": seeds,
            "reference": reference,
            "treatment": treatment,
            "direction": "treatment_minus_reference",
            "all_strictly_positive": bool(np.all(delta > 0)),
            "all_strictly_negative": bool(np.all(delta < 0)),
            "all_nonnegative": bool(np.all(delta >= 0)),
            "same_nonzero_sign": bool(np.all(delta > 0) or np.all(delta < 0)),
            "inference": "descriptive_only_no_significance_claim",
        }
    )
    return result
