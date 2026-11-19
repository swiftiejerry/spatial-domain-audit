"""Endpoint attainability diagnostics: partition merge bounds and Leiden resolution frontier."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


def binary_rare_f1_from_counts(
    group_counts: Sequence[tuple[int, int]],
    n_total: int,
    n_true_rare: int,
    *,
    pred_thr: float = 0.045,
) -> float:
    """Compute binary rare-vs-non-rare F1 from per-group (size, rare_count) pairs."""
    limit = pred_thr * n_total
    n_pred = sum(size for size, _ in group_counts if size < limit)
    tp = sum(rare for size, rare in group_counts if size < limit)
    if n_pred == 0 or n_true_rare == 0:
        return 0.0
    precision = tp / n_pred
    recall = tp / n_true_rare
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def stirling_k_partitions(items: Sequence[int], k: int):
    """Yield every partition of ``items`` into exactly ``k`` nonempty blocks (S(n, k))."""
    if k < 1 or k > len(items):
        return
    if k == 1:
        yield [list(items)]
        return
    if len(items) == k:
        yield [[x] for x in items]
        return
    first, rest = items[0], items[1:]
    for blocks in stirling_k_partitions(rest, k - 1):
        yield [[first]] + blocks
    for blocks in stirling_k_partitions(rest, k):
        for idx in range(len(blocks)):
            yield blocks[:idx] + [[first] + blocks[idx]] + blocks[idx + 1 :]


def greedy_rare_partition(
    class_counts: pd.Series,
    rare_types: Sequence[str],
    n_total: int,
    *,
    target_k: int = 5,
    pred_thr: float = 0.045,
) -> dict[str, Any]:
    """Constructive partition isolating rare domains into compact groups under threshold."""
    limit = pred_thr * n_total
    remaining = sorted(((int(class_counts[t]), str(t)) for t in rare_types), reverse=True)
    small_groups: list[list[str]] = []
    small_sizes: list[int] = []
    for size, name in remaining:
        placed = False
        for idx, existing in enumerate(small_sizes):
            if existing + size < limit:
                small_groups[idx].append(name)
                small_sizes[idx] = existing + size
                placed = True
                break
        if not placed:
            small_groups.append([name])
            small_sizes.append(size)
    n_true_rare = int(class_counts[list(rare_types)].sum())
    groups = [(size, size) for size in small_sizes] + [(n_total - sum(small_sizes), 0)]
    return {
        "n_small_clusters": len(small_groups),
        "small_cluster_sizes": small_sizes,
        "small_cluster_members": small_groups,
        "clusters_used": len(small_groups) + 1,
        "feasible_at_target_k": bool(len(small_groups) + 1 <= target_k),
        "rare_f1": binary_rare_f1_from_counts(groups, n_total, n_true_rare, pred_thr=pred_thr),
    }


def enumerate_partition_merges(
    class_counts: pd.Series,
    rare_types: Sequence[str],
    n_total: int,
    *,
    target_k: int = 5,
    pred_thr: float = 0.045,
) -> dict[str, Any]:
    """Enumerate all S(n_classes, target_k) annotation merges to bound binary rare-F1."""
    class_names = list(class_counts.index)
    sizes = np.asarray([int(class_counts[t]) for t in class_names], dtype=np.int64)
    rare_set = set(rare_types)
    rare_counts = np.asarray(
        [int(class_counts[t]) if t in rare_set else 0 for t in class_names], dtype=np.int64
    )
    n_true_rare = int(rare_counts.sum())
    best_f1 = 0.0
    best_partition: list[list[str]] | None = None
    total_partitions = 0
    for blocks in stirling_k_partitions(list(range(len(class_names))), target_k):
        total_partitions += 1
        groups = [(int(sizes[b].sum()), int(rare_counts[b].sum())) for b in blocks]
        value = binary_rare_f1_from_counts(groups, n_total, n_true_rare, pred_thr=pred_thr)
        if value > best_f1:
            best_f1 = value
            best_partition = [[class_names[i] for i in b] for b in blocks]
    return {
        "n_partitions": total_partitions,
        "best_rare_f1": best_f1,
        "best_partition": best_partition,
    }


def sweep_leiden_frontier(
    expression_space: np.ndarray,
    truth: Sequence[str] | np.ndarray,
    resolutions: Sequence[float],
    *,
    n_neighbors: int = 10,
    seed: int = 0,
    ref_thr: float = 0.05,
    pred_thr: float = 0.045,
) -> list[dict[str, Any]]:
    """Measure global agreement and rare-domain attainability across Leiden resolutions."""
    import anndata as ad
    import scanpy as sc

    truth_s = pd.Series(np.asarray(truth).astype(str))
    n_total = int(truth_s.size)
    share = truth_s.value_counts(normalize=True)
    rare_types = set(share[share < ref_thr].index)
    is_rare = truth_s.isin(rare_types).to_numpy()
    y_true = is_rare.astype(int)
    baseline_rare = float(is_rare.mean())
    limit = pred_thr * n_total

    holder = ad.AnnData(X=np.ascontiguousarray(expression_space, dtype=np.float32))
    sc.pp.neighbors(holder, n_neighbors=n_neighbors, use_rep="X", random_state=seed)

    rows: list[dict[str, Any]] = []
    for res in resolutions:
        sc.tl.leiden(holder, resolution=float(res), key_added="sweep", random_state=seed)
        labels = holder.obs["sweep"].astype(int).to_numpy()
        sizes = np.bincount(labels)
        nonzero = sizes[sizes > 0]
        small_clusters = np.where((sizes > 0) & (sizes < limit))[0]
        small_mask = np.isin(labels, small_clusters)
        y_pred = small_mask.astype(int)
        n_small_spots = int(small_mask.sum())
        small_rare_share = float(is_rare[small_mask].mean()) if n_small_spots else None
        rows.append(
            {
                "resolution": float(res),
                "k": int(nonzero.size),
                "ari": float(adjusted_rand_score(truth_s, labels)),
                "nmi": float(normalized_mutual_info_score(truth_s, labels)),
                "rare_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "rare_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "rare_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "min_cluster_share": float(nonzero.min() / n_total),
                "n_small_clusters": int(small_clusters.size),
                "n_small_spots": n_small_spots,
                "small_rare_share": small_rare_share,
                "baseline_rare_share": baseline_rare,
            }
        )
    return rows
