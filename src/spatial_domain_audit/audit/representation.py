"""Representation geometry diagnostics: effective rank, k-NN purity, and separability."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def effective_rank(matrix: np.ndarray) -> float:
    """Participation ratio (sum s^2)^2 / sum s^4 of mean-centered singular values."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    return float(total**2 / float((energy**2).sum()))


def cosine_knn_purity(matrix: np.ndarray, labels: np.ndarray, *, k: int = 10) -> dict[str, float]:
    """Cosine k-NN reference label purity paired with the imbalanced chance level."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    k_eff = min(int(k), max(matrix.shape[0] - 1, 1))
    finder = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(unit)
    _, indices = finder.kneighbors(unit)
    neighbor_labels = labels[indices[:, 1:]]
    purity = float((neighbor_labels == labels[:, None]).mean())
    shares = np.bincount(labels) / labels.size
    return {"knn_purity": purity, "chance": float((shares**2).sum())}


def cosine_class_separation(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-class cosine similarity minus mean between-class cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
    gram = unit @ unit.T
    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)
    diff = ~(labels[:, None] == labels[None, :])
    return float(gram[same].mean() - gram[diff].mean())


def leiden_to_target_k(
    matrix: np.ndarray, *, target_k: int = 5, n_neighbors: int = 10, seed: int = 0
) -> np.ndarray:
    """Sweep Leiden resolution across [0.10, 3.00] without label feedback until target_k."""
    import anndata as ad
    import scanpy as sc

    holder = ad.AnnData(X=np.ascontiguousarray(matrix, dtype=np.float32))
    sc.pp.neighbors(holder, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    best: tuple[np.ndarray, int] | None = None
    for candidate in np.round(np.arange(0.10, 3.001, 0.05), 4):
        sc.tl.leiden(holder, resolution=float(candidate), key_added="dyn", random_state=seed)
        labels = holder.obs["dyn"].astype(int).to_numpy()
        k = int(np.unique(labels).size)
        if k == target_k:
            return labels
        if best is None or abs(k - target_k) < abs(best[1] - target_k):
            best = (labels, k)
    assert best is not None
    return best[0]


def upstream_pca_space(adata: Any, *, n_pcs: int = 40, seed: int = 0) -> np.ndarray:
    """Extract normalized log1p expression PCs matching upstream pseudo-label construction."""
    import scanpy as sc

    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.tl.pca(work, svd_solver="arpack", random_state=seed)
    return np.asarray(work.obsm["X_pca"][:, :n_pcs], dtype=np.float32)


def evaluate_representation(
    name: str,
    matrix: np.ndarray,
    truth_codes: np.ndarray,
    *,
    target_k: int = 5,
    knn_k: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute Leiden agreement, effective rank, cosine separability, and k-NN purity."""
    labels = leiden_to_target_k(matrix, target_k=target_k, n_neighbors=knn_k, seed=seed)
    purity = cosine_knn_purity(matrix, truth_codes, k=knn_k)
    return {
        "name": name,
        "dimension": int(matrix.shape[1]),
        "leiden_k": int(np.unique(labels).size),
        "ari": float(adjusted_rand_score(truth_codes, labels)),
        "nmi": float(normalized_mutual_info_score(truth_codes, labels)),
        "effective_rank": effective_rank(matrix),
        "class_separation": cosine_class_separation(matrix, truth_codes),
        **purity,
    }
