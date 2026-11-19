"""Deterministic normalized-log and gene-standardized Top-Z bipartite graph builder."""

from __future__ import annotations

import os
import pickle
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse


def _as_csr_genes_by_cells(rna_matrix: Any) -> csr_matrix:
    return rna_matrix.tocsr() if issparse(rna_matrix) else csr_matrix(rna_matrix)


def normalized_log_matrix(rna_matrix: Any) -> csr_matrix:
    """Compute library-size-normalized log1p expression matrix X (genes x cells).

    Each spot column is scaled to 10,000 total counts before applying log1p.
    Because log1p preserves the nonzero sparsity pattern of counts, structural
    connectivity remains consistent when swapping reconstruction targets.
    """
    counts = _as_csr_genes_by_cells(rna_matrix).tocsc(copy=True).astype(np.float64)
    library = np.asarray(counts.sum(axis=0)).ravel()
    library[library == 0] = 1.0
    for column in range(counts.shape[1]):
        start, end = counts.indptr[column], counts.indptr[column + 1]
        if end > start:
            counts.data[start:end] *= 1e4 / library[column]
    counts.data = np.log1p(counts.data)
    return counts.tocsr()


def gene_zscore(rna_matrix: Any) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Return sparse log1p matrix X together with per-gene population mean and std.

    Zero-variance genes are assigned standard deviation 1.0 to avoid division by
    zero while keeping sparse X unmaterialized until batch assembly.
    """
    x = normalized_log_matrix(rna_matrix)
    n_cells = x.shape[1]
    total = np.asarray(x.sum(axis=1)).ravel()
    squared = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    mean = total / n_cells
    variance = np.maximum(squared / n_cells - mean**2, 0.0)
    std = np.sqrt(variance)
    std[std == 0] = 1.0
    return x, mean, std


def deterministic_topz_selection(rna_matrix: Any, k: int = 20, chunk: int = 256) -> dict[int, list[int]]:
    """Select the k highest gene-standardized Z-score indices per spot deterministically."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or int(k) < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    x, mean, std = gene_zscore(rna_matrix)
    n_genes, n_cells = x.shape
    if k > n_genes:
        raise ValueError(f"k={k} exceeds gene count {n_genes}")
    x_cells = x.T.tocsr()
    selection: dict[int, list[int]] = {}
    for start in range(0, n_cells, int(chunk)):
        stop = min(start + int(chunk), n_cells)
        block = np.asarray(x_cells[start:stop].todense(), dtype=np.float64)
        z = (block - mean) / std
        order = np.argsort(-z, axis=1, kind="stable")[:, :k]
        for offset in range(stop - start):
            selection[start + offset] = sorted(int(g) for g in order[offset])
    return selection


def deterministic_topz_batch_select(
    rna_matrix: Any,
    blocks: Sequence[np.ndarray],
    *,
    k: int = 20,
    save_path: str | None = None,
) -> tuple[list[dict[str, list[int]]], np.ndarray, dict[int, dict[str, Any]]]:
    """Package deterministic Top-Z selections into the upstream batch output contract."""
    matrix = _as_csr_genes_by_cells(rna_matrix)
    if save_path is not None:
        cache = os.path.join(save_path, "indices_ss.pkl")
        node_file = os.path.join(save_path, "Node_Ids.pkl")
        dic_file = os.path.join(save_path, "dic.pkl")
        if all(os.path.exists(path) for path in (cache, node_file, dic_file)):
            with open(cache, "rb") as handle:
                indices_ss = pickle.load(handle)
            with open(node_file, "rb") as handle:
                node_ids = pickle.load(handle)
            with open(dic_file, "rb") as handle:
                dic = pickle.load(handle)
            return indices_ss, node_ids, dic

    selection = deterministic_topz_selection(matrix, k=k)
    node_ids = np.concatenate([np.asarray(b, dtype=np.int64) for b in blocks])
    indices_ss: list[dict[str, list[int]]] = []
    dic: dict[int, dict[str, Any]] = {}
    for block in blocks:
        genes: list[int] = []
        cells = [int(cell) for cell in np.asarray(block).reshape(-1)]
        for cell in cells:
            picked = selection[cell]
            dic[cell] = {"g": list(picked)}
            genes.extend(picked)
        indices_ss.append({"gene_index": sorted(set(genes)), "cell_index": cells})

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "indices_ss.pkl"), "wb") as handle:
            pickle.dump(indices_ss, handle)
        with open(os.path.join(save_path, "Node_Ids.pkl"), "wb") as handle:
            pickle.dump(node_ids, handle)
        with open(os.path.join(save_path, "dic.pkl"), "wb") as handle:
            pickle.dump(dic, handle)
    return indices_ss, node_ids, dic


def selection_edge_count(selection: dict[int, Sequence[int]], batch: dict[str, Any]) -> int:
    """Count directed per-spot selected spot-gene edges present inside a batch."""
    genes = set(int(g) for g in batch["gene_index"])
    total = 0
    for cell in batch["cell_index"]:
        picked = selection.get(int(cell))
        if picked is not None:
            total += len(genes.intersection(int(g) for g in picked))
    return total
