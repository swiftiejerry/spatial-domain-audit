"""Shared determinism, hashing, pseudo-label, and batching utilities across runners."""

from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..graph.spatial_batching import (
    add_spatial_halo,
    core_incident_edge_retention,
    spatial_batch_order,
    spatial_batch_select_whole,
)
from ..graph.spatial_knn import build_spatial_graph
from ..graph.topz_normalized import deterministic_topz_batch_select, selection_edge_count

DEFAULT_HP = dict(
    cell_size=30,
    n_hidden=104,
    n_heads=8,
    n_layers=3,
    lr=5e-4,
    weight_decay=0.1,
    labsm=0.1,
    epochs=100,
)
UPSTREAM_N_NEIGHBORS = 10


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically."""
    import torch
    from torch.backends import cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)


def snapshot_rng_state() -> tuple[Any, list[Any]]:
    import torch

    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def restore_rng_state(state: tuple[Any, list[Any]]) -> None:
    import torch

    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def file_sha256(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Return canonical SHA-256 digest over a sorted PyTorch state_dict."""
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def resolve_default_resolution(n_spots: int) -> float:
    """Mirror upstream ``scformer.utils.initial_clustering`` segment resolution threshold."""
    if n_spots <= 500:
        return 0.2
    if n_spots <= 5000:
        return 0.5
    return 0.8


def upstream_leiden_labels(
    rna_matrix: Any,
    resolution: float | None = None,
    n_neighbors: int = UPSTREAM_N_NEIGHBORS,
) -> list[int]:
    """Generate Leiden pseudo-labels matching the untouched upstream scFormer pipeline."""
    from scformer.utils import initial_clustering

    if resolution is None:
        return [int(x) for x in initial_clustering(rna_matrix)]
    return [
        int(x)
        for x in initial_clustering(
            rna_matrix,
            custom_resolution=float(resolution),
            custom_n_neighbors=int(n_neighbors),
        )
    ]


def prepare_batches_and_graph(
    rna_matrix: Any,
    coords: np.ndarray,
    *,
    spatial_k: int = 10,
    cell_size: int = 30,
    graph_mode: str = "upstream_sampled",
    topz_k: int = 20,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, list[int]], Any, dict[str, float]]:
    """Construct spatial graph, tile-and-halo batches, and spot-gene selections."""
    n_cells = int(rna_matrix.shape[1])
    graph = build_spatial_graph(coords, mode="knn", k=spatial_k, weighted=False)
    n_batches = math.ceil(n_cells / int(cell_size))
    blocks = spatial_batch_order(coords, n_batches=n_batches)
    halo = add_spatial_halo(blocks, graph.edge_index)
    retention = core_incident_edge_retention(graph.edge_index, halo)

    save_arg = str(cache_dir) if cache_dir is not None else None
    if graph_mode == "deterministic_topz":
        gene_batches, node_ids, gene_dic = deterministic_topz_batch_select(
            rna_matrix, blocks, k=topz_k, save_path=save_arg
        )
    else:
        gene_batches, node_ids, gene_dic = spatial_batch_select_whole(
            rna_matrix,
            coords,
            cell_size=cell_size,
            save_path=save_arg or "processed_data_spatial_subset",
        )
    selection = {int(cell): [int(g) for g in info["g"]] for cell, info in gene_dic.items()}
    if len(gene_batches) != len(halo):
        raise RuntimeError("gene batches and spatial halo batches disagree on batch count")
    batches: list[dict[str, Any]] = []
    for gene_batch, halo_batch in zip(gene_batches, halo):
        if [int(c) for c in gene_batch["cell_index"]] != list(halo_batch["core_index"]):
            raise RuntimeError("core cell index order mismatch between gene and halo batches")
        batches.append(
            {
                "gene_index": gene_batch["gene_index"],
                "cell_index": halo_batch["cell_index"],
                "core_index": halo_batch["core_index"],
            }
        )
    return batches, node_ids, selection, graph, retention


def compute_graph_contract(
    rna_matrix: Any,
    batches: Sequence[dict[str, Any]],
    selection: dict[int, list[int]],
    *,
    graph_mode: str,
    topz_k: int,
    edge_source: str,
    kl_target: str,
    gene_input: str,
) -> dict[str, Any]:
    """Summarize exact spot-gene edge counts fed into HGT forward passes."""
    csr = rna_matrix.tocsr()
    if edge_source == "per_spot_selection":
        edge_counts = [selection_edge_count(selection, batch) for batch in batches]
        core_edge_counts = [
            selection_edge_count(
                selection, {"gene_index": batch["gene_index"], "cell_index": batch["core_index"]}
            )
            for batch in batches
        ]
    else:
        edge_counts = [
            int((csr[batch["gene_index"], :][:, batch["cell_index"]] > 0).sum()) for batch in batches
        ]
        core_edge_counts = [
            int((csr[batch["gene_index"], :][:, batch["core_index"]] > 0).sum()) for batch in batches
        ]
    per_spot = [count / max(len(batch["cell_index"]), 1) for count, batch in zip(edge_counts, batches)]
    per_core = [count / max(len(batch["core_index"]), 1) for count, batch in zip(core_edge_counts, batches)]
    return {
        "gene_selection": graph_mode,
        "topz_k": int(topz_k) if graph_mode == "deterministic_topz" else None,
        "edge_source": edge_source,
        "kl_target": kl_target,
        "gene_input": gene_input,
        "spot_gene_edges_total": int(sum(edge_counts)),
        "spot_gene_edges_per_spot_median": float(np.median(per_spot)),
        "spot_gene_edges_per_core_spot_median": float(np.median(per_core)),
        "batch_gene_union_median": float(np.median([len(b["gene_index"]) for b in batches])),
    }
