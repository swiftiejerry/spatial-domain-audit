"""Check 3a CLI: Quantify spatial edge survival under uniform vs. tile-and-halo batching."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_domain_audit.graph.spatial_batching import (  # noqa: E402
    add_spatial_halo,
    core_incident_edge_retention,
    intra_batch_edge_retention,
    spatial_batch_order,
)
from spatial_domain_audit.graph.spatial_knn import build_spatial_graph  # noqa: E402

DEFAULT_REF = ROOT / "data" / "reference" / "A1_reference.npz"
DEFAULT_OUT = ROOT / "experiments" / "reports" / "batching_edge_retention.json"


def _random_batches(n_cells: int, cell_size: int, seed: int) -> list[dict[str, list[int]]]:
    rng = np.random.RandomState(seed)
    order = rng.choice(n_cells, size=n_cells, replace=False)
    n_batches = math.ceil(n_cells / cell_size)
    return [
        {"cell_index": [int(c) for c in order[i * cell_size : (i + 1) * cell_size]]}
        for i in range(n_batches)
    ]


def _block_compactness(coords: np.ndarray, blocks: list[np.ndarray]) -> dict[str, float]:
    slide_span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    diameters = [float(np.linalg.norm(coords[b].max(axis=0) - coords[b].min(axis=0))) for b in blocks]
    return {
        "slide_diameter": slide_span,
        "mean_block_diameter": float(np.mean(diameters)),
        "max_block_diameter": float(np.max(diameters)),
        "mean_relative_diameter": float(np.mean(diameters) / slide_span),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit intra-batch spatial edge retention.")
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REF)
    parser.add_argument("--h5ad", type=Path, default=None, help="Optional AnnData file instead of reference npz.")
    parser.add_argument("--cell-size", type=int, default=30)
    parser.add_argument("--k-values", type=int, nargs="+", default=[6, 10, 15])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.h5ad is not None:
        import anndata as ad

        adata = ad.read_h5ad(args.h5ad, backed="r")
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        adata.file.close()
    else:
        coords = np.asarray(np.load(args.reference_npz, allow_pickle=False)["coordinates"], dtype=np.float64)

    n_cells = int(coords.shape[0])
    n_batches = math.ceil(n_cells / args.cell_size)
    blocks = spatial_batch_order(coords, n_batches=n_batches)
    spatial_batches = [{"cell_index": [int(c) for c in block]} for block in blocks]
    sizes = [len(b["cell_index"]) for b in spatial_batches]

    covered = sorted(int(c) for b in spatial_batches for c in b["cell_index"])
    if covered != list(range(n_cells)):
        raise SystemExit("spatial batching failed exact cell coverage check")

    report: dict[str, object] = {
        "n_cells": n_cells,
        "cell_size": args.cell_size,
        "n_batches": n_batches,
        "spatial_batch_sizes": {"min": int(min(sizes)), "max": int(max(sizes)), "sum": int(sum(sizes))},
        "compactness": _block_compactness(coords, blocks),
        "per_k": {},
    }

    for k in args.k_values:
        graph = build_spatial_graph(coords, mode="knn", k=k, weighted=False)
        spatial_ret = intra_batch_edge_retention(graph.edge_index, spatial_batches)
        random_rets = [
            intra_batch_edge_retention(graph.edge_index, _random_batches(n_cells, args.cell_size, s))["retention"]
            for s in args.seeds
        ]
        halo_batches = add_spatial_halo(blocks, graph.edge_index)
        halo_ret = core_incident_edge_retention(graph.edge_index, halo_batches)
        halo_sizes = [len(b["cell_index"]) for b in halo_batches]
        mean_rand = float(np.mean(random_rets))
        report["per_k"][str(k)] = {
            "n_directed_edges": spatial_ret["n_edges"],
            "spatial_batching_retention": spatial_ret["retention"],
            "spatial_batching_retained_edges": spatial_ret["n_retained"],
            "random_batching_retention_mean": mean_rand,
            "random_batching_retention_per_seed": {str(s): float(v) for s, v in zip(args.seeds, random_rets)},
            "improvement_factor": float(spatial_ret["retention"] / mean_rand) if mean_rand > 0 else None,
            "halo_retention": halo_ret["retention"],
            "halo_retained_edges": halo_ret["n_retained"],
            "halo_batch_size_min": int(min(halo_sizes)),
            "halo_batch_size_mean": float(np.mean(halo_sizes)),
            "halo_batch_size_max": int(max(halo_sizes)),
        }
        entry = report["per_k"][str(k)]
        print(
            f"k={k:>2} edges={entry['n_directed_edges']:>6} "
            f"uniform={entry['random_batching_retention_mean']:.4f} "
            f"bisection={entry['spatial_batching_retention']:.4f} "
            f"bisection+halo={entry['halo_retention']:.4f} "
            f"(effective batch {entry['halo_batch_size_min']}..{entry['halo_batch_size_max']})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
