"""Aligned deterministic input warm-up runner (no-coordinate A0 vs. tile+halo A* and B0)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "scFormer"
sys.path[:0] = [str(ROOT / "src"), str(UPSTREAM)]

from spatial_domain_audit.audit.representation import (  # noqa: E402
    cosine_class_separation,
    cosine_knn_purity,
    effective_rank,
    leiden_to_target_k,
)
from spatial_domain_audit.graph.spatial_batching import (  # noqa: E402
    add_spatial_halo,
    core_incident_edge_retention,
    spatial_batch_order,
)
from spatial_domain_audit.graph.spatial_knn import build_spatial_graph  # noqa: E402
from spatial_domain_audit.graph.topz_normalized import (  # noqa: E402
    deterministic_topz_batch_select,
    selection_edge_count,
)
from spatial_domain_audit.training.common import seed_all  # noqa: E402
from spatial_domain_audit.training.fixed_target_trainer import (  # noqa: E402
    FixedTargetConfig,
    FixedTargetTrainer,
)

DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"
DEFAULT_PROTOCOL = ROOT / "configs" / "aligned_warmup_protocol.json"


def _random_blocks(n: int, cell_size: int, seed: int) -> list[np.ndarray]:
    perm = np.random.RandomState(seed).permutation(n).astype(np.int64)
    return [perm[i : i + cell_size] for i in range(0, n, cell_size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aligned 30-epoch reconstruction warm-up.")
    parser.add_argument("--arm", choices=["A0", "Astar", "B0"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cell-size", type=int, default=30)
    parser.add_argument("--spatial-k", type=int, default=10)
    parser.add_argument("--topz-k", type=int, default=20)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import scanpy as sc
    import torch
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

    from scformer.utils import initial_clustering

    if not args.protocol_file.exists():
        raise SystemExit(f"missing protocol: {args.protocol_file}")
    protocol = json.loads(args.protocol_file.read_text(encoding="utf-8"))
    if args.dry_run:
        base = protocol.get("input_base", {})
        sampling = protocol.get("sampling", {})
        missing = []
        if base.get("topz_k") != args.topz_k:
            missing.append("input_base.topz_k")
        if base.get("gene_node_input") != "gene_wise_zscore_over_all_spots":
            missing.append("input_base.gene_node_input")
        if base.get("kl_target") != "X":
            missing.append("input_base.kl_target")
        if args.arm == "A0" and sampling.get("A0", {}).get("uses_coordinates") is not False:
            missing.append("sampling.A0.uses_coordinates")
        if args.arm in ("Astar", "B0") and sampling.get("A_star", {}).get("spatial_tile_sampling") is not True:
            missing.append("sampling.A_star.spatial_tile_sampling")
        print(
            json.dumps(
                {
                    "status": "PASS" if not missing else "FAIL",
                    "arm": args.arm,
                    "missing_protocol_fields": missing,
                    "sampling": (
                        "random_permutation_no_coordinate"
                        if args.arm == "A0"
                        else "spatial_bisection_one_hop_halo"
                    ),
                    "relation_enabled": args.arm == "B0",
                    "epochs": args.epochs,
                },
                indent=2,
            )
        )
        return 0 if not missing else 1

    if not args.h5ad.exists():
        raise SystemExit(f"missing h5ad: {args.h5ad}")

    seed_all(args.seed, deterministic=True)
    started = time.time()
    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    adata.X = csr_matrix(adata.X)
    rna = adata.X.T
    n = int(adata.n_obs)
    pseudo = [int(x) for x in initial_clustering(rna)]

    graph = None
    retention = None
    if args.arm == "A0":
        blocks = _random_blocks(n, args.cell_size, args.seed)
        halo = None
        sampler = "seeded_random_permutation_no_coordinate"
    else:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        graph = build_spatial_graph(coords, mode="knn", k=args.spatial_k, weighted=False)
        blocks = spatial_batch_order(coords, n_batches=math.ceil(n / args.cell_size))
        halo = add_spatial_halo(blocks, graph.edge_index)
        retention = core_incident_edge_retention(graph.edge_index, halo)
        sampler = "spatial_median_bisection_with_one_hop_halo"

    out = args.runs_root / f"p1_{args.arm}_seed{args.seed}_topz{args.topz_k}_klX_gz"
    cache = out / "graph_preprocess"
    genes, node_ids, gene_dic = deterministic_topz_batch_select(
        rna, blocks, k=args.topz_k, save_path=str(cache)
    )
    batches: list[dict] = []
    if halo is None:
        for gb in genes:
            cells = [int(c) for c in gb["cell_index"]]
            batches.append({"gene_index": gb["gene_index"], "cell_index": cells, "core_index": cells})
    else:
        for gb, hb in zip(genes, halo):
            if [int(c) for c in gb["cell_index"]] != list(hb["core_index"]):
                raise RuntimeError("core order mismatch")
            batches.append(
                {
                    "gene_index": gb["gene_index"],
                    "cell_index": hb["cell_index"],
                    "core_index": hb["core_index"],
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    spatial_relation = args.arm == "B0"
    selection_map = {int(c): v["g"] for c, v in gene_dic.items()}
    config = FixedTargetConfig(
        n_hidden=104,
        n_heads=8,
        n_layers=3,
        labsm=0.1,
        lr=5e-4,
        weight_decay=0.1,
        epochs=args.epochs,
        spatial_relation=spatial_relation,
        kl_target="normalized_x",
        gene_input="standardized_z",
        edge_source="per_spot_selection",
        kl_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedTargetTrainer(
        rna,
        batches,
        pseudo,
        config,
        device,
        spatial_edge_index=graph.edge_index if spatial_relation and graph is not None else None,
        selection=selection_map,
    )
    relation_stats = model.spatial_edge_stats()
    history = model.train()
    model.gnn.eval()
    result = model.predict()
    emb = result["cell_embedding"].astype(np.float32)

    np.save(out / "warmup_embedding_ep30.npy", emb)
    np.save(out / "cell_embedding.npy", emb)
    np.save(out / "pred.npy", result["pred_label"])
    np.save(out / "leiden_pseudolabels.npy", np.asarray(pseudo))
    np.save(out / "Node_Ids.npy", node_ids)

    _, truth_codes = np.unique(truth.to_numpy(), return_inverse=True)
    labels = leiden_to_target_k(emb, target_k=5, seed=args.seed)
    knn = cosine_knn_purity(emb, truth_codes, k=10)
    quality = {
        "embedding_dim": int(emb.shape[1]),
        "leiden_k": int(np.unique(labels).size),
        "ari": float(adjusted_rand_score(truth_codes, labels)),
        "nmi": float(normalized_mutual_info_score(truth_codes, labels)),
        "knn_purity": knn["knn_purity"],
        "chance": knn["chance"],
        "effective_rank": effective_rank(emb),
        "class_separation": cosine_class_separation(emb, truth_codes),
    }
    core_edges = [
        selection_edge_count(
            selection_map, {"gene_index": b["gene_index"], "cell_index": b["core_index"]}
        )
        for b in batches
    ]
    payload = {
        "run_id": out.name,
        "stage": "paired_warmup",
        "arm": args.arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "protocol": {
            "path": args.protocol_file.name,
            "argv": sys.argv,
        },
        "common_input": {
            "gene_graph": "deterministic_topz",
            "topz_k": args.topz_k,
            "gene_input": "standardized_z",
            "kl_target": "normalized_x",
            "spatial_relation": spatial_relation,
        },
        "sampling": {
            "sampler": sampler,
            "n_batches": len(batches),
            "core_size_min": min(len(b["core_index"]) for b in batches),
            "core_size_max": max(len(b["core_index"]) for b in batches),
            "halo_size_median": float(np.median([len(b["cell_index"]) - len(b["core_index"]) for b in batches])),
        },
        "spatial": {
            "read": args.arm in ("Astar", "B0"),
            "k": args.spatial_k if args.arm in ("Astar", "B0") else None,
            "relation_enabled": spatial_relation,
            "weighted": False,
            "n_directed_edges": int(retention["n_edges"]) if retention else 0,
            "core_incident_retention": retention["retention"] if retention else None,
            "edges_consumed_by_hgt": relation_stats["total_spatial_edges_in_batches"],
        },
        "graph_contract": {
            "spot_gene_edges_per_core_spot_median": float(
                np.median([x / max(len(b["core_index"]), 1) for x, b in zip(core_edges, batches)])
            ),
            "edge_source": "per_spot_selection",
        },
        "warmup_quality": quality,
        "history_tail": history[-5:],
        "runtime_sec": round(time.time() - started, 1),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
