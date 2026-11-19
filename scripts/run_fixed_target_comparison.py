"""Fixed pseudo-label runner comparing expression-only (A) vs. spatial relation (B)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "scFormer"
sys.path[:0] = [str(ROOT / "src"), str(UPSTREAM)]

from spatial_domain_audit.audit.metrics import (  # noqa: E402
    binary_rare_scores,
    cluster_summary_table,
    per_rare_class_capture,
)
from spatial_domain_audit.training.common import (  # noqa: E402
    DEFAULT_HP,
    UPSTREAM_N_NEIGHBORS,
    compute_graph_contract,
    file_sha256,
    prepare_batches_and_graph,
    resolve_default_resolution,
    seed_all,
    upstream_leiden_labels,
)
from spatial_domain_audit.training.fixed_target_trainer import (  # noqa: E402
    FixedTargetConfig,
    FixedTargetTrainer,
)

DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-target expression vs. spatial comparison.")
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    parser.add_argument("--cell-size", type=int, default=DEFAULT_HP["cell_size"])
    parser.add_argument("--pseudo-resolution", type=float, default=None)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--graph", choices=["upstream_sampled", "deterministic_topz"], default="upstream_sampled")
    parser.add_argument("--topz-k", type=int, default=20)
    parser.add_argument("--kl-target", choices=["counts", "normalized_x"], default="counts")
    parser.add_argument("--gene-input", choices=["counts", "standardized_z"], default="counts")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--protocol-file", type=Path, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import scanpy as sc
    import torch
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

    protocol_snapshot = None
    if args.protocol_file is not None:
        protocol_path = args.protocol_file if args.protocol_file.is_absolute() else ROOT / args.protocol_file
        if not protocol_path.exists():
            raise SystemExit(f"protocol file not found: {protocol_path}")
        protocol_snapshot = {
            "name": json.loads(protocol_path.read_text(encoding="utf-8"))["protocol"]["name"],
            "path": str(protocol_path.relative_to(ROOT)) if protocol_path.is_relative_to(ROOT) else protocol_path.name,
            "sha256": file_sha256(protocol_path),
            "argv": list(sys.argv),
        }

    spatial_relation = args.arm == "B"
    suffix = f"_k{args.spatial_k}" if spatial_relation else ""
    if args.pseudo_resolution is not None:
        suffix += f"_pr{args.pseudo_resolution:g}"
    if args.graph == "deterministic_topz":
        suffix += f"_topz{args.topz_k}"
    if args.kl_target == "normalized_x":
        suffix += "_klX"
    if args.gene_input == "standardized_z":
        suffix += "_gz"
    run_id = f"stage1_{args.arm}{suffix}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = args.runs_root / run_id
    (out / "graph_preprocess").mkdir(parents=True, exist_ok=True)

    seed_all(args.seed, deterministic=args.deterministic)
    started = time.time()

    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    adata.X = csr_matrix(adata.X)
    rna_matrix = adata.X.T
    n_cells = int(adata.n_obs)

    leiden = upstream_leiden_labels(rna_matrix, args.pseudo_resolution)
    ari_leiden = float(adjusted_rand_score(truth, leiden))
    nmi_leiden = float(normalized_mutual_info_score(truth, leiden))

    batches, node_ids, selection, graph, retention = prepare_batches_and_graph(
        rna_matrix,
        coords,
        spatial_k=args.spatial_k,
        cell_size=args.cell_size,
        graph_mode=args.graph,
        topz_k=args.topz_k,
        cache_dir=out / "graph_preprocess",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    edge_source = "per_spot_selection" if args.graph == "deterministic_topz" else "batch_nonzero"
    config = FixedTargetConfig(
        n_hidden=DEFAULT_HP["n_hidden"],
        n_heads=DEFAULT_HP["n_heads"],
        n_layers=DEFAULT_HP["n_layers"],
        labsm=DEFAULT_HP["labsm"],
        lr=DEFAULT_HP["lr"],
        weight_decay=DEFAULT_HP["weight_decay"],
        epochs=args.epochs,
        spatial_relation=spatial_relation,
        kl_target=args.kl_target,
        gene_input=args.gene_input,
        edge_source=edge_source,
    )
    model = FixedTargetTrainer(
        rna_matrix,
        batches,
        leiden,
        config,
        device,
        spatial_edge_index=graph.edge_index if spatial_relation else None,
        selection=selection,
    )
    graph_contract = compute_graph_contract(
        rna_matrix,
        batches,
        selection,
        graph_mode=args.graph,
        topz_k=args.topz_k,
        edge_source=edge_source,
        kl_target=config.kl_target,
        gene_input=config.gene_input,
    )
    edge_stats = model.spatial_edge_stats()
    history = model.train()
    train_sec = time.time() - started
    torch.save(model.gnn.state_dict(), out / "node_model.pth")

    result = model.predict()
    pred_labels = result["pred_label"]
    np.save(out / "pred.npy", pred_labels)
    np.save(out / "cell_embedding.npy", result["cell_embedding"])
    np.save(out / "leiden_pseudolabels.npy", np.asarray(leiden))
    np.save(out / "Node_Ids.npy", node_ids)

    ari = float(adjusted_rand_score(truth, pred_labels))
    nmi = float(normalized_mutual_info_score(truth, pred_labels))
    ari_vs_pseudo = float(adjusted_rand_score(leiden, pred_labels))
    cluster_summary_table(pred_labels, truth).to_csv(out / "cluster_sizes.csv", index=False)
    rare = binary_rare_scores(pred_labels, truth)
    rare_leiden = binary_rare_scores(np.asarray(leiden), truth)
    per_type = per_rare_class_capture(pred_labels, truth)
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

    payload = {
        "run_id": run_id,
        "protocol": protocol_snapshot,
        "stage": "1",
        "arm": args.arm,
        "experiment": (
            "A (expression only, spatial tile batching)"
            if args.arm == "A"
            else f"B (expression + unweighted spatial kNN k={args.spatial_k})"
        ),
        "dataset": args.h5ad.name,
        "h5ad_sha256": file_sha256(args.h5ad),
        "label_key": args.label_key,
        "n_spots": n_cells,
        "n_genes": int(adata.shape[1]),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "hyperparams": {**DEFAULT_HP, "epochs": args.epochs, "cell_size": args.cell_size},
        "graph_contract": graph_contract,
        "batching": {
            "sampler": "spatial_median_bisection_with_one_hop_halo",
            "n_batches": len(batches),
            "core_sizes": {
                "min": int(min(len(b["core_index"]) for b in batches)),
                "max": int(max(len(b["core_index"]) for b in batches)),
            },
            "batch_sizes_with_halo": {
                "min": int(min(len(b["cell_index"]) for b in batches)),
                "max": int(max(len(b["cell_index"]) for b in batches)),
            },
        },
        "spatial": {
            "relation_enabled": spatial_relation,
            "num_relations": config.num_relations,
            "mode": "knn",
            "k": args.spatial_k,
            "weighted": False,
            "n_directed_edges": int(retention["n_edges"]),
            "core_incident_retention": retention["retention"],
            **edge_stats,
        },
        "scformer": {
            "ari": ari,
            "nmi": nmi,
            "n_clusters": int(len(set(pred_labels.tolist()))),
            "ari_vs_pseudo": ari_vs_pseudo,
            **rare,
        },
        "rare_per_type": per_type,
        "leiden_baseline": {
            "ari": ari_leiden,
            "nmi": nmi_leiden,
            "n_clusters": len(set(leiden)),
            "resolution": (
                float(args.pseudo_resolution)
                if args.pseudo_resolution is not None
                else resolve_default_resolution(n_cells)
            ),
            "resolution_source": (
                "cli --pseudo-resolution"
                if args.pseudo_resolution is not None
                else "upstream segment_function"
            ),
            "n_neighbors": UPSTREAM_N_NEIGHBORS,
            **rare_leiden,
        },
        "truth_distribution": {str(k): int(v) for k, v in truth.value_counts().items()},
        "history_tail": history[-5:],
        "runtime_sec": round(time.time() - started, 1),
        "train_sec": round(train_sec, 1),
        "peak_gpu_mem_gb": round(peak, 3),
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[complete] {run_id}: ARI={ari:.4f} ARI_vs_pseudo={ari_vs_pseudo:.4f} rare_f1={rare['rare_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
