"""Prototype-refresh dynamic clustering runner (arm C without spatial, arm D with spatial)."""

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
    compute_graph_contract,
    prepare_batches_and_graph,
    resolve_default_resolution,
    seed_all,
    upstream_leiden_labels,
)
from spatial_domain_audit.training.prototype_trainer import (  # noqa: E402
    PrototypeConfig,
    PrototypeTrainer,
)

DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prototype-refresh dynamic clustering arms.")
    parser.add_argument("--arm", choices=["C", "D"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lambda-proto", type=float, default=1.0)
    parser.add_argument("--target-k", type=int, default=None)
    parser.add_argument("--leiden-resolution", type=float, default=None)
    parser.add_argument("--free-k0", action="store_true")
    parser.add_argument("--no-preserve-k", action="store_true")
    parser.add_argument("--no-center-embedding", action="store_true")
    parser.add_argument("--delta-mode", choices=["quantile", "absolute"], default="quantile")
    parser.add_argument("--delta-quantile", type=float, default=0.2)
    parser.add_argument(
        "--confident-only-refresh",
        action="store_true",
        help="Update prototype centroids and labels using only assignments above the active confidence threshold.",
    )
    parser.add_argument("--init-source", choices=["warmup_leiden", "a0_pseudo"], default="warmup_leiden")
    parser.add_argument("--probe-every", type=int, default=0)
    parser.add_argument("--pseudo-resolution", type=float, default=None)
    parser.add_argument("--cell-size", type=int, default=DEFAULT_HP["cell_size"])
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--protocol-file", type=Path, default=None)
    parser.add_argument("--graph", choices=["upstream_sampled", "deterministic_topz"], default="upstream_sampled")
    parser.add_argument("--topz-k", type=int, default=20)
    parser.add_argument("--kl-target", choices=["counts", "normalized_x"], default="counts")
    parser.add_argument("--gene-input", choices=["counts", "standardized_z"], default="counts")
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
            "argv": list(sys.argv),
        }

    spatial_relation = args.arm == "D"
    parts = [f"stage{'3' if spatial_relation else '2'}_{args.arm}"]
    if spatial_relation:
        parts.append(f"k{args.spatial_k}")
    parts.append(f"t{args.tau:g}")
    parts.append(f"dq{args.delta_quantile:g}" if args.delta_mode == "quantile" else f"d{args.delta:g}")
    parts.append(f"seed{args.seed}")
    if args.init_source != "warmup_leiden":
        parts.append("initA0")
    if args.pseudo_resolution is not None:
        parts.append(f"pr{args.pseudo_resolution:g}")
    if args.graph == "deterministic_topz":
        parts.append(f"topz{args.topz_k}")
    if args.confident_only_refresh:
        parts.append("confident-only")
    if args.kl_target == "normalized_x":
        parts.append("klX")
    if args.gene_input == "standardized_z":
        parts.append("gz")
    run_id = "_".join(parts) + (f"_{args.tag}" if args.tag else "")
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
    if args.target_k is not None:
        target_k = args.target_k
    elif args.free_k0:
        target_k = None
    else:
        target_k = len(set(leiden))

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
    config = PrototypeConfig(
        n_hidden=DEFAULT_HP["n_hidden"],
        n_heads=DEFAULT_HP["n_heads"],
        n_layers=DEFAULT_HP["n_layers"],
        labsm=DEFAULT_HP["labsm"],
        lr=DEFAULT_HP["lr"],
        weight_decay=DEFAULT_HP["weight_decay"],
        epochs=args.epochs,
        spatial_relation=spatial_relation,
        warmup_epochs=args.warmup_epochs,
        update_interval=args.update_interval,
        delta=args.delta,
        tau=args.tau,
        lambda_proto=args.lambda_proto,
        target_k=target_k,
        leiden_resolution=args.leiden_resolution,
        seed=args.seed,
        preserve_k=not args.no_preserve_k,
        center_embedding=not args.no_center_embedding,
        delta_mode=args.delta_mode,
        delta_quantile=args.delta_quantile,
        init_source=args.init_source,
        probe_every=args.probe_every,
        probe_dir=str(out),
        kl_target=args.kl_target,
        gene_input=args.gene_input,
        edge_source=edge_source,
        confident_only_refresh=args.confident_only_refresh,
    )
    model = PrototypeTrainer(
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
    if model.initial_assignment is None:
        raise RuntimeError("dynamic clustering did not record warm-up initial assignment")
    initial_labels = np.asarray(model.initial_assignment, dtype=np.int64)
    np.save(out / "pred.npy", pred_labels)
    np.save(out / "warmup_init_labels.npy", initial_labels)
    np.save(out / "cell_embedding.npy", result["cell_embedding"])
    np.save(out / "confidence.npy", result["confidence"])
    np.save(out / "leiden_pseudolabels.npy", np.asarray(leiden))
    np.save(out / "Node_Ids.npy", node_ids)

    ari = float(adjusted_rand_score(truth, pred_labels))
    nmi = float(normalized_mutual_info_score(truth, pred_labels))
    ari_vs_pseudo = float(adjusted_rand_score(leiden, pred_labels))
    cluster_summary_table(pred_labels, truth).to_csv(out / "cluster_sizes.csv", index=False)
    rare = binary_rare_scores(pred_labels, truth)
    rare_leiden = binary_rare_scores(np.asarray(leiden), truth)
    rare_initial = binary_rare_scores(initial_labels, truth)
    ari_initial = float(adjusted_rand_score(truth, initial_labels))
    nmi_initial = float(normalized_mutual_info_score(truth, initial_labels))
    per_type = per_rare_class_capture(pred_labels, truth)
    confidence = result["confidence"]
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

    payload = {
        "run_id": run_id,
        "protocol": protocol_snapshot,
        "stage": "3" if spatial_relation else "2",
        "arm": args.arm,
        "experiment": (
            "D (expression + spatial + prototype refresh)"
            if spatial_relation
            else "C (expression + prototype refresh)"
        ),
        "dataset": args.h5ad.name,
        "label_key": args.label_key,
        "n_spots": n_cells,
        "n_genes": int(adata.shape[1]),
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "hyperparams": {**DEFAULT_HP, "epochs": args.epochs, "cell_size": args.cell_size},
        "graph_contract": graph_contract,
        "dynamic": {
            "warmup_epochs": args.warmup_epochs,
            "update_interval": args.update_interval,
            "delta": args.delta,
            "tau": args.tau,
            "lambda_proto": args.lambda_proto,
            "target_k": target_k,
            "objective": "L_KL + lambda_proto * L_proto (replaces fixed classification and intra-cluster terms)",
            "cluster_init": model.cluster_init,
            "initial_partition": {
                "artifact": "warmup_init_labels.npy",
                "ari": ari_initial,
                "nmi": nmi_initial,
                "n_clusters": int(np.unique(initial_labels).size),
                **rare_initial,
            },
            "n_clusters_used": int(np.unique(pred_labels).size),
            "confidence_mean": float(confidence.mean()),
            "confident_fraction": float(
                (confidence > float(np.quantile(confidence, args.delta_quantile))).mean()
            )
            if args.delta_mode == "quantile"
            else float((confidence > args.delta).mean()),
            "confidence_percentiles": {
                "p10": float(np.quantile(confidence, 0.10)),
                "p50": float(np.quantile(confidence, 0.50)),
                "p90": float(np.quantile(confidence, 0.90)),
            },
            "center_embedding": bool(not args.no_center_embedding),
            "delta_mode": args.delta_mode,
            "delta_quantile": args.delta_quantile,
            "init_source": args.init_source,
            "pseudo_resolution": (
                float(args.pseudo_resolution)
                if args.pseudo_resolution is not None
                else resolve_default_resolution(n_cells)
            ),
            "delta_effective_final": float(model.delta_value),
            "preserve_k": bool(not args.no_preserve_k),
            "confident_only_refresh": bool(args.confident_only_refresh),
            "refresh_trace": getattr(model, "refresh_trace", []),
        },
        "batching": {
            "sampler": "spatial_median_bisection_with_one_hop_halo",
            "n_batches": len(batches),
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
            "n_clusters": int(np.unique(pred_labels).size),
            "ari_vs_pseudo": ari_vs_pseudo,
            **rare,
        },
        "rare_per_type": per_type,
        "leiden_baseline": {
            "ari": ari_leiden,
            "nmi": float(normalized_mutual_info_score(truth, leiden)),
            "n_clusters": len(set(leiden)),
            "resolution": (
                float(args.pseudo_resolution)
                if args.pseudo_resolution is not None
                else resolve_default_resolution(n_cells)
            ),
            **rare_leiden,
        },
        "truth_distribution": {str(k): int(v) for k, v in truth.value_counts().items()},
        "history_tail": history[-8:],
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
    print(f"[complete] {run_id}: ARI={ari:.4f} NMI={nmi:.4f} rare_f1={rare['rare_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
