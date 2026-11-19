"""Upstream scFormer expression-only baseline runner (no spatial coordinates)."""

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

from spatial_domain_audit.audit.metrics import binary_rare_scores, cluster_summary_table  # noqa: E402
from spatial_domain_audit.training.common import DEFAULT_HP, seed_all  # noqa: E402

DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run expression-only upstream scFormer baseline.")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    parser.add_argument("--cell-size", type=int, default=DEFAULT_HP["cell_size"])
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--tag", default="")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import scanpy as sc
    import torch
    from scipy.sparse import csr_matrix
    from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

    from scformer.model import NDR_2, pred
    from scformer.utils import batch_select_whole, initial_clustering

    run_id = f"stage0_A0_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = args.runs_root / run_id
    (out / "graph_preprocess").mkdir(parents=True, exist_ok=True)

    seed_all(args.seed, deterministic=args.deterministic)
    started = time.time()

    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    adata.X = csr_matrix(adata.X)
    rna_matrix = adata.X.T

    leiden = [int(x) for x in initial_clustering(rna_matrix)]
    ari_leiden = float(adjusted_rand_score(truth, leiden))
    nmi_leiden = float(normalized_mutual_info_score(truth, leiden))

    indices, node_ids, _ = batch_select_whole(
        rna_matrix, cell_size=args.cell_size, save_path=str(out / "graph_preprocess")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = NDR_2(
        rna_matrix,
        indices,
        leiden,
        n_hid=DEFAULT_HP["n_hidden"],
        n_heads=DEFAULT_HP["n_heads"],
        n_layers=DEFAULT_HP["n_layers"],
        labsm=DEFAULT_HP["labsm"],
        lr=DEFAULT_HP["lr"],
        wd=DEFAULT_HP["weight_decay"],
        device=device,
        num_types=2,
        num_relations=2,
        epochs=args.epochs,
    )
    gnn, _, _, _ = model.train_model(n_batch=len(indices))
    train_sec = time.time() - started
    model.save_model(str(out / "node_model.pth"))

    results = pred(rna_matrix, gnn=gnn, indices=indices, device=device)
    order = np.argsort(node_ids)
    pred_labels = np.asarray(results["pred_label"])[order]
    embedding = np.asarray(results["cell_embedding"])[order]

    np.save(out / "Node_Ids.npy", node_ids)
    np.save(out / "pred.npy", pred_labels)
    np.save(out / "cell_embedding.npy", embedding)
    np.save(out / "leiden_pseudolabels.npy", np.asarray(leiden))

    ari = float(adjusted_rand_score(truth, pred_labels))
    nmi = float(normalized_mutual_info_score(truth, pred_labels))
    tbl = cluster_summary_table(pred_labels, truth)
    tbl.to_csv(out / "cluster_sizes.csv", index=False)
    rare = binary_rare_scores(pred_labels, truth)
    rare_leiden = binary_rare_scores(np.asarray(leiden), truth)
    peak = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

    payload = {
        "run_id": run_id,
        "stage": "0",
        "experiment": "A0 (upstream expression-only baseline, no spatial coordinates)",
        "h5ad": str(args.h5ad),
        "label_key": args.label_key,
        "n_spots": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "use_spatial": False,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "hyperparams": {**DEFAULT_HP, "epochs": args.epochs, "cell_size": args.cell_size},
        "n_batch": len(indices),
        "scformer": {"ari": ari, "nmi": nmi, "n_clusters": int(len(set(pred_labels))), **rare},
        "leiden_baseline": {
            "ari": ari_leiden,
            "nmi": nmi_leiden,
            "n_clusters": len(set(leiden)),
            **rare_leiden,
        },
        "sanity_floor_passed": bool(ari >= ari_leiden),
        "truth_distribution": {str(k): int(v) for k, v in truth.value_counts().items()},
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
