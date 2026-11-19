"""Check 2 CLI: Exact S(N_c, K) partition-merge ceiling and Leiden resolution frontier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_domain_audit.audit.attainability import (  # noqa: E402
    enumerate_partition_merges,
    greedy_rare_partition,
    sweep_leiden_frontier,
)
from spatial_domain_audit.audit.representation import upstream_pca_space  # noqa: E402

DEFAULT_REF = ROOT / "data" / "reference" / "A1_reference.npz"
DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"
DEFAULT_CEILING_OUT = ROOT / "experiments" / "reports" / "rare_f1_merge_ceiling.json"
DEFAULT_FRONTIER_OUT = ROOT / "experiments" / "reports" / "leiden_resolution_frontier.json"
RESOLUTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def _load_truth(args: argparse.Namespace) -> pd.Series:
    if args.reference_npz.exists():
        ref = np.load(args.reference_npz, allow_pickle=False)
        return pd.Series(ref["labels"].astype(str))
    import scanpy as sc

    return sc.read_h5ad(args.h5ad).obs[args.label_key].astype(str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rare-domain endpoint attainability.")
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REF)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--ref-thr", type=float, default=0.05)
    parser.add_argument("--pred-thr", type=float, default=0.045)
    parser.add_argument("--sweep-frontier", action="store_true", help="Also run Leiden sweep (requires --h5ad).")
    parser.add_argument("--ceiling-out", type=Path, default=DEFAULT_CEILING_OUT)
    parser.add_argument("--frontier-out", type=Path, default=DEFAULT_FRONTIER_OUT)
    args = parser.parse_args()

    truth = _load_truth(args)
    n_total = int(truth.size)
    share = truth.value_counts(normalize=True)
    counts = truth.value_counts()
    rare_types = list(share[share < args.ref_thr].index)
    is_rare = truth.isin(rare_types).to_numpy()
    limit = args.pred_thr * n_total

    constructive = greedy_rare_partition(
        counts, rare_types, n_total, target_k=args.target_k, pred_thr=args.pred_thr
    )
    exhaustive = enumerate_partition_merges(
        counts, rare_types, n_total, target_k=args.target_k, pred_thr=args.pred_thr
    )

    observed = []
    if args.runs_dir.is_dir():
        for run_dir in sorted(args.runs_dir.iterdir()):
            metrics_file = run_dir / "metrics.json"
            pred_file = run_dir / "pred.npy"
            if not run_dir.is_dir() or not metrics_file.is_file() or not pred_file.is_file():
                continue
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            sc_block = payload.get("scformer", {})
            pred = np.load(pred_file)
            sizes = np.bincount(pred)
            nonzero = sizes[sizes > 0]
            small_ids = np.where((sizes > 0) & (sizes < limit))[0]
            small_rare_share = None
            small_spots = None
            if small_ids.size:
                mask = np.isin(pred, small_ids)
                small_spots = int(mask.sum())
                if small_spots:
                    small_rare_share = float(is_rare[mask].mean())
            observed.append(
                {
                    "run_id": run_dir.name,
                    "rare_f1": sc_block.get("rare_f1"),
                    "n_clusters": sc_block.get("n_clusters"),
                    "min_cluster_share": float(nonzero.min() / nonzero.sum()),
                    "n_small_spots": small_spots,
                    "small_rare_share": small_rare_share,
                    "baseline_rare_share": float(is_rare.mean()),
                }
            )

    ceiling_report = {
        "n_total": n_total,
        "n_true_rare": int(is_rare.sum()),
        "rare_types": {str(t): int(counts[t]) for t in rare_types},
        "pred_rare_limit_spots": float(limit),
        "target_k": args.target_k,
        "constructive_bound": constructive,
        "exhaustive_enumeration": exhaustive,
        "observed_runs": observed,
        "summary": {
            "max_rare_f1_at_target_k": max(constructive["rare_f1"], exhaustive["best_rare_f1"]),
            "attainable_under_k_constraint": bool(
                max(constructive["rare_f1"], exhaustive["best_rare_f1"]) > 0.0
            ),
        },
    }
    args.ceiling_out.parent.mkdir(parents=True, exist_ok=True)
    args.ceiling_out.write_text(json.dumps(ceiling_report, indent=2), encoding="utf-8")
    print(
        f"[ceiling] K={args.target_k} across S({len(counts)},{args.target_k})="
        f"{exhaustive['n_partitions']} merges -> max rare-F1 = "
        f"{ceiling_report['summary']['max_rare_f1_at_target_k']:.4f}"
    )

    if args.sweep_frontier:
        import scanpy as sc

        adata = sc.read_h5ad(args.h5ad)
        space = upstream_pca_space(adata, n_pcs=40, seed=0)
        rows = sweep_leiden_frontier(
            space,
            adata.obs[args.label_key].astype(str).to_numpy(),
            RESOLUTIONS,
            n_neighbors=10,
            seed=0,
            ref_thr=args.ref_thr,
            pred_thr=args.pred_thr,
        )
        frame = pd.DataFrame(rows)
        best_rare = frame.loc[frame["rare_f1"].idxmax()].to_dict()
        best_ari = frame.loc[frame["ari"].idxmax()].to_dict()
        frontier_report = {
            "n_total": n_total,
            "rare_types": rare_types,
            "baseline_rare_share": float(is_rare.mean()),
            "pred_rare_limit_spots": float(limit),
            "n_neighbors": 10,
            "representation": "upstream_pca_40",
            "rows": rows,
            "best_rare_f1": best_rare,
            "best_ari": best_ari,
        }
        args.frontier_out.parent.mkdir(parents=True, exist_ok=True)
        args.frontier_out.write_text(json.dumps(frontier_report, indent=2), encoding="utf-8")
        print(
            f"[frontier] peak ARI={best_ari['ari']:.4f} (res={best_ari['resolution']}) vs. "
            f"peak rare-F1={best_rare['rare_f1']:.4f} (res={best_rare['resolution']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
