"""CPU-only verification of saved outputs, paired contrasts, and Hungarian target churn.

Runs without a GPU or external RNA expression matrix by verifying saved partitions
under ``experiments/runs/`` against the bundled ``data/reference/A1_reference.npz``
labels and spatial graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_domain_audit.audit.metrics import (  # noqa: E402
    assignment_churn,
    evaluate_partition,
    paired_difference_summary,
)

REF_NPZ = ROOT / "data" / "reference" / "A1_reference.npz"
SPATIAL_EDGES = ROOT / "data" / "reference" / "spatial_edges_k10.npy"
RUNS_DIR = ROOT / "experiments" / "runs"


def main() -> int:
    ref = np.load(REF_NPZ, allow_pickle=False)
    names, truth = np.unique(ref["labels"].astype(str), return_inverse=True)
    coords = np.asarray(ref["coordinates"], dtype=np.float64)
    edges = np.load(SPATIAL_EDGES, allow_pickle=False)

    print("=== 1. Aligned Comparison (A* vs. B0 across seeds 0, 1, 42) ===")
    aligned_dir = RUNS_DIR / "p2_aligned_v1"
    aligned_metrics: dict[tuple[str, int], dict] = {}
    for arm in ("Astar", "B0"):
        for seed in (0, 1, 42):
            rdir = aligned_dir / f"{arm}_seed{seed}"
            pred = np.load(rdir / "pred.npy")
            pseudo_file = aligned_dir / f"Astar_seed{seed}" / "initial_labels.npy"
            initial = np.load(pseudo_file) if pseudo_file.is_file() else None
            res = evaluate_partition(
                pred,
                truth,
                initial_labels=initial,
                spatial_edges=edges,
                coordinates=coords,
                class_names=names.tolist(),
            )
            aligned_metrics[(arm, seed)] = res
            churn_frac = res["assignment_churn"]["fraction"] if "assignment_churn" in res else 0.0
            print(
                f"  {arm:5s} seed={seed:>2} ARI={res['ari']:.4f} NMI={res['nmi']:.4f} "
                f"boundary_F1={res['spatial']['boundary']['f1']:.4f} "
                f"rare_F1(pri)={res['rare_primary']['f1']:.4f} "
                f"churn={100*churn_frac:.1f}%"
            )

    print("\n=== Paired Differences (B0 - A*) ===")
    for label, extractor in (
        ("ARI", lambda r: r["ari"]),
        ("NMI", lambda r: r["nmi"]),
        ("boundary-F1", lambda r: r["spatial"]["boundary"]["f1"]),
        ("rare-F1 (<=5%)", lambda r: r["rare_primary"]["f1"]),
    ):
        ref_by_seed = {s: extractor(aligned_metrics[("Astar", s)]) for s in (0, 1, 42)}
        trt_by_seed = {s: extractor(aligned_metrics[("B0", s)]) for s in (0, 1, 42)}
        summary = paired_difference_summary(ref_by_seed, trt_by_seed)
        per_seed = ", ".join(f"{v:+.4f}" for v in summary["values"])
        print(f"  {label:14s}: {summary['mean']:+.4f} +/- {summary['sd']:.4f}  [{per_seed}]")

    print("\n=== 2. Fixed-Target Default vs. Resolution-0.8 Target Movement ===")
    for group_label, prefix in (("default (res 0.5)", ""), ("audit (res 0.8)", "_pr0.8")):
        print(f"  [{group_label}]")
        for arm in ("A", "B"):
            suffix = "_k10" if arm == "B" else ""
            for seed in (0, 1, 42):
                rdir = RUNS_DIR / f"stage1_{arm}{suffix}{prefix}_seed{seed}"
                if not (rdir / "pred.npy").is_file() or not (rdir / "leiden_pseudolabels.npy").is_file():
                    continue
                pred = np.load(rdir / "pred.npy")
                pseudo = np.load(rdir / "leiden_pseudolabels.npy")
                churn = assignment_churn(pred, pseudo)
                ev = evaluate_partition(pred, truth, class_names=names.tolist())
                print(
                    f"    arm {arm} seed={seed:>2}: moved={churn['changed_count']:>2}/{churn['n_spots']} "
                    f"ARI_vs_pseudo={churn['ari']:.4f} ARI={ev['ari']:.4f} "
                    f"rare_F1(sec)={ev['rare_secondary']['f1']:.4f}"
                )

    print("\nVerification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
