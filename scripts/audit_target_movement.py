"""Check 1 CLI: Measure target movement and Hungarian alignment churn from fixed pseudo-labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_domain_audit.audit.metrics import align_labels_to_reference, assignment_churn  # noqa: E402

DEFAULT_REF = ROOT / "data" / "reference" / "A1_reference.npz"
DEFAULT_RUNS = ROOT / "experiments" / "runs"
DEFAULT_OUT = ROOT / "experiments" / "reports" / "fixed_target_headroom.json"


def _array_digest(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _small_cluster_summary(pred: np.ndarray, is_rare: np.ndarray, *, pred_thr: float = 0.045) -> dict:
    n = int(pred.size)
    counts = Counter(pred.tolist())
    small_ids = [cid for cid, size in counts.items() if size / n < pred_thr]
    in_small = np.isin(pred, small_ids)
    n_small_spots = int(in_small.sum())
    n_rare_hits = int((is_rare & in_small).sum())
    return {
        "n_small_clusters": len(small_ids),
        "small_cluster_sizes": sorted((counts[c] for c in small_ids), reverse=True),
        "n_spots_in_small_clusters": n_small_spots,
        "n_rare_in_small_clusters": n_rare_hits,
        "rare_share_in_small_clusters": float(n_rare_hits / n_small_spots) if n_small_spots else None,
        "min_cluster_share": min(size / n for size in counts.values()),
    }


def _find_arm_dir(runs_dir: Path, arm: str, resolution: float, seed: int) -> Path | None:
    suffix = "_k10" if arm == "B" else ""
    pr = "" if resolution == 0.5 else f"_pr{resolution:g}"
    candidate = runs_dir / f"stage1_{arm}{suffix}{pr}_seed{seed}"
    return candidate if (candidate / "pred.npy").is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify spot-level movement relative to frozen targets.")
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REF)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--resolution", type=float, default=0.8)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 42])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ref = np.load(args.reference_npz, allow_pickle=False)
    truth = ref["labels"].astype(str)
    n = int(truth.size)
    counts = Counter(truth.tolist())
    rare_types = sorted(name for name, count in counts.items() if count / n < 0.05)
    is_rare = np.isin(truth, rare_types)

    rows = []
    for seed in args.seeds:
        dir_a = _find_arm_dir(args.runs_dir, "A", args.resolution, seed)
        dir_b = _find_arm_dir(args.runs_dir, "B", args.resolution, seed)
        if dir_a is None or dir_b is None:
            rows.append({"seed": seed, "status": "missing_run"})
            continue
        pseudo_a = np.load(dir_a / "leiden_pseudolabels.npy")
        pseudo_b = np.load(dir_b / "leiden_pseudolabels.npy")
        if _array_digest(pseudo_a) != _array_digest(pseudo_b):
            raise SystemExit(f"seed {seed} pseudo-label mismatch between arm A and arm B")
        pred_a = np.load(dir_a / "pred.npy")
        pred_b = np.load(dir_b / "pred.npy")
        churn_a = assignment_churn(pred_a, pseudo_a)
        churn_b = assignment_churn(pred_b, pseudo_b)
        churn_ab = assignment_churn(pred_a, pred_b)
        aligned_b = align_labels_to_reference(pred_b, pseudo_b)
        moved_b = aligned_b != pseudo_b
        rows.append(
            {
                "seed": seed,
                "status": "complete",
                "arm_a": dir_a.name,
                "arm_b": dir_b.name,
                "n_spots": n,
                "arm_a_changed_spots": churn_a["changed_count"],
                "arm_b_changed_spots": churn_b["changed_count"],
                "arm_a_headroom_fraction": churn_a["fraction"],
                "arm_b_headroom_fraction": churn_b["fraction"],
                "between_arm_changed_spots": churn_ab["changed_count"],
                "arm_b_moved_truth_breakdown": dict(Counter(truth[moved_b].tolist()).most_common()),
                "arm_b_moved_rare_count": int(is_rare[moved_b].sum()),
                "arm_a_small_clusters": _small_cluster_summary(pred_a, is_rare),
                "arm_b_small_clusters": _small_cluster_summary(pred_b, is_rare),
            }
        )

    complete = [r for r in rows if r["status"] == "complete"]
    report = {
        "pseudo_resolution": args.resolution,
        "rare_types": rare_types,
        "rare_baseline_share": float(is_rare.mean()),
        "small_cluster_threshold": 0.045,
        "rows": rows,
        "summary": {
            "n_seeds": len(complete),
            "arm_a_max_headroom": max((r["arm_a_headroom_fraction"] for r in complete), default=None),
            "arm_b_max_headroom": max((r["arm_b_headroom_fraction"] for r in complete), default=None),
            "max_between_arms_spots": max((r["between_arm_changed_spots"] for r in complete), default=None),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in complete:
        print(
            f"seed {row['seed']:>2}: armA changed={row['arm_a_changed_spots']} ({row['arm_a_headroom_fraction']:.4%}), "
            f"armB changed={row['arm_b_changed_spots']} ({row['arm_b_headroom_fraction']:.4%}), "
            f"A-vs-B={row['between_arm_changed_spots']}"
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
