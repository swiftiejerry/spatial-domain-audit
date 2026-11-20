"""Independently recount saved A1 GraphST predictions on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "src"

import sys

sys.path.insert(0, str(sys_path))

from spatial_domain_audit.audit.metrics import evaluate_partition  # noqa: E402


def verify() -> None:
    reference = np.load(ROOT / "data" / "reference" / "A1_reference.npz", allow_pickle=False)
    _, truth = np.unique(reference["labels"].astype(str), return_inverse=True)
    expected_seeds = (0, 1, 42)

    print("=== External A1 GraphST CPU recount ===")
    for seed in expected_seeds:
        run_dir = ROOT / "experiments" / "runs" / "external_A1_GraphST" / f"seed{seed}"
        prediction = np.load(run_dir / "pred.npy", allow_pickle=False).reshape(-1)
        saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        result = evaluate_partition(prediction, truth)
        values = {
            "ari": result["ari"],
            "nmi": result["nmi"],
            "rare_f1_primary": result["rare_primary"]["f1"],
            "n_spots": result["n_spots"],
            "n_clusters": result["n_clusters"],
        }
        saved_values = {
            "ari": saved["ari"],
            "nmi": saved["nmi"],
            "rare_f1_primary": saved["evaluation"]["rare_primary"]["f1"],
            "n_spots": saved["evaluation"]["n_spots"],
            "n_clusters": saved["n_clusters"],
        }
        for key, value in values.items():
            if isinstance(value, float):
                if not np.isclose(value, saved_values[key], atol=1e-12, rtol=0):
                    raise AssertionError(f"seed {seed} {key} differs from saved metrics")
            elif value != saved_values[key]:
                raise AssertionError(f"seed {seed} {key} differs from saved metrics")
        print(
            f"seed={seed:>2} n={values['n_spots']} K={values['n_clusters']} "
            f"ARI={values['ari']:.4f} NMI={values['nmi']:.4f} "
            f"rare-F1<=5%={values['rare_f1_primary']:.4f} saved_metrics_match=True"
        )


def main() -> int:
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
