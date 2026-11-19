"""Check 4b CLI: Full-ranking Wilcoxon DE marker screen and spatial localization check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_domain_audit.audit.marker_support import screen_domain_markers  # noqa: E402

DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_OUT = ROOT / "experiments" / "reports" / "marker_signal_screen.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen rare reference domains for localized markers.")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--rare-threshold", type=float, default=0.05)
    parser.add_argument("--min-pts-in", type=float, default=0.60)
    parser.add_argument("--min-ratio", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import scanpy as sc

    adata = sc.read_h5ad(args.h5ad)
    domains = screen_domain_markers(
        adata,
        label_key=args.label_key,
        rare_threshold=args.rare_threshold,
        min_pts_in=args.min_pts_in,
        min_ratio=args.min_ratio,
    )
    report = {
        "description": "Full-ranking Wilcoxon marker screen across all genes (normalize_total 1e4 + log1p).",
        "criteria": {
            "pvals_adj": "< 0.05",
            "pts_in": f">= {args.min_pts_in:.2f}",
            "pts_in_over_pts_rest": f">= {args.min_ratio:.1f}",
        },
        "domains": domains,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for d in domains:
        print(
            f"{d['domain']:20s} n={d['n_spots']:>3} gene={d['gene']:8s} rank={d['rank']:>3} "
            f"p_adj={d['pvals_adj']:.2e} pts_in={d['pts_in']:.3f} ratio={d['detection_ratio']:.2f} "
            f"top_n={d['top_n_hits']}/{d['n_domain']} tier={d['tier']}"
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
