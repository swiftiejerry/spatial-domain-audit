"""Check 4a / Diagnostic CLI: Evaluate learned spot embedding geometry and k-NN purity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spatial_domain_audit.audit.representation import (  # noqa: E402
    evaluate_representation,
    upstream_pca_space,
)

DEFAULT_H5AD = ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad"
DEFAULT_RUNS = ROOT / "experiments" / "runs"
DEFAULT_OUT = ROOT / "experiments" / "reports" / "representation_quality.json"

TARGET_RUNS = [
    ("warmup_30ep_kl_only", "stage2_C_t0.08_d0.8_seed0_preflight"),
    ("prototype_v2b_50ep", "stage2_C_t0.2_dq0.2_seed0_v2b_preflight"),
    ("prototype_v2a_collapsed_50ep", "stage2_C_t0.2_dq0.2_seed0_v2a_preflight"),
    ("fixed_target_armA_100ep", "stage1_A_seed0"),
    ("fixed_target_armB_k10_100ep", "stage1_B_k10_seed0"),
    ("fixed_target_armA_res0.8_seed0", "stage1_A_pr0.8_seed0"),
    ("fixed_target_armB_k10_res0.8_seed0", "stage1_B_k10_pr0.8_seed0"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit spot embedding representation quality.")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    adata = sc.read_h5ad(args.h5ad)
    truth = adata.obs[args.label_key].astype(str)
    _, codes = np.unique(truth.to_numpy(), return_inverse=True)

    entries = [
        evaluate_representation(
            "upstream_expression_pca40",
            upstream_pca_space(adata, n_pcs=40, seed=0),
            codes,
            target_k=args.target_k,
            knn_k=args.knn,
            seed=0,
        )
    ]
    for label, run_id in TARGET_RUNS:
        emb_path = args.runs_dir / run_id / "cell_embedding.npy"
        if not emb_path.is_file():
            continue
        entry = evaluate_representation(
            label,
            np.load(emb_path).astype(np.float32),
            codes,
            target_k=args.target_k,
            knn_k=args.knn,
            seed=0,
        )
        entry["run_id"] = run_id
        pseudo_path = args.runs_dir / run_id / "leiden_pseudolabels.npy"
        if pseudo_path.is_file():
            entry["pseudolabel_ari"] = float(adjusted_rand_score(codes, np.load(pseudo_path)))
        entries.append(entry)

    report = {"target_k": args.target_k, "knn": args.knn, "representations": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for e in entries:
        print(
            f"{e['name']:34s} dim={e['dimension']:>4} ARI={e['ari']:+.4f} "
            f"kNN={e['knn_purity']:.4f}(chance {e['chance']:.4f}) "
            f"eff_rank={e['effective_rank']:.2f} sep={e['class_separation']:+.4f}"
        )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
