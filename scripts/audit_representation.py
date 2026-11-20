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
DEFAULT_ALIGNED_RUNS = DEFAULT_RUNS / "p2_aligned_v1"
DEFAULT_MEAN_ENCODER_RUNS = DEFAULT_RUNS / "mean_encoder_p2_20261119"
DEFAULT_REPRODUCED_WARMUP_RUNS = DEFAULT_RUNS / "representation_audit_20261120"
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

ALIGNED_RUNS = [
    (f"aligned_Astar_seed{seed}", f"Astar_seed{seed}") for seed in (0, 1, 42)
] + [
    (f"aligned_B0_seed{seed}", f"B0_seed{seed}") for seed in (0, 1, 42)
]

REPRODUCED_WARMUP_RUNS = [
    (f"reproduced_Astar_seed{seed}", f"p1_Astar_seed{seed}_topz20_klX_gz")
    for seed in (0, 1, 42)
]

MEAN_ENCODER_RUNS = [
    (f"{arm}_seed{seed}", f"{arm}_seed{seed}")
    for arm in ("Astar_mean", "B0_mean", "B0_mean_perm")
    for seed in (0, 1, 42)
]


def embedding_path(run_dir: Path) -> Path | None:
    for name in ("spot_embedding.npy", "cell_embedding.npy"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def cached_representation(run_dir: Path) -> dict[str, object] | None:
    """Read lightweight representation metrics when a large embedding was omitted."""
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        quality = payload.get("warmup_quality")
        if isinstance(quality, dict):
            return quality

    evaluation_path = run_dir / "evaluation.json"
    if evaluation_path.is_file():
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        quality = payload.get("embedding")
        if isinstance(quality, dict):
            return quality
    return None


def run_seed(run_dir: Path) -> int:
    for filename in ("metrics.json", "config.json"):
        path = run_dir / filename
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "seed" in payload:
                return int(payload["seed"])
    return 0


def run_metadata(run_dir: Path) -> dict[str, object]:
    for filename in ("metrics.json", "config.json"):
        path = run_dir / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = (
            "stage",
            "arm",
            "seed",
            "epochs",
            "common_input",
            "sampling",
            "spatial",
            "encoder",
            "layout",
        )
        return {field: payload[field] for field in fields if field in payload}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit spot embedding representation quality.")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument(
        "--aligned-runs-dir",
        type=Path,
        default=DEFAULT_ALIGNED_RUNS,
        help="Directory containing Astar_seed* and B0_seed* embeddings.",
    )
    parser.add_argument(
        "--mean-encoder-runs-dir",
        type=Path,
        default=DEFAULT_MEAN_ENCODER_RUNS,
        help="Optional directory containing the relation-mean comparison runs.",
    )
    parser.add_argument(
        "--reproduced-warmup-runs-dir",
        type=Path,
        default=DEFAULT_REPRODUCED_WARMUP_RUNS,
        help="Optional directory containing p1_Astar_seed*_topz20_klX_gz runs.",
    )
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

    upstream_entry = evaluate_representation(
            "upstream_expression_pca40",
            upstream_pca_space(adata, n_pcs=40, seed=0),
            codes,
            target_k=args.target_k,
            knn_k=args.knn,
            seed=0,
        )
    upstream_entry["status"] = "computed"
    entries = [upstream_entry]
    excluded_runs: list[dict[str, str]] = []
    aligned_root = args.aligned_runs_dir or args.runs_dir
    runs_to_audit = [
        (label, run_id, args.runs_dir)
        for label, run_id in TARGET_RUNS
    ] + [
        (label, run_id, aligned_root)
        for label, run_id in ALIGNED_RUNS
    ]
    if args.mean_encoder_runs_dir is not None and args.mean_encoder_runs_dir.is_dir():
        runs_to_audit.extend(
            (label, run_id, args.mean_encoder_runs_dir)
            for label, run_id in MEAN_ENCODER_RUNS
        )
    if args.reproduced_warmup_runs_dir is not None and args.reproduced_warmup_runs_dir.is_dir():
        runs_to_audit.extend(
            (label, run_id, args.reproduced_warmup_runs_dir)
            for label, run_id in REPRODUCED_WARMUP_RUNS
        )

    for label, run_id, run_root in runs_to_audit:
        run_dir = run_root / run_id
        emb_path = embedding_path(run_dir)
        if emb_path is None:
            cached = cached_representation(run_dir)
            if cached is not None:
                entry = dict(cached)
                entry.update(
                    {
                        "name": label,
                        "run_id": run_id,
                        "status": "saved_summary",
                        "summary_source": "evaluation.json or metrics.json",
                        "embedding_file_available": False,
                    }
                )
                entries.append(entry)
                continue
            excluded_runs.append(
                {
                    "name": label,
                    "run_id": run_id,
                    "reason": "No embedding file or saved representation summary is available.",
                }
            )
            continue
        entry = evaluate_representation(
            label,
            np.load(emb_path).astype(np.float32),
            codes,
            target_k=args.target_k,
            knn_k=args.knn,
            seed=run_seed(run_dir),
        )
        entry["run_id"] = run_id
        entry["status"] = "computed"
        entry["protocol"] = run_metadata(run_dir)
        pseudo_path = run_dir / "leiden_pseudolabels.npy"
        if not pseudo_path.is_file():
            pseudo_path = run_dir / "initial_labels.npy"
        if pseudo_path.is_file():
            entry["pseudolabel_ari"] = float(adjusted_rand_score(codes, np.load(pseudo_path)))
        entries.append(entry)

    report = {
        "target_k": args.target_k,
        "knn": args.knn,
        "label_scope": "Metrics using final_annot are conditional on the supplied local reference labels.",
        "representations": entries,
        "excluded_runs": excluded_runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for e in entries:
        ari = f"ARI={e['ari']:+.4f}" if "ari" in e else "ARI=summary-only"
        print(f"{e['name']:34s} status={e['status']} dim={e.get('dimension', 'n/a'):>4} "
              f"{ari} kNN={e.get('knn_purity', float('nan')):.4f} "
              f"eff_rank={e.get('effective_rank', float('nan')):.2f}")
    print(f"Excluded runs without reproducible representation artifacts: {len(excluded_runs)}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
