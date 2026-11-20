"""Run a low-power spatial-block sensitivity check for five candidate markers."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5AD = (
    ROOT.parent
    / "experiments_archive"
    / "recovered_runs"
    / "p2_aligned_v1"
    / "source_snapshot"
    / "data"
    / "raw"
    / "human_lymph_node_A1"
    / "adata_RNA.h5ad"
)
MARKERS = {
    "follicle": "CXCL13",
    "medulla vessels": "NOTCH3",
    "subcapsular sinus": "CD9",
    "hilum": "COL1A2",
    "trabeculae": "ERRFI1",
}


def exact_sign_flip(contrasts: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value for equally weighted block contrasts."""
    observed = abs(float(np.mean(contrasts)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(contrasts)):
        permuted = abs(float(np.mean(contrasts * np.asarray(signs))))
        extreme += permuted >= observed - 1e-12
        total += 1
    return extreme / total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assess candidate marker directions across five coordinate-defined spatial blocks."
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "reports" / "spatial_block_marker_sensitivity.json",
    )
    args = parser.parse_args()

    import scanpy as sc

    adata = sc.read_h5ad(args.h5ad)
    labels = adata.obs[args.label_key].astype(str).to_numpy()
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    blocks = KMeans(n_clusters=5, n_init=20, random_state=0).fit_predict(coordinates)

    expression = adata.copy()
    sc.pp.normalize_total(expression, target_sum=1e4)
    sc.pp.log1p(expression)

    rows = []
    for domain, gene in MARKERS.items():
        gene_positions = np.flatnonzero(np.asarray(expression.var_names.astype(str)) == gene)
        if gene_positions.size != 1:
            raise ValueError(f"Expected one feature named {gene}, found {gene_positions.size}")
        values = expression.X[:, int(gene_positions[0])]
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values, dtype=np.float64).reshape(-1)

        contrasts = []
        block_rows = []
        for block in sorted(np.unique(blocks)):
            in_domain = (blocks == block) & (labels == domain)
            outside = (blocks == block) & (labels != domain)
            if not in_domain.any() or not outside.any():
                continue
            difference = float(values[in_domain].mean() - values[outside].mean())
            contrasts.append(difference)
            block_rows.append({
                "block": int(block),
                "n_domain": int(in_domain.sum()),
                "n_outside": int(outside.sum()),
                "mean_log_expression_difference": difference,
            })

        if len(contrasts) >= 2:
            contrast_array = np.asarray(contrasts, dtype=np.float64)
            p_value = exact_sign_flip(contrast_array)
            mean_difference = float(contrast_array.mean())
            status = "ESTIMABLE_LOW_POWER_SENSITIVITY"
        else:
            p_value = None
            mean_difference = float(contrasts[0]) if contrasts else None
            status = "NOT_ESTIMABLE_FEWER_THAN_TWO_BLOCKS"

        rows.append({
            "domain": domain,
            "gene": gene,
            "status": status,
            "n_blocks_with_both_groups": len(contrasts),
            "mean_block_difference_log1p_normalized_expression": mean_difference,
            "exact_two_sided_sign_flip_p": p_value,
            "blocks": block_rows,
        })

    report = {
        "scope": "Spatial-block sensitivity analysis of five prespecified marker candidates.",
        "label_provenance": "The source and revision of final_annot remain unresolved; results are conditional on the supplied labels.",
        "input": {
            "n_spots": int(labels.size),
            "label_key": args.label_key,
            "markers": MARKERS,
        },
        "method": {
            "block_assignment": "KMeans on two-dimensional spot coordinates, five blocks, n_init=20, random_state=0; labels and expression were not used to assign blocks.",
            "expression": "Per-spot normalize_total(1e4) followed by log1p, matching the descriptive marker screen.",
            "contrast": "Within each block, mean expression in the named domain minus mean expression outside it; blocks with either group absent are omitted.",
            "test": "Exact two-sided sign-flip test of equally weighted block contrasts. With five blocks there are at most 32 sign patterns, so p-values are coarse.",
            "interpretation": "Low-power sensitivity check for direction consistency. It does not remove within-block dependence or establish independent biological replication.",
        },
        "spatial_blocks": {
            "sizes": {
                str(int(block)): int(np.count_nonzero(blocks == block))
                for block in np.unique(blocks)
            }
        },
        "domains": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        p_text = "not estimable" if row["exact_two_sided_sign_flip_p"] is None else f"p={row['exact_two_sided_sign_flip_p']:.4f}"
        difference = row["mean_block_difference_log1p_normalized_expression"]
        difference_text = "NA" if difference is None else f"mean block difference={difference:.4f}"
        print(f"{row['domain']} / {row['gene']}: {row['n_blocks_with_both_groups']} blocks, {difference_text}, {p_text}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
