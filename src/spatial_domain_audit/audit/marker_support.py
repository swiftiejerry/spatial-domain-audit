"""Univariate marker-gene and spatial localization screen for input signal support."""

from __future__ import annotations

from typing import Any

import numpy as np


def vectorized_gene_detection(x_matrix: Any, in_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized in-domain, out-of-domain, and section-wide detection fractions & means."""

    def _block_stats(block: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(block, "getnnz"):
            detected = np.asarray((block > 0).sum(axis=0)).ravel()
            mean_val = np.asarray(block.mean(axis=0)).ravel()
        else:
            arr = np.asarray(block)
            detected = (arr > 0).sum(axis=0).astype(float)
            mean_val = arr.mean(axis=0)
        return detected / max(count, 1), mean_val

    pts_in, mean_in = _block_stats(x_matrix[in_mask], int(in_mask.sum()))
    pts_rest, _ = _block_stats(x_matrix[~in_mask], int((~in_mask).sum()))
    _, mean_all = _block_stats(x_matrix, x_matrix.shape[0])
    return {
        "pts_in": pts_in,
        "pts_rest": pts_rest,
        "mean_in": mean_in,
        "mean_all": mean_all,
    }


def evaluate_marker_criteria(
    row: dict[str, Any],
    *,
    max_pvals_adj: float = 0.05,
    min_pts_in: float = 0.60,
    min_ratio: float = 2.0,
) -> list[str]:
    """Return failing filter clauses for a single candidate gene entry."""
    failures: list[str] = []
    if not (row["pvals_adj"] < max_pvals_adj):
        failures.append(f"pvals_adj >= {max_pvals_adj:g}")
    if row["pts_in"] < min_pts_in:
        failures.append(f"pts_in < {min_pts_in:.2f}")
    if row["pts_in"] < min_ratio * max(row["pts_rest"], 1e-9):
        failures.append(f"pts_in/pts_rest < {min_ratio:.1f}")
    return failures


def classify_domain_markers(
    ranked_genes: list[dict[str, Any]],
    *,
    max_pvals_adj: float = 0.05,
    fallback_min_pts_in: float = 0.30,
) -> dict[str, Any]:
    """Select leading marker using a strict three-tier rule over the complete DE ranking.

    Tier 1: highest-ranked gene passing all statistical & detection criteria.
    Tier 2: if none pass all criteria, best detection-ratio significant gene (pts_in >= fallback).
    Tier 3: if no gene reaches adjusted p < max_pvals_adj, top-ranked DE candidate.
    """
    passing = [r for r in ranked_genes if not r["failures"]]
    significant = [r for r in ranked_genes if r["pvals_adj"] < max_pvals_adj]
    base = {"n_passing": len(passing), "n_significant": len(significant)}
    if passing:
        return {
            **base,
            "selected": passing[0],
            "second_choice": passing[1] if len(passing) > 1 else None,
            "tier": 1,
            "stable": True,
            "reason": "",
        }
    readable = sorted(
        (r for r in significant if r["pts_in"] >= fallback_min_pts_in),
        key=lambda r: -r["pts_in"] / max(r["pts_rest"], 1e-9),
    )
    if readable:
        return {
            **base,
            "selected": readable[0],
            "second_choice": readable[1] if len(readable) > 1 else None,
            "tier": 2,
            "stable": False,
            "reason": (
                f"no gene met all three criteria; selected highest detection-ratio "
                f"significant gene with pts_in >= {fallback_min_pts_in:.2f}"
            ),
        }
    return {
        **base,
        "selected": ranked_genes[0],
        "second_choice": ranked_genes[1] if len(ranked_genes) > 1 else None,
        "tier": 3,
        "stable": False,
        "reason": f"no gene reached pvals_adj < {max_pvals_adj:g}; selected rank-1 gene",
    }


def top_expression_localization(expression_vec: np.ndarray, in_mask: np.ndarray) -> dict[str, float | int]:
    """Measure spatial concentration of the top-n expressing spots inside a domain of size n."""
    n_domain = int(in_mask.sum())
    n_total = int(in_mask.size)
    if n_domain == 0 or n_total == 0:
        return {"n_domain": 0, "top_n_hits": 0, "top_n_precision": 0.0, "top_n_enrichment": 0.0}
    top_indices = np.argsort(-expression_vec, kind="stable")[:n_domain]
    hits = int(in_mask[top_indices].sum())
    precision = hits / n_domain
    baseline = n_domain / n_total
    return {
        "n_domain": n_domain,
        "top_n_hits": hits,
        "top_n_precision": float(precision),
        "top_n_enrichment": float(precision / baseline) if baseline > 0 else 0.0,
    }


def screen_domain_markers(
    adata: Any,
    *,
    label_key: str = "final_annot",
    rare_threshold: float = 0.05,
    max_pvals_adj: float = 0.05,
    min_pts_in: float = 0.60,
    min_ratio: float = 2.0,
) -> list[dict[str, Any]]:
    """Run full-ranking Wilcoxon differential expression screen for all rare reference domains."""
    import scanpy as sc

    work = adata.copy()
    truth = np.asarray(work.obs[label_key].astype(str).to_numpy(), dtype=str)
    unique_labels = sorted(set(truth.tolist()))
    shares = {t: float((truth == t).mean()) for t in unique_labels}
    counts = {t: int((truth == t).sum()) for t in unique_labels}
    rare_domains = sorted((t for t in unique_labels if shares[t] < rare_threshold), key=lambda t: -counts[t])

    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.tl.rank_genes_groups(work, groupby=label_key, method="wilcoxon")
    de = work.uns["rank_genes_groups"]

    x_mat = work.X
    gene_to_col = {g: i for i, g in enumerate(map(str, work.var_names))}
    results: list[dict[str, Any]] = []

    for domain in rare_domains:
        mask = truth == domain
        stats = vectorized_gene_detection(x_mat, mask)
        ranked: list[dict[str, Any]] = []
        for idx, gene in enumerate(map(str, de["names"][domain])):
            col = gene_to_col[gene]
            mean_in = float(stats["mean_in"][col])
            mean_all = float(stats["mean_all"][col])
            entry: dict[str, Any] = {
                "gene": gene,
                "rank": idx + 1,
                "score": float(de["scores"][domain][idx]),
                "logfoldchange": float(de["logfoldchanges"][domain][idx]),
                "pvals_adj": float(de["pvals_adj"][domain][idx]),
                "pts_in": float(stats["pts_in"][col]),
                "pts_rest": float(stats["pts_rest"][col]),
                "mean_log1p_in": mean_in,
                "mean_log1p_all": mean_all,
                "fold_vs_tissue_mean": mean_in / max(mean_all, 1e-12),
            }
            entry["failures"] = evaluate_marker_criteria(
                entry, max_pvals_adj=max_pvals_adj, min_pts_in=min_pts_in, min_ratio=min_ratio
            )
            ranked.append(entry)

        decision = classify_domain_markers(ranked, max_pvals_adj=max_pvals_adj)
        sel = decision["selected"]
        col_vec = x_mat[:, gene_to_col[sel["gene"]]]
        expr_vec = np.asarray(
            col_vec.toarray() if hasattr(col_vec, "toarray") else col_vec, dtype=np.float64
        ).ravel()
        loc = top_expression_localization(expr_vec, mask)
        results.append(
            {
                "domain": domain,
                "n_spots": counts[domain],
                "share": shares[domain],
                "gene": sel["gene"],
                "rank": sel["rank"],
                "score": sel["score"],
                "logfoldchange": sel["logfoldchange"],
                "pvals_adj": sel["pvals_adj"],
                "pts_in": sel["pts_in"],
                "pts_rest": sel["pts_rest"],
                "detection_ratio": float(sel["pts_in"] / max(sel["pts_rest"], 1e-9)),
                "mean_log1p_in": sel["mean_log1p_in"],
                "mean_log1p_all": sel["mean_log1p_all"],
                "fold_vs_tissue_mean": sel["fold_vs_tissue_mean"],
                "stable": decision["stable"],
                "tier": decision["tier"],
                "second_choice": decision["second_choice"]["gene"] if decision["second_choice"] else None,
                "reason": decision["reason"],
                **loc,
            }
        )
    return results
