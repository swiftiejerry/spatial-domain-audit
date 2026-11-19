# Vendored Baseline: scFormer (`third_party/scFormer`)

This directory preserves an unmodified upstream snapshot of `scFormer`
(commit `7401620`), retaining only the Python package (`scformer/`) and its
upstream dependency manifest (`requirements.txt`).

## Purpose

- Baseline scripts (`scripts/run_upstream_baseline.py`,
  `scripts/run_fixed_target_comparison.py`, `scripts/run_prototype_clustering.py`,
  `scripts/run_aligned_warmup.py`, `scripts/run_paired_spatial_comparison.py`)
  import `scformer.utils.initial_clustering`, `scformer.utils.batch_select_whole`,
  `scformer.model.GNN_from_raw`, `scformer.model.NDR_2`, and `scformer.model.pred`
  directly from this frozen snapshot.
- Pinning the exact upstream commit guarantees that baseline pseudo-label
  generation and graph initialization remain invariant across experimental arms.

## Modification Boundary

- Upstream source files in `third_party/scFormer/scformer/` are imported as-is
  and never edited.
- All spatial-relation extensions, deterministic graph builders, spatially
  contiguous halo batching, dynamic prototype clustering, and construct-validity
  audits live strictly inside `src/spatial_domain_audit/`.

## Upstream Citation & License Notice

- Reference method: *scFormer: A heterogeneous graph transformer framework for
  rare cell identification in single-cell expression data*
  (DOI: `10.1007/s44307-026-00121-y`).
- The upstream repository did not bundle a separate `LICENSE` file; copyright for
  files inside `third_party/scFormer/` remains with the original authors.
