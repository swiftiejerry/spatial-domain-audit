# spatial-domain-audit

Construct-validity audit and experimental evaluation toolkit for rare-domain
clustering in heterogeneous-graph spatial transcriptomics models.

## Motivation

Evaluating spatial clustering interventions strictly through an end-to-end
rare-domain score conflates distinct structural failures before architectural
effects become measurable:

1. **Target movement**: whether predictions deviate from frozen pseudo-labels
   (measured via Hungarian-aligned spot-level churn and target ARI).
2. **Endpoint attainability**: whether binary rare-domain thresholds can register
   non-zero values given the requested cluster count $K$ and initial partition
   geometry (verified via exact Stirling $S(N_c, K)$ group-merge enumeration and
   Leiden resolution sweeps).
3. **Intervention activity**: whether added spatial edges actually remain intact
   inside mini-batches and propagate forward/backward signals through HGT layers
   (measured via intra-batch edge retention, forward message hooks, spatial
   gradient norms, and reversible relation-ablation checks).
4. **Input signal support**: whether reference rare domains carry localized
   univariate differential-expression markers or separable representation geometry.

This repository packages the modular Python audit library
(`spatial_domain_audit`), paired fixed-target and prototype-refresh HGT
implementations built over the frozen upstream `scFormer` baseline, protocol
configurations, and CPU-verifiable benchmark outputs for Human Lymph Node Visium
section A1 (`3,484` spots, `18,085` genes, `10` supplied reference domains with
`5` rare domains totaling `8.0%` of spots).

## Construct-Validity Findings on Human Lymph Node A1

| Audit Check | Outcome on Evaluated Settings | Methodological Implication |
|---|---|---|
| **Target movement** | Default 100-epoch runs reproduce Leiden targets at ARI `0.9948--1.0000` (`0--7` / `3,484` spots moved). Aligned 30-epoch runs move `46--67%` of spots. | Near-zero paired differences under frozen resolution-0.5 targets reflect constrained label copying rather than spatial nullity. |
| **Endpoint attainability** | Enumerating all `42,525` five-group reference merges reaches binary `rare-F1 = 1.0000`; res-`0.8` runs retain `0.3810--0.3927`. Default res-`0.5` has `0` clusters below `<4.5%`. | $K=5$ is not the binding constraint; initial Leiden resolution pins default `rare-F1` at zero prior to training. |
| **Intervention activity** | Uniform shuffling drops `k=10` spatial edges to `0.78%` in-batch; recursive median bisection + 1-hop halo recovers `100%` (batch size `58--81`), with non-zero layer messages & gradients. | Sampler loss would obscure spatial effects unless contiguous halo batching is verified explicitly. |
| **Input signal support** | Complete-ranking Wilcoxon screen + spatial top-$n$ localization identifies spatially localized expression support only for follicle (`CXCL13`, `89.6%` in-domain, `10.1x` mean). | Negative univariate marker screens prevent conflating absent marginal markers with zero spatial/multivariate signal. |

### Aligned Spatial Comparison Summary (`B0 - A*`, 3 Seeds: `0, 1, 42`)

Under deterministic standardized Top-Z (`K=20`), library-normalized $\log 1\text{p}$
reconstruction (`normalized_x`), matched tile-and-halo batching, and shared
parameter initialization:

| Metric | Mean $\Delta \pm$ s.d. | Direction Across Seeds |
|---|---|---|
| `ARI` | `-0.1177 +/- 0.0585` | Consistent negative (`-0.1445 / -0.0506 / -0.1579`) |
| `NMI` | `-0.1411 +/- 0.0806` | Consistent negative (`-0.1707 / -0.0499 / -0.2026`) |
| `boundary-F1` | `-0.0399 +/- 0.0141` | Consistent negative (`-0.0466 / -0.0237 / -0.0494`) |
| `rare-F1 (<=5%)` | `+0.0592 +/- 0.0986` | Non-consistent (`+0.1730 / 0.0000 / +0.0046`) |

Under prototype-refresh dynamic clustering (`C - A`, default basis), ARI fell by
`-0.1966 +/- 0.0340` (`NMI -0.2047 +/- 0.0231`, `rare-F1 +0.0594 +/- 0.0262`)
while near-collinear warm-up prototypes kept the confidence gate shut in seeds
`1` and `42` (`0.00057`, `0.00000`, `0.00000` confident fraction at initialization).

## Repository Layout

```text
src/spatial_domain_audit/
  audit/
    metrics.py                # CPU metric engine (ARI, NMI, boundary-F1, rare-F1, enrichment, churn)
    attainability.py          # Exact Stirling S(N_c, K) merge enumeration & Leiden frontier sweeps
    marker_support.py         # Full-ranking Wilcoxon DE & spatial localization screen
    representation.py         # PCA reference projection, effective rank, cosine separability, k-NN purity
  graph/
    spatial_knn.py            # Deterministic Euclidean spatial k-NN and radius graphs
    spatial_batching.py       # Recursive spatial median bisection, 1-hop halo, edge retention metrics
    topz_sampled.py           # Sampled Top-Z spot-gene graph construction
    topz_normalized.py        # Normalized-log X target, gene-wise Z, deterministic Top-Z selection
  clustering/
    prototype_dynamic.py      # Prototype initialization, confidence gating, centroid refresh
  models/
    hetero_encoder.py         # Relation-mean heterogeneous graph encoder backbone
  training/
    objectives.py             # Reconstruction KL & masked prototype contrastive losses
    fixed_target_trainer.py   # Fixed-target HGT trainer (2 or 3 relations + halo isolation)
    prototype_trainer.py      # Warm-up + confidence-gated prototype clustering trainer
    training_loops.py         # Lightweight full-graph reference loops
    common.py                 # Seeding, SHA-256 verification, shared batch/graph contract helpers
scripts/
  verify_saved_outputs.py     # CPU-only verification over saved run partitions (no GPU/H5AD needed)
  audit_target_movement.py    # Check 1 CLI: Hungarian alignment churn & small-cluster enrichment
  audit_attainability.py      # Check 2 CLI: S(10, 5) partition merge bound & Leiden frontier
  audit_edge_retention.py     # Check 3a CLI: Sampler edge retention & halo compactness audit
  audit_representation.py     # Embedding geometry, effective rank & k-NN purity audit
  audit_marker_support.py     # Check 4b CLI: Full-ranking Wilcoxon DE + top-n spatial localization
  run_upstream_baseline.py    # Upstream expression-only baseline runner
  run_fixed_target_comparison.py
  run_prototype_clustering.py
  run_aligned_warmup.py
  run_paired_spatial_comparison.py
configs/                      # Standardized YAML & JSON execution protocols
data/reference/               # Bundled lightweight labels, spot coordinates & fixed spatial graph
experiments/
  results.json                # Unified structured index across saved experimental runs
  reports/                    # Audit JSON outputs
  runs/                       # Saved prediction partitions and per-run metrics JSON files
third_party/scFormer/         # Untouched frozen upstream snapshot (commit 7401620)
docs/                         # Data provenance specification and formal metric definitions
tests/                        # CPU unit & integration test suite
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Verification (CPU-Only, No External Data Required)

All saved partitions, paired differences, Hungarian target-movement counts,
merge-ceiling bounds, and spatial edge-retention measurements can be verified
directly from the bundled `data/reference/` and `experiments/` files:

```bash
pytest
python scripts/verify_saved_outputs.py
python scripts/audit_target_movement.py
python scripts/audit_attainability.py
python scripts/audit_edge_retention.py
```

## Running Full Training & Expression-Dependent Audits

Full model training, differential-expression screening, and expression-PCA
Leiden sweeps require placing the public Human Lymph Node A1 `adata_RNA.h5ad`
matrix under `data/raw/human_lymph_node_A1/adata_RNA.h5ad` (see
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md)):

```bash
# Baseline & fixed-target spatial vs. expression runs
python scripts/run_upstream_baseline.py --seed 0
python scripts/run_fixed_target_comparison.py --arm A --seed 0 --protocol-file configs/mainline_protocol.json
python scripts/run_fixed_target_comparison.py --arm B --seed 0 --spatial-k 10 --protocol-file configs/mainline_protocol.json

# Prototype-refresh clustering
python scripts/run_prototype_clustering.py --arm C --seed 0 --warmup-epochs 30 --update-interval 10 --tau 0.005 --delta 0.8 --delta-mode absolute --no-center-embedding --free-k0 --leiden-resolution 0.5

# Aligned deterministic basis + live layer-message/gradient wiring probes
python scripts/run_aligned_warmup.py --arm Astar --seed 0 --protocol-file configs/aligned_warmup_protocol.json
python scripts/run_paired_spatial_comparison.py --arm B0 --seed 0

# Expression-space audits
python scripts/audit_attainability.py --sweep-frontier
python scripts/audit_marker_support.py
python scripts/audit_representation.py
```

## Scope and Methodological Boundaries

- All empirical measurements cover a single 10x Genomics Visium human lymph-node
  section across three seeds (`0`, `1`, `42`) using RNA counts alone.
- Reference agreement and rare-domain endpoints depend on the supplied
  `obs['final_annot']` annotation field; see
  [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) for provenance details.
- Results falsify only the specific evaluated configurations (unweighted
  $k=10$ spatial relation under relation-mean / multi-head HGT and stated
  prototype-refresh hyperparameters), not spatial graph methods or dynamic
  clustering in general.

## Attribution & License

- **Upstream Baseline**: Built over an unmodified snapshot of `scFormer` (Huang
  et al., *Advanced Biotechnology* 2026, DOI: `10.1007/s44307-026-00121-y`)
  stored under `third_party/scFormer/`.
- **Dataset Source**: Processed human lymph-node Visium section
  `Dataset11_Human_Lymph_Node_A1` from the SpatialGlue release (GEO:
  `GSE263617`; Zenodo: `10.5281/zenodo.7879713` / `10.5281/zenodo.10362607`).
- **License**: New audit and extension modules are released under the
  [MIT License](LICENSE).
