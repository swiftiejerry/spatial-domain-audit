# Construct-Validity Audit and Metric Definitions

This library formalizes four diagnostic checks and paired evaluation protocols for
interpreting rare-domain outcomes in heterogeneous-graph spatial transcriptomics
clustering. Let $z_0$ denote the initial pseudo-label partition, $\hat{z}$ the
predicted cluster partition, $I$ the graph or clustering intervention, and $S$ the
mini-batch sampler.

## 1. Target Movement (`audit_target_movement.py`)

Evaluates whether a trained model's output can deviate meaningfully from its
fixed training target:

- **Target Agreement**: $\operatorname{ARI}(z_0, \hat{z})$ between the starting
  pseudo-label $z_0$ and predicted partition $\hat{z}$.
- **Aligned Assignment Churn**:
  $$m = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[z_{0,i} \ne \operatorname{align}(\hat{z}_i)\right]$$
  where $\operatorname{align}(\cdot)$ solves the maximum-overlap one-to-one cluster
  bijection via linear sum assignment (Hungarian matching) over the contingency
  table. Notice that raw label-ID inequality is permutation-sensitive and is
  never used for churn reporting.

## 2. Endpoint Attainability (`audit_attainability.py`)

Assesses whether the binary rare-domain objective can register a non-zero score
under the cluster budget $K$ and initial partition $z_0$:

- **Primary binary endpoint (`rare_primary`)**: truth class share $\le 5\%$,
  predicted cluster share $\le 5\%$.
- **Secondary binary endpoint (`rare_secondary`)**: truth class share $< 5\%$,
  predicted cluster share $< 4.5\%$.
- **Exact Partition-Merge Upper Bound**: Enumerates all $S(N_c, K)$ ways of
  merging $N_c$ reference domains into $K$ disjoint groups (e.g., $S(10,5) = 42{,}525$)
  and evaluates the maximum attainable binary `rare_f1`, verifying whether $K$
  imposes a structural ceiling.
- **Leiden Resolution Frontier**: Sweeps resolution across the exact
  expression-PCA neighbor graph matching `scformer.utils.initial_clustering` to
  measure whether global agreement (ARI) and rare-domain reachability co-align
  or trade off.
- **Rare-Cluster Enrichment**: Compares the fraction of true-rare spots inside
  predicted sub-threshold clusters against the section-wide rare prevalence
  ($8.04\%$ on Human Lymph Node A1) so unenriched small noise clusters are
  distinguished from genuine rare-domain capture.

## 3. Intervention Activity (`audit_edge_retention.py`, `run_paired_spatial_comparison.py`)

Verifies that an architectural relation enters computation rather than existing
solely as an unexercised configuration entry:

- **Intra-Batch Edge Retention**: Fraction of directed spatial edges with both
  endpoints jointly present inside a mini-batch (`intra_batch_edge_retention`)
  and touching at least one core spot weighted by loss gradients
  (`core_incident_edge_retention`).
- **Live Forward/Gradient Probes**: Per-layer spatial message $L_2$ norm hooks,
  per-layer gradient norms on the spatial relation parameters, reversible
  inference-time relation removal check against the nested two-relation model,
  and bitwise inference determinism verification.
- **Boundary-F1 and Spatial Connectivity**: On deduplicated undirected coordinate
  $k$-NN edges $(i, j) \in E$, edge-level boundary F1 between
  $\mathbb{1}[y_i \ne y_j]$ and $\mathbb{1}[\hat{z}_i \ne \hat{z}_j]$, paired with
  within-cluster connected component counts.

## 4. Input Signal Support and Representation Geometry (`audit_marker_support.py`, `audit_representation.py`)

Tests whether candidate rare domains possess measurable expression or spatial
structure in the feature representation:

- **Full-Ranking Differential Expression Screen**: Evaluates Wilcoxon rank-sum
  tests across the complete gene ranking (without truncating top candidates)
  requiring adjusted $p < 0.05$, in-domain detection $\ge 0.60$, and
  in/out-of-domain detection ratio $\ge 2.0$, followed by spatial localization of
  top-$n$ highest-expressing spots relative to domain prevalence.
- **Representation Diagnostics**: Effective rank (participation ratio
  $(\sum s_i^2)^2 / \sum s_i^4$ of mean-centered singular values), cosine
  $k$-NN label purity against imbalanced-class chance level $\sum_c (n_c / N)^2$,
  and same- vs. different-class cosine separation. The reported kNN purity uses
  **$k=10$**. Callers pass this value explicitly and evaluation JSON records it.
