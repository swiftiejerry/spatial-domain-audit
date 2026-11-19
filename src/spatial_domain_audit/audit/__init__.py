"""Construct-validity metrics and standalone diagnostic checks."""

from .attainability import (
    binary_rare_f1_from_counts,
    enumerate_partition_merges,
    greedy_rare_partition,
    stirling_k_partitions,
    sweep_leiden_frontier,
)
from .marker_support import (
    classify_domain_markers,
    evaluate_marker_criteria,
    screen_domain_markers,
    top_expression_localization,
    vectorized_gene_detection,
)
from .metrics import (
    METRIC_DEFINITIONS,
    align_labels_to_reference,
    assignment_churn,
    binary_rare_scores,
    cluster_summary_table,
    embedding_diagnostics,
    evaluate_partition,
    mean_sample_sd,
    paired_difference_summary,
    per_rare_class_capture,
)
from .representation import (
    cosine_class_separation,
    cosine_knn_purity,
    effective_rank,
    evaluate_representation,
    leiden_to_target_k,
    upstream_pca_space,
)

__all__ = [
    "METRIC_DEFINITIONS",
    "align_labels_to_reference",
    "assignment_churn",
    "binary_rare_f1_from_counts",
    "binary_rare_scores",
    "classify_domain_markers",
    "cluster_summary_table",
    "cosine_class_separation",
    "cosine_knn_purity",
    "effective_rank",
    "embedding_diagnostics",
    "enumerate_partition_merges",
    "evaluate_marker_criteria",
    "evaluate_partition",
    "evaluate_representation",
    "greedy_rare_partition",
    "leiden_to_target_k",
    "mean_sample_sd",
    "paired_difference_summary",
    "per_rare_class_capture",
    "screen_domain_markers",
    "stirling_k_partitions",
    "sweep_leiden_frontier",
    "top_expression_localization",
    "upstream_pca_space",
    "vectorized_gene_detection",
]
