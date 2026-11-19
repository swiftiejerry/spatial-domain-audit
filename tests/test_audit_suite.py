"""CPU unit and integration test suite for spatial_domain_audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from spatial_domain_audit.audit.attainability import (
    binary_rare_f1_from_counts,
    enumerate_partition_merges,
    greedy_rare_partition,
)
from spatial_domain_audit.audit.marker_support import (
    classify_domain_markers,
    top_expression_localization,
)
from spatial_domain_audit.audit.metrics import (
    align_labels_to_reference,
    assignment_churn,
    binary_rare_scores,
    evaluate_partition,
    paired_difference_summary,
)
from spatial_domain_audit.audit.representation import (
    cosine_class_separation,
    cosine_knn_purity,
    effective_rank,
)
from spatial_domain_audit.clustering.prototype_dynamic import (
    assign_to_prototypes,
    compute_prototypes,
    high_confidence_label_update,
    refresh_prototypes,
)
from spatial_domain_audit.graph.spatial_batching import (
    add_spatial_halo,
    core_incident_edge_retention,
    intra_batch_edge_retention,
    spatial_batch_order,
)
from spatial_domain_audit.graph.spatial_knn import build_spatial_graph
from spatial_domain_audit.graph.topz_normalized import (
    deterministic_topz_selection,
    normalized_log_matrix,
)
from spatial_domain_audit.graph.topz_sampled import build_top_z_graph
from spatial_domain_audit.models.hetero_encoder import (
    LightweightHeteroEncoder,
    make_bidirectional_relations,
)
from spatial_domain_audit.training.training_loops import TrainingConfig, train_full_graph


def test_spatial_graph_and_batch_retention() -> None:
    coords = np.array([[x, y] for x in range(10) for y in range(6)], dtype=np.float64)
    graph = build_spatial_graph(coords, mode="knn", k=4, weighted=False)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0
    assert np.all(graph.edge_index[0] != graph.edge_index[1])

    blocks = spatial_batch_order(coords, n_batches=4)
    assert sorted(int(c) for b in blocks for c in b) == list(range(len(coords)))

    halo = add_spatial_halo(blocks, graph.edge_index)
    halo_ret = core_incident_edge_retention(graph.edge_index, halo)
    assert halo_ret["retention"] == 1.0

    raw_batches = [{"cell_index": [int(c) for c in b]} for b in blocks]
    tile_ret = intra_batch_edge_retention(graph.edge_index, raw_batches)
    assert 0.0 < tile_ret["retention"] < 1.0


def test_deterministic_topz_and_sampled_graph() -> None:
    rng = np.random.default_rng(7)
    counts = rng.poisson(2.0, size=(20, 40)).astype(np.float64)  # genes x spots
    norm_x = normalized_log_matrix(counts)
    assert norm_x.shape == counts.shape
    sel_a = deterministic_topz_selection(counts, k=5, chunk=10)
    sel_b = deterministic_topz_selection(counts, k=5, chunk=16)
    assert sel_a == sel_b
    assert all(len(v) == 5 for v in sel_a.values())

    sampled_g = build_top_z_graph(counts.T, k=5)
    assert sampled_g.edge_index.shape == (2, 40 * 5)


def test_metrics_alignment_and_paired_contrasts() -> None:
    truth = np.array([0] * 40 + [1] * 40 + [2] * 16 + [3] * 4, dtype=np.int64)
    # Permuted predicted labels should produce 0 churn after Hungarian alignment.
    permuted = np.array([3] * 40 + [2] * 40 + [0] * 16 + [1] * 4, dtype=np.int64)
    churn = assignment_churn(permuted, truth)
    assert churn["changed_count"] == 0
    assert np.array_equal(align_labels_to_reference(permuted, truth), truth)

    ev = evaluate_partition(permuted, truth, initial_labels=truth)
    assert abs(ev["ari"] - 1.0) < 1e-12
    assert ev["rare_primary"]["f1"] == 1.0

    rare_none = binary_rare_scores(np.zeros(100, dtype=int), truth, pred_thr=0.045)
    assert rare_none["rare_f1"] == 0.0

    paired = paired_difference_summary({0: 0.5, 1: 0.6, 42: 0.7}, {0: 0.4, 1: 0.5, 42: 0.6})
    assert paired["all_strictly_negative"]
    assert abs(paired["mean"] - (-0.1)) < 1e-12


def test_attainability_enumeration_and_representation_metrics() -> None:
    counts = pd.Series(
        {"major_a": 1000, "major_b": 1000, "major_c": 1000, "rare_1": 30, "rare_2": 20}
    )
    n_total = int(counts.sum())
    greedy = greedy_rare_partition(counts, ["rare_1", "rare_2"], n_total, target_k=4, pred_thr=0.045)
    assert greedy["feasible_at_target_k"]
    assert greedy["rare_f1"] == 1.0

    exhaustive = enumerate_partition_merges(
        counts, ["rare_1", "rare_2"], n_total, target_k=4, pred_thr=0.045
    )
    assert exhaustive["best_rare_f1"] == 1.0
    assert binary_rare_f1_from_counts([(50, 50), (3000, 0)], 3050, 50) == 1.0

    rng = np.random.default_rng(11)
    emb = rng.normal(size=(60, 8))
    labels = np.repeat(np.arange(3), 20)
    emb[labels == 0] += 3.0
    emb[labels == 1] -= 3.0
    assert effective_rank(emb) > 1.0
    purity = cosine_knn_purity(emb, labels, k=5)
    assert purity["knn_purity"] > purity["chance"]
    assert cosine_class_separation(emb, labels) > 0.0


def test_marker_support_selector_and_localization() -> None:
    ranked = [
        {"gene": "G1", "pvals_adj": 1e-5, "pts_in": 0.4, "pts_rest": 0.2, "failures": ["pts_in < 0.60"]},
        {"gene": "G2", "pvals_adj": 1e-8, "pts_in": 0.85, "pts_rest": 0.1, "failures": []},
    ]
    res = classify_domain_markers(ranked)
    assert res["tier"] == 1 and res["stable"] and res["selected"]["gene"] == "G2"

    vec = np.array([5.0, 4.0, 0.1, 0.2, 0.0, 0.0])
    mask = np.array([True, True, False, False, False, False])
    loc = top_expression_localization(vec, mask)
    assert loc["top_n_hits"] == 2 and loc["top_n_precision"] == 1.0


def test_hetero_encoder_and_prototype_training_loop() -> None:
    torch.manual_seed(0)
    n_cells, n_genes = 24, 16
    cell_x = torch.randn(n_cells, n_genes)
    gene_x = torch.randn(n_genes, n_cells)
    topz = build_top_z_graph(torch.rand(n_cells, n_genes).numpy(), k=4)
    spatial = build_spatial_graph(np.random.default_rng(0).normal(size=(n_cells, 2)), k=3)
    rels = make_bidirectional_relations(
        torch.as_tensor(topz.edge_index, dtype=torch.long),
        torch.as_tensor(spatial.edge_index, dtype=torch.long),
    )

    model = LightweightHeteroEncoder(
        cell_input_dim=n_genes,
        gene_input_dim=n_cells,
        hidden_dim=16,
        num_clusters=3,
        num_layers=2,
        dropout=0.1,
    )
    cfg = TrainingConfig(epochs=4, mode="dynamic", warmup_epochs=2, update_interval=1, seed=0)
    result = train_full_graph(model, cell_x, gene_x, rels, num_clusters=3, config=cfg)
    assert len(result.history) == 4
    assert result.labels.shape == (n_cells,)

    embs = torch.randn(18, 8)
    init_lbls = torch.arange(18) % 3
    proto = compute_prototypes(embs, init_lbls, num_clusters=3)
    assigned = assign_to_prototypes(embs, proto.prototypes, tau=0.2)
    assert assigned.confidence.shape == (18,)
    upd = high_confidence_label_update(embs, proto.prototypes, threshold=0.5, tau=0.2, previous_labels=init_lbls)
    assert upd.proposed_labels.shape == (18,)
    ref = refresh_prototypes(embs, proto.prototypes, threshold=0.2, tau=0.5)
    assert ref.state.prototypes.shape == (3, 8)
