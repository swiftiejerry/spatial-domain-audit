"""Paired aligned spatial comparison with parameter grafting, wiring probes, and topology nulls."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "scFormer"
sys.path[:0] = [str(ROOT / "src"), str(UPSTREAM)]

from spatial_domain_audit.audit.metrics import evaluate_partition  # noqa: E402
from spatial_domain_audit.audit.representation import leiden_to_target_k  # noqa: E402
from spatial_domain_audit.graph.spatial_batching import add_spatial_halo, spatial_batch_order  # noqa: E402
from spatial_domain_audit.graph.spatial_knn import build_spatial_graph  # noqa: E402
from spatial_domain_audit.graph.topz_normalized import deterministic_topz_batch_select  # noqa: E402
from spatial_domain_audit.training.common import (  # noqa: E402
    file_sha256,
    restore_rng_state,
    seed_all,
    snapshot_rng_state,
    state_dict_sha256,
)
from spatial_domain_audit.training.fixed_target_trainer import (  # noqa: E402
    FixedTargetConfig,
    FixedTargetTrainer,
    _spatial_pairs_by_cell,
)

PROTOCOL = ROOT / "configs" / "paired_spatial_protocol.json"
RUNS = ROOT / "experiments" / "runs" / "p2_aligned_v1"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def graft_spatial_model(base: FixedTargetTrainer) -> tuple[torch.nn.Module, dict]:
    """Copy 2-relation state into a 3-relation HGT and restore post-init RNG state."""
    from scformer.model import GNN_from_raw

    rng = snapshot_rng_state()
    c = base.config
    gnn = GNN_from_raw(
        in_dim=[base.rna_matrix.shape[0], base.rna_matrix.shape[1]],
        n_hid=c.n_hidden,
        num_types=2,
        num_relations=3,
        n_heads=c.n_heads,
        n_layers=c.n_layers,
        dropout=c.dropout,
    ).to(base.device)
    source = base.gnn.state_dict()
    target = gnn.state_dict()
    projected = {}
    for key, value in source.items():
        if target[key].shape == value.shape:
            target[key].copy_(value)
            projected[key] = target[key]
        else:
            assert "relation_" in key and target[key].shape[0] == 3 and value.shape[0] == 2
            target[key][:2].copy_(value)
            projected[key] = target[key][:2]
    gnn.load_state_dict(target)
    assert state_dict_sha256(source) == state_dict_sha256(projected)
    restore_rng_state(rng)
    return gnn, {
        "Astar_shared_initial_sha256": state_dict_sha256(source),
        "B0_shared_initial_sha256": state_dict_sha256(projected),
        "all_shared_parameters_equal": True,
        "post_init_rng_restored": bool(torch.equal(torch.get_rng_state(), rng[0])),
    }


def _forward_payload(gnn: torch.nn.Module, payload: dict) -> torch.Tensor:
    return gnn(payload["node_feature"], payload["node_type"], payload["edge_index"], payload["edge_type"])


def audit_wiring(model: FixedTargetTrainer, reference_gnn: torch.nn.Module, batch: dict) -> dict:
    """Measure live per-layer spatial message norms, gradients, and reversible removal parity."""
    rng = snapshot_rng_state()
    model.gnn.eval()
    reference_gnn.eval()
    payload = model._build_batch(batch)
    off = dict(payload)
    keep = payload["edge_type"] != 2
    off["edge_type"] = payload["edge_type"][keep]
    off["edge_index"] = payload["edge_index"][:, keep]
    records = []
    handles = []
    for index, layer in enumerate(model.gnn.gcs):

        def hook(module, inputs, output, index=index):
            data = inputs[0]
            mask = data["edge_type"] == 2
            if bool(mask.any()):
                records.append(
                    {
                        "layer": index,
                        "n_spatial_edges": int(mask.sum()),
                        "source_input_l2_mean": float(data["node_inp_j"][mask].norm(dim=1).mean()),
                        "weighted_message_l2_mean": float(output[mask].norm(dim=1).mean()),
                    }
                )

        handles.append(layer.base_conv.register_message_forward_hook(hook))

    with torch.no_grad():
        enabled = _forward_payload(model.gnn, payload)
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        disabled = _forward_payload(model.gnn, off)
        reference = _forward_payload(reference_gnn, off)
        repeated = _forward_payload(model.gnn, payload)

    value = _forward_payload(model.gnn, payload)
    loss = model._reconstruction_loss(
        value[payload["node_type"] == 0], value[payload["node_type"] == 1], payload["kl_target"]
    )
    loss.backward()
    gradients = [float(layer.base_conv.relation_msg.grad[2].norm()) for layer in model.gnn.gcs]
    model.gnn.zero_grad(set_to_none=True)

    result = {
        "batch": 0,
        "layers": records,
        "spatial_relation_msg_gradient_norms": gradients,
        "enabled_minus_disabled_max_abs": float((enabled - disabled).abs().max()),
        "disabled_minus_Astar_max_abs": float((disabled - reference).abs().max()),
        "eval_repeat_bitwise_equal": bool(torch.equal(enabled, repeated)),
        "dropout_disabled": all(
            not m.training for m in model.gnn.modules() if isinstance(m, torch.nn.Dropout)
        ),
    }
    assert len(records) == len(model.gnn.gcs) and all(x["weighted_message_l2_mean"] > 0 for x in records)
    assert result["enabled_minus_disabled_max_abs"] > 0 and result["disabled_minus_Astar_max_abs"] < 1e-6
    assert result["eval_repeat_bitwise_equal"] and all(x > 0 for x in gradients)
    model.gnn.train()
    restore_rng_state(rng)
    return result


def infer_full(model: FixedTargetTrainer, spatial_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run full-graph inference returning spot embeddings, gene embeddings, and edge counts."""
    model.gnn.eval()
    n_genes, n_cells = model.rna_matrix.shape
    model.spatial_neighbours = _spatial_pairs_by_cell(spatial_edges) if model.config.spatial_relation else {}
    batch = {
        "gene_index": list(range(n_genes)),
        "cell_index": list(range(n_cells)),
        "core_index": list(range(n_cells)),
    }
    payload = model._build_batch(batch)
    with torch.no_grad():
        value = _forward_payload(model.gnn, payload)
    spots = value[:n_cells].cpu().numpy().copy()
    genes = value[n_cells:].cpu().numpy().copy()
    stats = {
        "n_spots": n_cells,
        "n_genes": n_genes,
        "spot_gene_edges": payload["n_spot_gene_edges"],
        "spatial_edges": payload["n_spatial_edges"],
        "eval_mode": not model.gnn.training,
    }
    del payload, value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return spots, genes, stats


def null_edges(coords: np.ndarray, edges: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    """Construct coordinate-permuted, degree-preserving rewired, and relation-off inference graphs."""
    import networkx as nx

    perm = np.random.RandomState(seed + 10000).permutation(len(coords))
    shuffled = build_spatial_graph(coords[perm], k=10).edge_index
    graph = nx.Graph()
    graph.add_nodes_from(range(len(coords)))
    graph.add_edges_from(edges.T.tolist())
    before = np.asarray([graph.degree(i) for i in range(len(coords))])
    nx.double_edge_swap(
        graph, nswap=5 * graph.number_of_edges(), max_tries=100 * graph.number_of_edges(), seed=seed + 20000
    )
    after = np.asarray([graph.degree(i) for i in range(len(coords))])
    assert np.array_equal(before, after) and nx.number_of_selfloops(graph) == 0
    rewired = np.asarray(sorted((a, b) for u, v in graph.edges for a, b in ((u, v), (v, u))), dtype=np.int64).T
    assert rewired.shape == edges.shape
    return {
        "coordinate_permutation": shuffled,
        "degree_rewire": rewired,
        "relation_off": np.empty((2, 0), dtype=np.int64),
    }, {
        "permutation": perm.tolist(),
        "rewire_degree_preserved": True,
        "rewire_selfloops": 0,
        "swap_count": 5 * graph.number_of_edges(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict paired spatial comparison with live probes.")
    parser.add_argument("--arm", choices=["Astar", "B0", "B0_holdout"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 42], required=True)
    parser.add_argument("--h5ad", type=Path, default=ROOT / "data" / "raw" / "human_lymph_node_A1" / "adata_RNA.h5ad")
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    out = args.runs_root / (f"{args.arm}_seed{args.seed}" + ("_smoke" if args.smoke else ""))
    if (out / "manifest.json").exists() and not args.evaluate_only:
        raise SystemExit(f"completed run already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    import scanpy as sc
    from scipy.sparse import csr_matrix
    from scformer.utils import initial_clustering

    seed_all(args.seed, deterministic=True)
    adata = sc.read_h5ad(args.h5ad)
    class_names, truth = np.unique(adata.obs["final_annot"].astype(str).to_numpy(), return_inverse=True)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    rna = csr_matrix(adata.X).T
    pseudo = np.asarray(initial_clustering(rna), dtype=np.int64)

    graph = build_spatial_graph(coords, k=protocol["spatial_k"], weighted=False)
    blocks = spatial_batch_order(coords, n_batches=math.ceil(adata.n_obs / protocol["cell_size"]))
    halo = add_spatial_halo(blocks, graph.edge_index)
    gene_batches, node_ids, gene_dic = deterministic_topz_batch_select(rna, blocks, k=protocol["topz_k"])
    batches = [
        {"gene_index": gb["gene_index"], "cell_index": hb["cell_index"], "core_index": hb["core_index"]}
        for gb, hb in zip(gene_batches, halo)
    ]
    assert all(gb["cell_index"] == hb["core_index"] for gb, hb in zip(gene_batches, halo))
    selection = {int(cell): row["g"] for cell, row in gene_dic.items()}
    np.save(out / "selection.npy", np.asarray([selection[i] for i in range(adata.n_obs)], dtype=np.int64))
    np.save(out / "spatial_edges.npy", graph.edge_index)
    np.save(out / "initial_labels.npy", pseudo)
    np.save(out / "Node_Ids.npy", node_ids)
    _write_json(out / "batches.json", batches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FixedTargetConfig(
        epochs=1 if args.smoke else protocol["epochs"],
        spatial_relation=False,
        kl_target="normalized_x",
        gene_input="standardized_z",
        edge_source="per_spot_selection",
        kl_only=True,
    )
    model = FixedTargetTrainer(rna, batches, pseudo, cfg, device, selection=selection)
    reference = model.gnn
    common_init = state_dict_sha256(reference.state_dict())
    alignment = {"Astar_shared_initial_sha256": common_init, "all_shared_parameters_equal": True}
    if args.arm != "Astar":
        model.gnn, alignment = graft_spatial_model(model)
        model.config = copy.copy(cfg)
        model.config.spatial_relation = True
        model.spatial_neighbours = _spatial_pairs_by_cell(graph.edge_index)
        model.optimizer = torch.optim.AdamW(
            model.gnn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        _write_json(out / "wiring.json", audit_wiring(model, reference, batches[0]))
    del reference
    model.gnn.train()
    gc.collect()

    if args.arm == "B0_holdout":
        holdout = np.lexsort((np.arange(len(coords)), coords[:, 0]))[: int(math.ceil(len(coords) * 0.2))]
        mask = np.ones(len(coords), dtype=bool)
        mask[holdout] = False
        train_edges = graph.edge_index[:, mask[graph.edge_index[0]] & mask[graph.edge_index[1]]]
        model.spatial_neighbours = _spatial_pairs_by_cell(train_edges)
        np.save(out / "holdout_spots.npy", holdout)
        np.save(out / "train_spatial_edges.npy", train_edges)

    stats = model.spatial_edge_stats()
    _write_json(
        out / "config.json",
        {
            "arm": args.arm,
            "seed": args.seed,
            "protocol": protocol,
            "protocol_sha256": file_sha256(PROTOCOL),
            "model": asdict(model.config),
            "initialization": alignment,
            "training_spatial_edges": stats,
            "argv": sys.argv,
        },
    )

    if not args.evaluate_only:
        torch.save(model.gnn.state_dict(), out / "initial_state.pth")
        history = model.train()
        _write_json(out / "history.json", history)
        torch.save(model.gnn.state_dict(), out / "checkpoint.pth")
    else:
        model.gnn.load_state_dict(
            torch.load(out / "checkpoint.pth", weights_only=True, map_location=model.device)
        )

    model.gnn.eval()
    mini = model.predict()
    np.save(out / "minibatch_spot_embedding.npy", mini["cell_embedding"])
    np.save(out / "minibatch_argmax.npy", mini["pred_label"])

    spot, gene, full_stats = infer_full(model, graph.edge_index)
    np.save(out / "spot_embedding.npy", spot)
    np.save(out / "gene_embedding.npy", gene)
    _write_json(out / "inference.json", full_stats)

    pred = leiden_to_target_k(spot, target_k=5, seed=args.seed)
    np.save(out / "pred.npy", pred)
    mini_labels = leiden_to_target_k(mini["cell_embedding"], target_k=5, seed=args.seed)
    np.save(out / "minibatch_labels.npy", mini_labels)

    ev = evaluate_partition(
        pred,
        truth,
        initial_labels=pseudo,
        coordinates=coords,
        embedding=spot,
        class_names=class_names.tolist(),
        knn_k=10,
    )
    _write_json(out / "evaluation.json", ev)
    _write_json(
        out / "minibatch_evaluation.json",
        evaluate_partition(
            mini_labels,
            truth,
            initial_labels=pseudo,
            coordinates=coords,
            embedding=mini["cell_embedding"],
            class_names=class_names.tolist(),
            knn_k=10,
        ),
    )

    if args.arm == "B0":
        variants, null_info = null_edges(coords, graph.edge_index, args.seed)
        _write_json(out / "null_generation.json", null_info)
        for name, edges in variants.items():
            ns, _, ns_stats = infer_full(model, edges)
            nl = leiden_to_target_k(ns, target_k=5, seed=args.seed)
            np.save(out / f"{name}_edges.npy", edges)
            np.save(out / f"{name}_embedding.npy", ns)
            np.save(out / f"{name}_labels.npy", nl)
            _write_json(
                out / f"{name}_evaluation.json",
                evaluate_partition(
                    nl,
                    truth,
                    coordinates=coords,
                    initial_labels=pseudo,
                    embedding=ns,
                    class_names=class_names.tolist(),
                    knn_k=10,
                ),
            )
            _write_json(out / f"{name}_inference.json", ns_stats)

    artifacts = {
        x.name: file_sha256(x) for x in out.iterdir() if x.is_file() and x.name != "manifest.json"
    }
    _write_json(
        out / "manifest.json",
        {
            "status": "complete",
            "arm": args.arm,
            "seed": args.seed,
            "dataset_sha256": file_sha256(args.h5ad),
            "protocol_sha256": file_sha256(PROTOCOL),
            "artifacts": artifacts,
            "runtime_sec": time.time() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    print(f"[complete] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
