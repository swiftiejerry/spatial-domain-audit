"""Training infrastructure, loss functions, and shared execution helpers."""

from .common import (
    DEFAULT_HP,
    UPSTREAM_N_NEIGHBORS,
    compute_graph_contract,
    file_sha256,
    prepare_batches_and_graph,
    resolve_default_resolution,
    restore_rng_state,
    seed_all,
    snapshot_rng_state,
    state_dict_sha256,
    upstream_leiden_labels,
)
from .fixed_target_trainer import FixedTargetConfig, FixedTargetTrainer
from .objectives import (
    ReconstructionClusterLoss,
    assemble_clustering_loss,
    masked_cluster_cross_entropy,
    prototype_contrastive_loss,
    symmetric_kl_from_logits,
)
from .prototype_trainer import PrototypeConfig, PrototypeTrainer
from .training_loops import TrainingConfig, TrainingResult, sampled_edge_reconstruction_loss, train_full_graph

__all__ = [
    "DEFAULT_HP",
    "FixedTargetConfig",
    "FixedTargetTrainer",
    "PrototypeConfig",
    "PrototypeTrainer",
    "ReconstructionClusterLoss",
    "TrainingConfig",
    "TrainingResult",
    "UPSTREAM_N_NEIGHBORS",
    "assemble_clustering_loss",
    "compute_graph_contract",
    "file_sha256",
    "masked_cluster_cross_entropy",
    "prepare_batches_and_graph",
    "prototype_contrastive_loss",
    "resolve_default_resolution",
    "restore_rng_state",
    "sampled_edge_reconstruction_loss",
    "seed_all",
    "snapshot_rng_state",
    "state_dict_sha256",
    "symmetric_kl_from_logits",
    "train_full_graph",
    "upstream_leiden_labels",
]
