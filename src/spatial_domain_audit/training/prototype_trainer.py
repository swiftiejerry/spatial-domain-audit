"""Confidence-gated prototype-refresh clustering trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .fixed_target_trainer import CELL_TYPE, GENE_TYPE, FixedTargetConfig, FixedTargetTrainer
from ..clustering.prototype_dynamic import high_confidence_label_update, refresh_prototypes


@dataclass
class PrototypeConfig(FixedTargetConfig):
    """Configuration adding warm-up and confidence-gated prototype refresh settings."""

    warmup_epochs: int = 30
    update_interval: int = 10
    delta: float = 0.8
    tau: float = 0.2
    lambda_proto: float = 1.0
    leiden_neighbors: int = 10
    leiden_resolution: float | None = None
    target_k: int | None = None
    seed: int = 0
    preserve_k: bool = True
    center_embedding: bool = True
    delta_mode: str = "quantile"
    delta_quantile: float = 0.2
    init_source: str = "warmup_leiden"
    probe_every: int = 0
    probe_dir: str | None = None
    confident_only_refresh: bool = False


def leiden_on_embedding(
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    resolution: float | None,
    target_k: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster learned embeddings without label feedback, optionally scanning for target K."""
    import anndata as ad
    import scanpy as sc

    holder = ad.AnnData(X=np.ascontiguousarray(embedding, dtype=np.float32))
    sc.pp.neighbors(holder, n_neighbors=n_neighbors, use_rep="X", random_state=seed)

    def cluster_at(value: float) -> np.ndarray:
        sc.tl.leiden(holder, resolution=value, key_added="dyn", random_state=seed)
        return holder.obs["dyn"].astype(int).to_numpy()

    if target_k is None:
        chosen = 0.5 if resolution is None else float(resolution)
        labels = cluster_at(chosen)
        return labels, {
            "selection": "fixed_resolution",
            "resolution": chosen,
            "k": int(np.unique(labels).size),
        }

    scan: list[dict[str, float | int]] = []
    best: tuple[np.ndarray, float, int] | None = None
    for candidate in np.round(np.arange(0.10, 3.001, 0.05), 4):
        labels = cluster_at(float(candidate))
        k = int(np.unique(labels).size)
        scan.append({"resolution": float(candidate), "k": k})
        if k == int(target_k):
            return labels, {
                "selection": "first_exact_k",
                "resolution": float(candidate),
                "k": k,
                "scan": scan,
            }
        if best is None or abs(k - int(target_k)) < abs(best[2] - int(target_k)):
            best = (labels, float(candidate), k)
    if best is None:
        raise RuntimeError("leiden scan produced no partition")
    return best[0], {
        "selection": "closest_k",
        "resolution": best[1],
        "k": best[2],
        "scan": scan,
    }


def embedding_center(embedding: torch.Tensor, enabled: bool) -> torch.Tensor | None:
    """Mean centering vector separating cone anisotropy from relative prototype direction."""
    if not enabled:
        return None
    return embedding.detach().mean(dim=0)


def resolve_delta(confidence: torch.Tensor, *, mode: str, absolute: float, quantile: float) -> float:
    """Resolve active confidence threshold via absolute value or quantile cut."""
    if mode == "quantile":
        share = min(max(float(quantile), 0.0), 0.95)
        return float(torch.quantile(confidence.detach().float(), share))
    return float(absolute)


def frame_geometry(
    embedding: torch.Tensor, centers: torch.Tensor, center: torch.Tensor | None
) -> dict[str, float]:
    """Compute top-1 similarity spread and pairwise prototype collinearity metrics."""
    with torch.no_grad():
        points = embedding if center is None else embedding - center
        protos = centers if center is None else centers - center
        similarity = F.cosine_similarity(points.unsqueeze(1), protos.unsqueeze(0), dim=2)
        ordered = similarity.sort(dim=1, descending=True).values
        gap = ordered[:, 0] - ordered[:, 1] if ordered.size(1) > 1 else torch.zeros_like(ordered[:, 0])
        normalized = F.normalize(protos, dim=1)
        pairwise = normalized @ normalized.t()
        mask = ~torch.eye(protos.size(0), dtype=torch.bool, device=protos.device)
        off_diagonal = pairwise[mask]
        return {
            "top1_similarity_median": float(ordered[:, 0].median()),
            "top1_top2_gap_median": float(gap.median()),
            "prototype_cosine_median": float(off_diagonal.median()) if off_diagonal.numel() else 0.0,
            "prototype_cosine_max": float(off_diagonal.max()) if off_diagonal.numel() else 0.0,
        }


def prototypes_from_assignment(
    embedding: torch.Tensor,
    assignment: torch.Tensor,
    n_clusters: int,
    *,
    previous: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    preserve_k: bool = True,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """Compute cluster centroids and optionally re-seed emptied slots from uncertain spots."""
    centers = torch.zeros(
        n_clusters, embedding.size(1), device=embedding.device, dtype=embedding.dtype
    )
    empty: list[int] = []
    for cluster in range(n_clusters):
        mask = assignment == cluster
        if bool(mask.any()):
            centers[cluster] = embedding[mask].mean(dim=0)
        else:
            empty.append(cluster)

    reseeded: dict[int, list[int]] = {}
    if not empty or not preserve_k:
        return centers, reseeded

    with torch.no_grad():
        framed = embedding if center is None else embedding - center
        normalized = F.normalize(framed, dim=1)
        if confidence is not None and confidence.numel() == embedding.size(0):
            per_slot = max(10, embedding.size(0) // (4 * max(n_clusters, 1)))
            take = min(embedding.size(0), per_slot * len(empty))
            pool = torch.argsort(confidence)[:take]
        else:
            pool = torch.arange(embedding.size(0), device=embedding.device)
        for cluster in empty:
            if pool.numel() == 0:
                if previous is not None:
                    centers[cluster] = previous[cluster]
                continue
            framed_centers = centers if center is None else centers - center
            similarity = normalized[pool] @ F.normalize(framed_centers, dim=1).t()
            seed = int(pool[int(torch.argmin(similarity.max(dim=1).values))])
            neighbours = (normalized @ normalized[seed]).topk(min(10, embedding.size(0))).indices
            centers[cluster] = embedding[neighbours].mean(dim=0)
            reseeded[int(cluster)] = [int(v) for v in neighbours.tolist()]
    return centers, reseeded


def confidence_and_assignment(
    embedding: torch.Tensor,
    centers: torch.Tensor,
    tau: float,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cosine similarity, assignment confidence q_i, and hard cluster IDs."""
    if center is not None:
        embedding = embedding - center
        centers = centers - center
    similarity = F.cosine_similarity(embedding.unsqueeze(1), centers.unsqueeze(0), dim=2)
    probabilities = F.softmax(similarity / tau, dim=1)
    confidence, assignment = probabilities.max(dim=1)
    return similarity, confidence, assignment


CONFIDENCE_TAU_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
CONFIDENCE_DELTA_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def confidence_grid(
    embedding: torch.Tensor,
    centers: torch.Tensor,
    tau_grid: Sequence[float] = CONFIDENCE_TAU_GRID,
    delta_grid: Sequence[float] = CONFIDENCE_DELTA_GRID,
    *,
    center: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Measure confident-spot fractions over (tau, delta) at initialization time."""
    with torch.no_grad():
        table: dict[str, dict[str, float]] = {}
        for tau in tau_grid:
            _, confidence, _ = confidence_and_assignment(embedding, centers, float(tau), center=center)
            table[f"{tau:g}"] = {
                f"{delta:g}": float((confidence > delta).float().mean()) for delta in delta_grid
            }
    return table


class PrototypeTrainer(FixedTargetTrainer):
    """HGT trainer pairing KL reconstruction with confidence-gated prototype clustering."""

    def __init__(
        self,
        rna_matrix: Any,
        batches: Sequence[dict[str, Any]],
        pseudo_labels: Sequence[int],
        config: PrototypeConfig,
        device: torch.device,
        spatial_edge_index: object | None = None,
        selection: dict[int, Sequence[int]] | None = None,
    ) -> None:
        super().__init__(
            rna_matrix,
            batches,
            pseudo_labels,
            config,
            device,
            spatial_edge_index=spatial_edge_index,
            selection=selection,
        )
        self.config: PrototypeConfig = config
        self.n_clusters = int(config.target_k or np.unique(self.pseudo_labels).size)
        self.centers: torch.Tensor | None = None
        self.assignment = np.full(rna_matrix.shape[1], -1, dtype=np.int64)
        self.initial_assignment: np.ndarray | None = None
        self.cluster_init: dict[str, Any] = {}
        self.center_vector: torch.Tensor | None = None
        self.delta_value = float(config.delta)
        self.refresh_trace: list[dict[str, Any]] = []

    def _core_embedding(self, prepared: list[dict[str, Any]]) -> np.ndarray:
        n_cells = self.rna_matrix.shape[1]
        embedding = np.zeros((n_cells, self.config.n_hidden), dtype=np.float32)
        filled = np.zeros(n_cells, dtype=bool)
        was_training = self.gnn.training
        self.gnn.eval()
        try:
            with torch.no_grad():
                for payload in prepared:
                    node_rep = self.gnn.forward(
                        payload["node_feature"],
                        payload["node_type"],
                        payload["edge_index"],
                        payload["edge_type"],
                    )
                    cell_emb = node_rep[payload["node_type"] == CELL_TYPE]
                    for position in payload["core_positions"].cpu().tolist():
                        global_id = payload["cell_index"][int(position)]
                        embedding[global_id] = cell_emb[int(position)].detach().cpu().numpy()
                        filled[global_id] = True
        finally:
            self.gnn.train(was_training)
        if not bool(filled.all()):
            raise RuntimeError("some spots were never visited as a core cell")
        return embedding

    def _synchronise_partition(
        self, embedding: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor, int]:
        assert self.centers is not None
        labels = self.assignment.copy()
        reseeded_total = 0
        confidence = torch.zeros(embedding.size(0), device=embedding.device)
        for _ in range(4):
            _, confidence, predicted = confidence_and_assignment(
                embedding, self.centers, self.config.tau, center=self.center_vector
            )
            candidate = predicted.cpu().numpy().astype(np.int64)
            if np.unique(candidate).size < self.n_clusters:
                self.centers, reseeded = prototypes_from_assignment(
                    embedding,
                    predicted,
                    self.n_clusters,
                    previous=self.centers,
                    confidence=confidence,
                    preserve_k=self.config.preserve_k,
                    center=self.center_vector,
                )
                for cluster, members in reseeded.items():
                    if members:
                        candidate[np.asarray(members, dtype=np.int64)] = int(cluster)
                reseeded_total += len(reseeded)
                labels = candidate
                label_tensor = torch.from_numpy(labels).to(self.device)
                self.centers, _ = prototypes_from_assignment(
                    embedding, label_tensor, self.n_clusters, preserve_k=False, center=self.center_vector
                )
                self.assignment = labels
                continue

            label_tensor = torch.from_numpy(candidate).to(self.device)
            self.centers, _ = prototypes_from_assignment(
                embedding, label_tensor, self.n_clusters, preserve_k=False, center=self.center_vector
            )
            _, confidence, final_assignment = confidence_and_assignment(
                embedding, self.centers, self.config.tau, center=self.center_vector
            )
            labels = final_assignment.cpu().numpy().astype(np.int64)
            self.assignment = labels
            if np.unique(labels).size == self.n_clusters:
                return labels, confidence, reseeded_total

        self.assignment = labels
        return labels, confidence, reseeded_total

    def _initialise_clusters(self, prepared: list[dict[str, Any]]) -> dict[str, Any]:
        embedding = self._core_embedding(prepared)
        if self.config.init_source == "a0_pseudo":
            labels = np.asarray(self.pseudo_labels, dtype=np.int64)
            info: dict[str, Any] = {
                "selection": "a0_pseudo",
                "resolution": None,
                "k": int(np.unique(labels).size),
            }
        else:
            labels, info = leiden_on_embedding(
                embedding,
                n_neighbors=self.config.leiden_neighbors,
                resolution=self.config.leiden_resolution,
                target_k=self.config.target_k,
                seed=self.config.seed,
            )
        self.n_clusters = int(np.unique(labels).size)
        self.assignment = labels.astype(np.int64)
        self.initial_assignment = self.assignment.copy()
        tensor = torch.from_numpy(embedding).to(self.device)
        self.center_vector = embedding_center(tensor, self.config.center_embedding)
        self.centers, _ = prototypes_from_assignment(
            tensor,
            torch.from_numpy(self.assignment).to(self.device),
            self.n_clusters,
            preserve_k=self.config.preserve_k,
            center=self.center_vector,
        )
        self.refresh_trace = []
        info["init_source"] = str(self.config.init_source)
        info["n_clusters"] = self.n_clusters
        _, confidence, _ = confidence_and_assignment(
            tensor, self.centers, self.config.tau, center=self.center_vector
        )
        self.delta_value = resolve_delta(
            confidence,
            mode=self.config.delta_mode,
            absolute=self.config.delta,
            quantile=self.config.delta_quantile,
        )
        info["center_embedding"] = bool(self.config.center_embedding)
        info["delta_mode"] = str(self.config.delta_mode)
        info["delta_quantile"] = float(self.config.delta_quantile)
        info["delta_effective_at_init"] = self.delta_value
        info["confident_fraction_at_init"] = float((confidence > self.delta_value).float().mean())
        info["geometry_raw"] = frame_geometry(tensor, self.centers, None)
        info["geometry_active"] = frame_geometry(tensor, self.centers, self.center_vector)
        info["confidence_grid"] = confidence_grid(tensor, self.centers, center=self.center_vector)
        self.cluster_init = info
        print(
            f"[prototype] init: selection={info.get('selection')} "
            f"resolution={info.get('resolution')} K0={self.n_clusters} "
            f"centered={info['center_embedding']} delta_mode={info['delta_mode']} "
            f"delta={self.delta_value:.4f} confident={info['confident_fraction_at_init']:.3f}",
            flush=True,
        )
        return info

    def _refresh_clusters(self, prepared: list[dict[str, Any]]) -> dict[str, Any]:
        embedding = torch.from_numpy(self._core_embedding(prepared)).to(self.device)
        assert self.centers is not None
        self.center_vector = embedding_center(embedding, self.config.center_embedding)
        _, confidence, assignment = confidence_and_assignment(
            embedding, self.centers, self.config.tau, center=self.center_vector
        )
        previous = self.assignment.copy()
        self.assignment = assignment.cpu().numpy().astype(np.int64)
        nonempty_before = int(np.unique(self.assignment).size)
        reseeded: dict[int, list[int]] = {}
        if self.config.confident_only_refresh:
            threshold = resolve_delta(
                confidence,
                mode=self.config.delta_mode,
                absolute=self.config.delta,
                quantile=self.config.delta_quantile,
            )
            # Work in the same centered coordinate system used for assignment.
            # ``refresh_prototypes`` computes the confident-only centroid and
            # returns the exact mask used for that update.
            active_embedding = embedding if self.center_vector is None else embedding - self.center_vector
            active_centers = self.centers if self.center_vector is None else self.centers - self.center_vector
            previous_tensor = torch.from_numpy(previous).to(self.device)
            label_update = high_confidence_label_update(
                active_embedding,
                active_centers,
                threshold=threshold,
                tau=self.config.tau,
                previous_labels=previous_tensor,
            )
            refreshed = refresh_prototypes(
                active_embedding,
                active_centers,
                threshold=threshold,
                tau=self.config.tau,
                momentum=0.0,
            )
            self.centers = (
                refreshed.state.prototypes
                if self.center_vector is None
                else refreshed.state.prototypes + self.center_vector
            )
            self.assignment = label_update.labels.cpu().numpy().astype(np.int64)
            # Empty confident groups still need the existing preserve-K
            # reseeding policy.  It is applied only to those empty slots.
            empty = torch.where(refreshed.state.counts == 0)[0].tolist()
            if empty and self.config.preserve_k:
                reseed_centers, reseeded = prototypes_from_assignment(
                    embedding,
                    torch.from_numpy(self.assignment).to(self.device),
                    self.n_clusters,
                    previous=self.centers,
                    confidence=confidence,
                    preserve_k=True,
                    center=self.center_vector,
                )
                for cluster in empty:
                    self.centers[cluster] = reseed_centers[cluster]
        else:
            self.centers, reseeded = prototypes_from_assignment(
                embedding,
                assignment,
                self.n_clusters,
                previous=self.centers,
                confidence=confidence,
                preserve_k=self.config.preserve_k,
                center=self.center_vector,
            )
        for cluster, members in reseeded.items():
            if members:
                self.assignment[np.asarray(members, dtype=np.int64)] = int(cluster)
        if self.config.confident_only_refresh:
            _, confidence_after, _ = confidence_and_assignment(
                embedding, self.centers, self.config.tau, center=self.center_vector
            )
            reseeded_count = len(reseeded)
        else:
            self.assignment, confidence_after, extra_reseeded = self._synchronise_partition(embedding)
            reseeded_count = len(reseeded) + extra_reseeded
        changed = float(np.mean(previous != self.assignment)) if previous.size else 0.0
        sizes = np.bincount(self.assignment, minlength=self.n_clusters).tolist()
        self.delta_value = resolve_delta(
            confidence_after,
            mode=self.config.delta_mode,
            absolute=self.config.delta,
            quantile=self.config.delta_quantile,
        )
        probes = torch.tensor([0.1, 0.5, 0.9], device=confidence_after.device)
        spread = torch.quantile(confidence_after.detach().float(), probes).tolist()
        stats: dict[str, Any] = {
            "changed_fraction": changed,
            "delta_effective": self.delta_value,
            "confident_fraction": float((confidence_after > self.delta_value).float().mean()),
            "confidence_mean": float(confidence_after.mean()),
            "confidence_p10": float(spread[0]),
            "confidence_p50": float(spread[1]),
            "confidence_p90": float(spread[2]),
            "n_nonempty_before_reseed": nonempty_before,
            "n_nonempty": int(np.unique(self.assignment).size),
            "n_reseeded": reseeded_count,
            **frame_geometry(embedding, self.centers, self.center_vector),
        }
        trace = getattr(self, "refresh_trace", None)
        if trace is None:
            trace = self.refresh_trace = []
        trace.append({**stats, "cluster_sizes": sizes, "reseeded": sorted(reseeded)})
        flags = ""
        if stats["n_nonempty"] != self.n_clusters:
            flags += " [warn: cluster collapse]"
        if stats["confident_fraction"] <= 0.0:
            flags += " [warn: L_proto inactive]"
        if stats["prototype_cosine_max"] > 0.99:
            flags += " [warn: prototype collinearity]"
        print(
            f"[prototype] refresh: changed={changed:.3f} "
            f"confident={stats['confident_fraction']:.3f} "
            f"delta={stats['delta_effective']:.4f} "
            f"proto_cos={stats['prototype_cosine_max']:.4f} "
            f"nonempty={stats['n_nonempty']}/{self.n_clusters} sizes={sizes}{flags}",
            flush=True,
        )
        return stats

    def _proto_loss(
        self, core_emb: torch.Tensor, core_labels: torch.Tensor
    ) -> tuple[torch.Tensor, int, float]:
        assert self.centers is not None
        similarity, confidence, _ = confidence_and_assignment(
            core_emb, self.centers, self.config.tau, center=self.center_vector
        )
        if self.config.delta_mode == "quantile":
            threshold = resolve_delta(
                confidence,
                mode="quantile",
                absolute=self.config.delta,
                quantile=self.config.delta_quantile,
            )
        else:
            threshold = self.delta_value
        keep = confidence > threshold
        n_kept = int(keep.sum())
        if n_kept == 0:
            return torch.zeros((), device=core_emb.device, dtype=core_emb.dtype), 0, threshold
        logits = similarity[keep] / self.config.tau
        return F.cross_entropy(logits, core_labels[keep]), n_kept, threshold

    def train(self, progress: bool = True) -> list[dict[str, float]]:
        from pathlib import Path
        from tqdm import tqdm

        prepared = [self._build_batch(batch) for batch in self.batches]
        history: list[dict[str, float]] = []
        for epoch in tqdm(range(self.config.epochs), desc="Epochs", disable=not progress):
            epoch_number = epoch + 1
            self.gnn.train()
            if epoch_number == self.config.warmup_epochs + 1 and self.centers is None:
                info = self._initialise_clusters(prepared)
                history.append(
                    {
                        "epoch": epoch_number - 0.5,
                        "event": 1.0,
                        "init_k": float(info["n_clusters"]),
                        "init_resolution": (
                            float(info["resolution"]) if info.get("resolution") is not None else -1.0
                        ),
                    }
                )
            elif (
                self.centers is not None
                and epoch_number > self.config.warmup_epochs + 1
                and (epoch_number - self.config.warmup_epochs - 1) % self.config.update_interval == 0
            ):
                refresh = self._refresh_clusters(prepared)
                history.append({"epoch": epoch_number - 0.5, "event": 2.0, **refresh})

            dynamic = self.centers is not None
            sums = {"loss": 0.0, "kl": 0.0, "proto": 0.0, "kept": 0.0, "delta": 0.0}
            for payload in prepared:
                node_rep = self.gnn.forward(
                    payload["node_feature"],
                    payload["node_type"],
                    payload["edge_index"],
                    payload["edge_type"],
                )
                cell_emb = node_rep[payload["node_type"] == CELL_TYPE]
                gene_emb = node_rep[payload["node_type"] == GENE_TYPE]
                loss_kl = self._reconstruction_loss(cell_emb, gene_emb, payload["kl_target"])

                if dynamic:
                    core_positions = payload["core_positions"]
                    core_emb = cell_emb[core_positions]
                    global_ids = [payload["cell_index"][int(p)] for p in core_positions.cpu()]
                    core_labels = torch.from_numpy(self.assignment[global_ids]).long().to(self.device)
                    loss_proto, n_kept, threshold = self._proto_loss(core_emb, core_labels)
                    loss = loss_kl + self.config.lambda_proto * loss_proto
                    sums["proto"] += float(loss_proto.detach())
                    sums["kept"] += n_kept
                    sums["delta"] += threshold
                else:
                    loss = loss_kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                sums["loss"] += float(loss.detach())
                sums["kl"] += float(loss_kl.detach())

            n = len(prepared)
            history.append(
                {
                    "epoch": epoch_number,
                    "stage": 2.0 if dynamic else 1.0,
                    "loss": sums["loss"] / n,
                    "reconstruction_loss": sums["kl"] / n,
                    "proto_loss": sums["proto"] / n,
                    "confident_spots": sums["kept"] / n,
                    "delta_batch_mean": sums["delta"] / n,
                }
            )

            if (
                self.config.probe_every > 0
                and self.config.probe_dir
                and not dynamic
                and epoch_number % self.config.probe_every == 0
            ):
                folder = Path(self.config.probe_dir)
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"warmup_embedding_ep{epoch_number}.npy"
                np.save(target, self._core_embedding(prepared))

        if self.centers is None:
            self._initialise_clusters(prepared)
        else:
            final = self._refresh_clusters(prepared)
            history.append({"epoch": float(self.config.epochs) + 0.5, "event": 3.0, **final})
        return history

    def predict(self) -> dict[str, np.ndarray]:
        """Assign spots by prototype cosine similarity."""
        if self.centers is None:
            raise RuntimeError("prototypes were never initialised")
        prepared = [self._build_batch(batch) for batch in self.batches]
        embedding = self._core_embedding(prepared)
        tensor = torch.from_numpy(embedding).to(self.device)
        assignment, confidence, _ = self._synchronise_partition(tensor)
        return {
            "pred_label": assignment,
            "cell_embedding": embedding,
            "confidence": confidence.cpu().numpy().astype(np.float32),
        }
