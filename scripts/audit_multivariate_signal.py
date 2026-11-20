"""Spatial-block cross-validation probes for RNA and RNA+ADT input signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def moran_i(values: np.ndarray, weights: sparse.csr_matrix) -> float:
    centered = values.astype(np.float64) - float(np.mean(values))
    denominator = float(centered @ centered)
    if denominator == 0.0:
        return 0.0
    s0 = float(weights.sum())
    return float(values.size / s0 * (centered @ (weights @ centered)) / denominator)


def moran_permutation_test(
    values: np.ndarray,
    weights: sparse.csr_matrix,
    *,
    rng: np.random.Generator,
    permutations: int,
) -> tuple[float, float]:
    observed = moran_i(values, weights)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        null[index] = moran_i(rng.permutation(values), weights)
    p_value = (1 + int(np.count_nonzero(null >= observed))) / (permutations + 1)
    return observed, float(p_value)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    total = len(values)
    for rank in range(total, 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * total / rank)
        adjusted[index] = running
    return adjusted.tolist()


def align_ids(ids: list[str], *, strip_prefix: bool) -> list[str]:
    if strip_prefix:
        return [value.removeprefix("s1-") for value in ids]
    return ids


def class_metrics(truth: np.ndarray, prediction: np.ndarray, classes: list[str]) -> dict:
    counts = pd.Series(truth).value_counts()
    rare_classes = [name for name in classes if counts[name] / len(truth) < 0.05]
    per_class = {}
    for name in classes:
        true_mask = truth == name
        pred_mask = prediction == name
        tp = int(np.count_nonzero(true_mask & pred_mask))
        fp = int(np.count_nonzero(~true_mask & pred_mask))
        fn = int(np.count_nonzero(true_mask & ~pred_mask))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": int(true_mask.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    rare_truth = np.isin(truth, rare_classes)
    rare_prediction = np.isin(prediction, rare_classes)
    tp = int(np.count_nonzero(rare_truth & rare_prediction))
    fp = int(np.count_nonzero(~rare_truth & rare_prediction))
    fn = int(np.count_nonzero(rare_truth & ~rare_prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    rare_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
        "rare_class_rule": "reference class share strictly below 5%",
        "rare_classes": rare_classes,
        "rare_union_precision": float(precision),
        "rare_union_recall": float(recall),
        "rare_union_f1": float(rare_f1),
        "per_class": per_class,
    }


def run_probe(
    rna_expression: sparse.spmatrix | np.ndarray,
    adt_expression: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    *,
    include_adt: bool,
) -> tuple[dict, np.ndarray, list[dict]]:
    predictions = np.empty(truth.size, dtype=object)
    fold_records = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(splitter.split(rna_expression, truth, groups), start=1):
        train_rna = rna_expression[train_index]
        test_rna = rna_expression[test_index]
        if sparse.issparse(train_rna):
            train_rna = train_rna.toarray()
            test_rna = test_rna.toarray()
        pca = PCA(n_components=40, svd_solver="randomized", random_state=0)
        train_rna_pca = pca.fit_transform(np.asarray(train_rna, dtype=np.float32))
        test_rna_pca = pca.transform(np.asarray(test_rna, dtype=np.float32))
        train_features = train_rna_pca
        test_features = test_rna_pca
        if include_adt:
            train_features = np.column_stack([train_features, adt_expression[train_index]])
            test_features = np.column_stack([test_features, adt_expression[test_index]])
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=0,
        )
        model.fit(train_features, truth[train_index])
        predictions[test_index] = model.predict(test_features)
        fold_records.append({
            "fold": fold,
            "held_out_blocks": sorted(int(value) for value in np.unique(groups[test_index])),
            "train_spots": int(train_index.size),
            "test_spots": int(test_index.size),
            "train_classes_missing": sorted(set(classes) - set(truth[train_index])),
            "test_class_counts": {
                name: int(np.count_nonzero(truth[test_index] == name))
                for name in classes
                if np.any(truth[test_index] == name)
            },
        })
    predicted = predictions.astype(str)
    return class_metrics(truth, predicted, classes), predicted, fold_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run label-conditional input-signal checks with spatially held-out blocks.")
    parser.add_argument("--rna-h5ad", type=Path, required=True)
    parser.add_argument("--adt-h5ad", type=Path, required=True)
    parser.add_argument("--label-key", default="final_annot")
    parser.add_argument("--out-directory", type=Path, default=ROOT / "experiments" / "runs" / "multivariate_signal_probe")
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()

    import scanpy as sc

    rna = sc.read_h5ad(args.rna_h5ad)
    adt = sc.read_h5ad(args.adt_h5ad)
    rna_ids = align_ids(rna.obs_names.astype(str).tolist(), strip_prefix=True)
    adt_ids = align_ids(adt.obs_names.astype(str).tolist(), strip_prefix=False)
    if len(set(rna_ids)) != len(rna_ids) or len(set(adt_ids)) != len(adt_ids):
        raise ValueError("RNA or ADT spot IDs are not unique after prefix normalization")
    if set(rna_ids) != set(adt_ids):
        raise ValueError("RNA and ADT spot-ID sets differ")
    adt_position = {spot_id: index for index, spot_id in enumerate(adt_ids)}
    adt = adt[[adt_position[spot_id] for spot_id in rna_ids], :].copy()

    truth = rna.obs[args.label_key].astype(str).to_numpy()
    classes = sorted(np.unique(truth).tolist())
    coordinates = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    groups = KMeans(n_clusters=5, n_init=20, random_state=0).fit_predict(coordinates)

    # PCA and per-spot normalization use expression values only. Labels enter after folds are fixed.
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.normalize_total(adt, target_sum=1e4)
    sc.pp.log1p(adt)
    adt_matrix = adt.X.toarray() if sparse.issparse(adt.X) else np.asarray(adt.X)
    adt_matrix = np.asarray(adt_matrix, dtype=np.float32)

    block_adjacency = NearestNeighbors(n_neighbors=11).fit(coordinates).kneighbors_graph(
        coordinates, mode="connectivity"
    )
    block_adjacency = block_adjacency.maximum(block_adjacency.T).tocsr()
    block_adjacency.setdiag(0)
    block_adjacency.eliminate_zeros()
    row_sums = np.asarray(block_adjacency.sum(axis=1)).reshape(-1)
    weights = sparse.diags(1.0 / np.maximum(row_sums, 1.0)) @ block_adjacency
    rng = np.random.default_rng(0)
    rare_classes = [name for name in classes if np.mean(truth == name) < 0.05]
    moran_rows = []
    for name in rare_classes + ["all_rare_classes"]:
        values = np.isin(truth, rare_classes) if name == "all_rare_classes" else (truth == name)
        observed, p_value = moran_permutation_test(
            values.astype(np.float64), weights, rng=rng, permutations=args.permutations
        )
        moran_rows.append({
            "domain": name,
            "support": int(values.sum()),
            "moran_i": observed,
            "permutation_p_one_sided": p_value,
        })
    adjusted = benjamini_hochberg([row["permutation_p_one_sided"] for row in moran_rows])
    for row, p_adjusted in zip(moran_rows, adjusted):
        row["permutation_p_bh"] = p_adjusted

    feature_sets = {
        "RNA_PCA40": False,
        "RNA_PCA40_plus_ADT31": True,
    }
    report = {
        "scope": "Conditional input-signal analysis using the supplied A1 final_annot labels.",
        "label_provenance": "The source and revision of final_annot remain unresolved; scores are conditional on that supplied reference.",
        "n_spots": int(truth.size),
        "n_reference_classes": len(classes),
        "preprocessing": {
            "rna": "per-spot normalize_total(1e4), log1p; PCA40 fitted on each training fold only",
            "adt": "per-spot normalize_total(1e4), log1p; matched to RNA by normalized spot ID",
            "labels_used_for_features": False,
            "labels_used_for_block_assignment": False,
        },
        "spatial_blocks": {
            "method": "KMeans on two-dimensional spot coordinates, five spatial regions, random_state=0",
            "cv": "5-fold GroupKFold, holding out complete spatial regions",
            "block_sizes": {str(int(group)): int(np.count_nonzero(groups == group)) for group in np.unique(groups)},
        },
        "classifier": "class-balanced multinomial logistic regression, C=1, fixed before evaluation",
        "moran": {
            "graph": "symmetric union of 10-nearest-neighbor spatial edges, row-standardized",
            "null": "global permutation of each binary label mask preserving its count",
            "permutations": args.permutations,
            "interpretation": "descriptive spatial clustering diagnostic; spot-level tests remain dependent",
            "domains": moran_rows,
        },
        "probes": {},
    }

    args.out_directory.mkdir(parents=True, exist_ok=True)
    class_codes = {name: index for index, name in enumerate(classes)}
    for name, include_adt in feature_sets.items():
        metrics, prediction, folds = run_probe(
            rna.X,
            adt_matrix,
            truth,
            groups,
            classes,
            include_adt=include_adt,
        )
        run_dir = args.out_directory / name
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "pred.npy", np.asarray([class_codes[value] for value in prediction], dtype=np.int16))
        (run_dir / "config.json").write_text(json.dumps({
            "feature_set": name,
            "classifier": report["classifier"],
            "spatial_blocks": report["spatial_blocks"],
            "class_order": classes,
        }, indent=2), encoding="utf-8")
        evaluation = {**metrics, "folds": folds, "class_order": classes}
        (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        report["probes"][name] = metrics
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f} "
            f"rare-union F1={metrics['rare_union_f1']:.4f}"
        )
        for rare in metrics["rare_classes"]:
            value = metrics["per_class"][rare]
            print(f"  {rare}: recall={value['recall']:.4f} F1={value['f1']:.4f} n={value['support']}")
    report_path = ROOT / "experiments" / "reports" / "multivariate_signal_probe.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Moran's I (label masks):")
    for row in moran_rows:
        print(f"  {row['domain']}: I={row['moran_i']:.4f}, p_BH={row['permutation_p_bh']:.4g}")
    print(f"-> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
