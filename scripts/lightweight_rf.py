"""Small random forest utilities used by NILM prototype scripts."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return
    with output.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    counts = Counter(labels.tolist())
    total = float(labels.size)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def majority_class(labels: np.ndarray) -> str:
    return Counter(labels.tolist()).most_common(1)[0][0]


@dataclass
class TreeNode:
    prediction: str
    probabilities: dict[str, float]
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int,
        max_thresholds: int,
        rng: random.Random,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.rng = rng
        self.root: TreeNode | None = None
        self.importances: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.importances = np.zeros(x.shape[1], dtype=np.float64)
        self.root = self._build(x, y, depth=0)
        total = float(np.sum(self.importances))
        if total > 0:
            self.importances = self.importances / total

    def _leaf(self, y: np.ndarray) -> TreeNode:
        counts = Counter(y.tolist())
        total = float(y.size)
        return TreeNode(
            prediction=majority_class(y),
            probabilities={label: count / total for label, count in counts.items()},
        )

    def _candidate_thresholds(self, values: np.ndarray) -> list[float]:
        unique = np.unique(values)
        if unique.size <= 1:
            return []
        if unique.size <= self.max_thresholds + 1:
            return [float((a + b) / 2.0) for a, b in zip(unique[:-1], unique[1:])]
        percentiles = np.linspace(5, 95, self.max_thresholds)
        return sorted(set(float(value) for value in np.percentile(values, percentiles)))

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        parent_gini = gini(y)
        best_feature: int | None = None
        best_threshold: float | None = None
        best_gain = 0.0
        feature_count = x.shape[1]
        candidates = self.rng.sample(range(feature_count), min(self.max_features, feature_count))

        for feature in candidates:
            values = x[:, feature]
            for threshold in self._candidate_thresholds(values):
                left_mask = values <= threshold
                left_count = int(np.sum(left_mask))
                right_count = y.size - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                left_gini = gini(y[left_mask])
                right_gini = gini(y[~left_mask])
                weighted = (left_count * left_gini + right_count * right_gini) / y.size
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_feature = feature
                    best_threshold = threshold
                    best_gain = gain
        return best_feature, best_threshold, best_gain

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        if depth >= self.max_depth or y.size < self.min_samples_leaf * 2 or len(set(y.tolist())) == 1:
            return self._leaf(y)

        feature, threshold, gain = self._best_split(x, y)
        if feature is None or threshold is None or gain <= 0:
            return self._leaf(y)

        assert self.importances is not None
        self.importances[feature] += gain * y.size
        left_mask = x[:, feature] <= threshold
        return TreeNode(
            prediction=majority_class(y),
            probabilities={label: count / y.size for label, count in Counter(y.tolist()).items()},
            feature=feature,
            threshold=threshold,
            left=self._build(x[left_mask], y[left_mask], depth + 1),
            right=self._build(x[~left_mask], y[~left_mask], depth + 1),
        )

    def predict_proba_one(self, row: np.ndarray, class_names: list[str]) -> np.ndarray:
        if self.root is None:
            raise ValueError("tree is not fitted")
        node = self.root
        while not node.is_leaf():
            assert node.feature is not None and node.threshold is not None and node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
        return np.asarray([node.probabilities.get(label, 0.0) for label in class_names], dtype=np.float64)

    def to_dict(self, feature_names: list[str]) -> dict[str, Any]:
        def encode(node: TreeNode) -> dict[str, Any]:
            result: dict[str, Any] = {"prediction": node.prediction, "probabilities": node.probabilities}
            if not node.is_leaf():
                assert node.feature is not None and node.threshold is not None and node.left is not None and node.right is not None
                result.update(
                    {
                        "feature": feature_names[node.feature],
                        "threshold": node.threshold,
                        "left": encode(node.left),
                        "right": encode(node.right),
                    }
                )
            return result

        if self.root is None:
            raise ValueError("tree is not fitted")
        return encode(self.root)


class RandomForest:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        max_thresholds: int,
        seed: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.seed = seed
        self.trees: list[DecisionTree] = []
        self.class_names: list[str] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = random.Random(self.seed)
        self.class_names = sorted(set(y.tolist()))
        self.trees = []
        importances = np.zeros(x.shape[1], dtype=np.float64)
        max_features = max(1, int(math.sqrt(x.shape[1])))

        for _ in range(self.n_estimators):
            sample_indices = np.asarray([rng.randrange(x.shape[0]) for _ in range(x.shape[0])], dtype=int)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                max_thresholds=self.max_thresholds,
                rng=random.Random(rng.randrange(1_000_000_000)),
            )
            tree.fit(x[sample_indices], y[sample_indices])
            self.trees.append(tree)
            if tree.importances is not None:
                importances += tree.importances

        total = float(np.sum(importances))
        self.feature_importances_ = importances / total if total > 0 else importances

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("forest is not fitted")
        probs = np.zeros((x.shape[0], len(self.class_names)), dtype=np.float64)
        for tree in self.trees:
            for index, row in enumerate(x):
                probs[index] += tree.predict_proba_one(row, self.class_names)
        return probs / len(self.trees)

    def to_dict(self, feature_names: list[str]) -> dict[str, Any]:
        return {
            "model_type": "lightweight_random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_thresholds": self.max_thresholds,
            "seed": self.seed,
            "classes": self.class_names,
            "feature_names": feature_names,
            "trees": [tree.to_dict(feature_names) for tree in self.trees],
        }


def classification_rows(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "类别": label,
                "精确率": round(precision, 6),
                "召回率": round(recall, 6),
                "F1": round(f1, 6),
                "样本数": support,
            }
        )
    return rows


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actual in labels:
        row: dict[str, object] = {"真实类别": actual}
        for predicted in labels:
            row[f"预测为_{predicted}"] = int(np.sum((y_true == actual) & (y_pred == predicted)))
        rows.append(row)
    return rows
