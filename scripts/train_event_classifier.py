"""Train a first-pass event classifier from rule-based pseudo labels.

The project currently has no external equipment operation log, so labels are
pseudo labels generated from voltage/current features. This model is therefore a
baseline for workflow validation, not a ground-truth equipment recognizer.

The implementation includes a small random forest classifier to avoid adding a
heavy dependency before the project environment is finalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


METADATA_COLUMNS = {
    "category",
    "timestamp",
    "relative_path",
    "previous_relative_path",
    "triggered_features",
}


def parse_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return
    with output.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    if column in METADATA_COLUMNS:
        return False
    seen = False
    for row in rows:
        value = row.get(column, "")
        if value == "":
            continue
        seen = True
        if not math.isfinite(parse_float(value)):
            return False
    return seen


def build_dataset(
    events: list[dict[str, str]], labels: list[dict[str, str]], include_uncertain: bool
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    label_by_path = {row["事件后文件"]: row for row in labels}
    feature_names = [name for name in events[0].keys() if is_numeric_column(events, name)]
    dataset_rows: list[dict[str, str]] = []
    x_values: list[list[float]] = []
    y_values: list[str] = []

    for event in events:
        label_row = label_by_path.get(event["relative_path"])
        if not label_row:
            continue
        label = label_row.get("自动判断", "")
        if not label or (label == "不确定" and not include_uncertain):
            continue
        values = [parse_float(event.get(name)) for name in feature_names]
        if not all(math.isfinite(value) for value in values):
            continue
        merged = dict(event)
        merged["label"] = label
        dataset_rows.append(merged)
        x_values.append(values)
        y_values.append(label)

    return np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=object), feature_names, dataset_rows


def time_split(rows: list[dict[str, str]], train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(len(rows)))
    indices.sort(key=lambda index: (rows[index].get("timestamp", ""), rows[index].get("relative_path", "")))
    split_at = max(1, min(len(indices) - 1, int(round(len(indices) * train_ratio))))
    return np.asarray(indices[:split_at], dtype=int), np.asarray(indices[split_at:], dtype=int)


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

    def fit(self, x: np.ndarray, y: np.ndarray, n_classes: int) -> None:
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
            tree.fit(x[sample_indices], y[sample_indices], len(self.class_names))
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.asarray([self.class_names[int(np.argmax(row))] for row in probs], dtype=object)

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


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


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
                "label": label,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
            }
        )
    return rows


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actual in labels:
        row: dict[str, object] = {"actual": actual}
        for predicted in labels:
            row[f"pred_{predicted}"] = int(np.sum((y_true == actual) & (y_pred == predicted)))
        rows.append(row)
    return rows


def prediction_rows(rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, labels: list[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row, actual, predicted, prob_row in zip(rows, y_true, y_pred, probs):
        result: dict[str, object] = {
            "timestamp": row.get("timestamp", ""),
            "category": row.get("category", ""),
            "relative_path": row.get("relative_path", ""),
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "confidence": round(float(np.max(prob_row)), 6),
        }
        for label, value in zip(labels, prob_row):
            result[f"prob_{label}"] = round(float(value), 6)
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first-pass NILM event classifier.")
    parser.add_argument("--events", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels/auto_event_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_reports/event_classifier"))
    parser.add_argument("--model-output", type=Path, default=Path("models/event_classifier_rf.json"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max-thresholds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    events = load_csv(args.events)
    labels = load_csv(args.labels)
    x, y, feature_names, dataset_rows = build_dataset(events, labels, args.include_uncertain)
    train_idx, test_idx = time_split(dataset_rows, args.train_ratio)

    x_train, x_test, mean, std = standardize_train_test(x[train_idx], x[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = RandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_thresholds=args.max_thresholds,
        seed=args.seed,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    class_names = model.class_names
    accuracy = float(np.mean(y_pred == y_test)) if y_test.size else math.nan

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "note": "Pseudo-label model. Labels are generated from voltage/current rules, not external operation logs.",
        "event_rows": len(dataset_rows),
        "train_rows": int(train_idx.size),
        "test_rows": int(test_idx.size),
        "train_ratio": args.train_ratio,
        "include_uncertain": args.include_uncertain,
        "classes": class_names,
        "train_class_counts": dict(Counter(y_train.tolist())),
        "test_class_counts": dict(Counter(y_test.tolist())),
        "accuracy": round(accuracy, 6),
        "feature_count": len(feature_names),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(classification_rows(y_test, y_pred, class_names), args.output_dir / "classification_report.csv")
    write_csv(confusion_rows(y_test, y_pred, class_names), args.output_dir / "confusion_matrix.csv")
    write_csv(
        prediction_rows([dataset_rows[index] for index in test_idx], y_test, y_pred, probs, class_names),
        args.output_dir / "test_predictions.csv",
    )

    importances = model.feature_importances_ if model.feature_importances_ is not None else np.zeros(len(feature_names))
    importance_rows = [
        {"feature": feature, "importance": round(float(value), 8)}
        for feature, value in sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    ]
    write_csv(importance_rows, args.output_dir / "feature_importance.csv")

    model_payload = model.to_dict(feature_names)
    model_payload["standardization"] = {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    args.model_output.write_text(json.dumps(model_payload, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote reports to {args.output_dir}")
    print(f"wrote model to {args.model_output}")


if __name__ == "__main__":
    main()
