"""Run the trained event classifier on candidate event feature rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return
    with output.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def traverse_tree(node: dict[str, Any], row: np.ndarray, feature_index: dict[str, int]) -> dict[str, float]:
    while "feature" in node:
        feature = node["feature"]
        threshold = float(node["threshold"])
        value = row[feature_index[feature]]
        node = node["left"] if value <= threshold else node["right"]
    return {str(label): float(probability) for label, probability in node.get("probabilities", {}).items()}


def predict_one(model: dict[str, Any], row: np.ndarray, feature_index: dict[str, int]) -> dict[str, float]:
    classes = [str(label) for label in model["classes"]]
    probabilities = {label: 0.0 for label in classes}
    trees = model["trees"]
    for tree in trees:
        tree_probs = traverse_tree(tree, row, feature_index)
        for label in classes:
            probabilities[label] += tree_probs.get(label, 0.0)
    return {label: probability / len(trees) for label, probability in probabilities.items()}


def build_feature_matrix(rows: list[dict[str, str]], model: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    feature_names = [str(name) for name in model["feature_names"]]
    valid_indices: list[int] = []
    values: list[list[float]] = []
    for index, row in enumerate(rows):
        sample = [parse_float(row.get(feature)) for feature in feature_names]
        if all(math.isfinite(value) for value in sample):
            valid_indices.append(index)
            values.append(sample)
    return np.asarray(values, dtype=np.float64), valid_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict event classes using a trained NILM event classifier.")
    parser.add_argument("--events", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/event_classifier_rf.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model_reports/event_classifier/predictions_all.csv"))
    args = parser.parse_args()

    model = json.loads(args.model.read_text(encoding="utf-8"))
    rows = load_csv(args.events)
    x_raw, valid_indices = build_feature_matrix(rows, model)
    feature_names = [str(name) for name in model["feature_names"]]
    feature_index = {feature: index for index, feature in enumerate(feature_names)}
    standardization = model["standardization"]
    mean = np.asarray(standardization["mean"], dtype=np.float64)
    std = np.asarray(standardization["std"], dtype=np.float64)
    std[std == 0] = 1.0
    x = (x_raw - mean) / std

    output_rows: list[dict[str, object]] = []
    classes = [str(label) for label in model["classes"]]
    for row_index, sample in zip(valid_indices, x):
        source = rows[row_index]
        probs = predict_one(model, sample, feature_index)
        prediction = max(classes, key=lambda label: probs.get(label, 0.0))
        result: dict[str, object] = {
            "timestamp": source.get("timestamp", ""),
            "category": source.get("category", ""),
            "relative_path": source.get("relative_path", ""),
            "previous_relative_path": source.get("previous_relative_path", ""),
            "predicted": prediction,
            "confidence": round(probs[prediction], 6),
            "event_score": source.get("event_score", ""),
            "triggered_features": source.get("triggered_features", ""),
        }
        for label in classes:
            result[f"prob_{label}"] = round(probs.get(label, 0.0), 6)
        output_rows.append(result)

    write_csv(output_rows, args.output)
    print(f"read {len(rows)} event rows")
    print(f"predicted {len(output_rows)} rows")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
