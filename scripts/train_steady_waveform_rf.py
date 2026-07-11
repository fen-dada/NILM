"""Train a paper-2-style random forest from steady event waveforms.

The workflow uses minute-level features only to shortlist candidate files. For
each candidate it returns to the raw binary records, locates the strongest
cycle-level step, aligns steady cycles at the rising Ua zero crossing, and
builds the event signature as:

    event current = mean(after steady cycles) - mean(before steady cycles)

Submeter folder names are weak load labels. Samples are split chronologically
within each class so the later 20% is reserved for preliminary testing. The
same feature construction is then applied to candidate main-meter events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from detect_events import DEFAULT_FEATURES, compute_deltas, grouped_rows, load_rows, parse_float, score_events
from lightweight_rf import RandomForest, classification_rows, confusion_rows, standardize_train_test, write_csv
from parse_waveform import inspect_waveform_file, parse_waveform_file


CURRENT_INDICES = (3, 4, 5)
VOLTAGE_INDICES = (0, 1, 2)


def candidate_events(
    rows: list[dict[str, str]],
    categories: set[str],
    min_score: float,
) -> list[dict[str, object]]:
    groups = {name: values for name, values in grouped_rows(rows).items() if name in categories}
    scored = score_events(compute_deltas(groups, DEFAULT_FEATURES), DEFAULT_FEATURES)
    candidates = [
        row
        for row in scored
        if parse_float(row.get("event_score")) >= min_score and row.get("triggered_features")
    ]
    return sorted(candidates, key=lambda row: (str(row.get("category", "")), str(row.get("timestamp", ""))))


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (prefix[width:] - prefix[:-width]) / width


def cycle_metrics(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = waveforms.astype(np.float64)
    current = values[:, CURRENT_INDICES, :]
    voltage = values[:, VOLTAGE_INDICES, :]
    current_rms = np.sqrt(np.mean(current * current, axis=2)).mean(axis=1)
    active_like = np.mean(np.sum(voltage * current, axis=1), axis=1)
    return current_rms, active_like


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale <= 0 or not math.isfinite(scale):
        scale = float(np.std(values))
    return scale if scale > 0 and math.isfinite(scale) else 1.0


def locate_step(waveforms: np.ndarray, steady_cycles: int) -> tuple[int, float, float, float]:
    current_rms, active_like = cycle_metrics(waveforms)
    n = len(current_rms)
    if n < steady_cycles * 2 + 3:
        raise ValueError("not enough waveform cycles around candidate event")

    current_mean = rolling_mean(current_rms, steady_cycles)
    power_mean = rolling_mean(active_like, steady_cycles)
    positions = np.arange(steady_cycles, n - steady_cycles + 1)
    current_delta = current_mean[positions] - current_mean[positions - steady_cycles]
    power_delta = power_mean[positions] - power_mean[positions - steady_cycles]

    current_score = np.abs(current_delta) / robust_scale(np.diff(current_rms))
    power_score = np.abs(power_delta) / robust_scale(np.diff(active_like))
    combined = current_score + power_score
    best = int(np.argmax(combined))
    return (
        int(positions[best]),
        float(current_delta[best]),
        float(power_delta[best]),
        float(combined[best]),
    )


def rising_zero_index(voltage: np.ndarray) -> int:
    centered = voltage.astype(np.float64) - float(np.mean(voltage))
    crossings = np.flatnonzero((centered[:-1] <= 0) & (centered[1:] > 0)) + 1
    if crossings.size:
        slopes = centered[(crossings + 1) % centered.size] - centered[(crossings - 1) % centered.size]
        return int(crossings[int(np.argmax(slopes))])
    return int(np.argmin(np.abs(centered)))


def align_cycles(cycles: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(cycles, dtype=np.float64)
    for index, cycle in enumerate(cycles):
        shift = rising_zero_index(cycle[0])
        aligned[index] = np.roll(cycle, -shift, axis=1)
    return aligned


def downsample_waveform(values: np.ndarray, points: int) -> np.ndarray:
    if values.shape[1] == points:
        return values
    source = np.arange(values.shape[1], dtype=np.float64)
    target = np.linspace(0, values.shape[1] - 1, points)
    return np.stack([np.interp(target, source, channel) for channel in values], axis=0)


def event_signature(
    before: np.ndarray,
    after: np.ndarray,
    points_per_phase: int,
) -> tuple[np.ndarray, float]:
    before_mean = align_cycles(before).mean(axis=0)
    after_mean = align_cycles(after).mean(axis=0)
    current_channels = list(CURRENT_INDICES)
    delta = after_mean[current_channels] - before_mean[current_channels]
    delta = downsample_waveform(delta, points_per_phase)

    magnitude = float(np.sqrt(np.mean(delta * delta)))
    if magnitude <= 0 or not math.isfinite(magnitude):
        raise ValueError("event waveform has zero or invalid magnitude")

    # Shape drives the load class; magnitude remains as one extra feature.
    normalized = delta / magnitude
    return np.concatenate((normalized.reshape(-1), np.asarray([math.log1p(magnitude)]))), magnitude


def load_event_window(
    raw_data_dir: Path,
    event: dict[str, object],
    previous_tail_cycles: int,
) -> tuple[np.ndarray, int]:
    previous_path = raw_data_dir / str(event["previous_relative_path"])
    current_path = raw_data_dir / str(event["relative_path"])
    previous_count = inspect_waveform_file(previous_path).record_count
    previous_start = max(previous_count - previous_tail_cycles, 0)
    previous = parse_waveform_file(previous_path, start_frame=previous_start)
    current = parse_waveform_file(current_path)
    tail = previous[-min(previous_tail_cycles, len(previous)) :]
    return np.concatenate((tail, current), axis=0), len(tail)


def build_sample(
    raw_data_dir: Path,
    event: dict[str, object],
    steady_cycles: int,
    previous_tail_cycles: int,
    points_per_phase: int,
) -> tuple[np.ndarray, dict[str, object]]:
    waveforms, boundary = load_event_window(raw_data_dir, event, previous_tail_cycles)
    position, current_delta, power_delta, step_score = locate_step(waveforms, steady_cycles)
    before = waveforms[position - steady_cycles : position]
    after = waveforms[position : position + steady_cycles]
    signature, magnitude = event_signature(before, after, points_per_phase)

    event_time = str(event.get("timestamp", ""))
    direction = "\u542f\u52a8" if current_delta >= 0 else "\u505c\u6b62"
    detail = {
        "\u7c7b\u522b": str(event.get("category", "")),
        "\u4e8b\u4ef6\u65f6\u95f4": event_time,
        "\u52a8\u4f5c": direction,
        "\u4e8b\u4ef6\u524d\u6587\u4ef6": str(event.get("previous_relative_path", "")),
        "\u4e8b\u4ef6\u540e\u6587\u4ef6": str(event.get("relative_path", "")),
        "\u5b9a\u4f4d\u5468\u671f\u5e8f\u53f7": position,
        "\u4f4d\u4e8e\u5f53\u524d\u6587\u4ef6\u7684\u5468\u671f\u5e8f\u53f7": position - boundary,
        "\u4e09\u76f8\u5e73\u5747\u7535\u6d41RMS\u53d8\u5316\u91cf": round(current_delta, 6),
        "\u6709\u529f\u7279\u5f81\u53d8\u5316\u91cf": round(power_delta, 6),
        "\u5468\u671f\u7ea7\u7a81\u53d8\u5206\u6570": round(step_score, 6),
        "\u4e8b\u4ef6\u5dee\u5206\u6ce2\u5f62RMS": round(magnitude, 6),
        "\u5206\u949f\u7ea7\u4e8b\u4ef6\u5206\u6570": event.get("event_score", ""),
    }
    return signature, detail


def build_samples(
    raw_data_dir: Path,
    events: list[dict[str, object]],
    steady_cycles: int,
    previous_tail_cycles: int,
    points_per_phase: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    x_values: list[np.ndarray] = []
    labels: list[str] = []
    details: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        try:
            signature, detail = build_sample(
                raw_data_dir,
                event,
                steady_cycles,
                previous_tail_cycles,
                points_per_phase,
            )
            x_values.append(signature)
            labels.append(str(event.get("category", "")))
            details.append(detail)
        except Exception as exc:  # Keep a full audit trail for problematic raw files.
            failures.append(
                {
                    "\u7c7b\u522b": str(event.get("category", "")),
                    "\u4e8b\u4ef6\u65f6\u95f4": str(event.get("timestamp", "")),
                    "\u4e8b\u4ef6\u540e\u6587\u4ef6": str(event.get("relative_path", "")),
                    "\u5931\u8d25\u539f\u56e0": str(exc),
                }
            )
        if index % 50 == 0 or index == len(events):
            print(f"processed {index}/{len(events)} waveform events", flush=True)

    width = points_per_phase * 3 + 1
    x = np.asarray(x_values, dtype=np.float64) if x_values else np.empty((0, width), dtype=np.float64)
    return x, np.asarray(labels, dtype=object), details, failures


def chronological_stratified_split(
    details: list[dict[str, object]],
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(details):
        by_class[str(row["\u7c7b\u522b"])].append(index)

    train: list[int] = []
    test: list[int] = []
    for label, indices in sorted(by_class.items()):
        indices.sort(key=lambda index: (str(details[index]["\u4e8b\u4ef6\u65f6\u95f4"]), index))
        if len(indices) < 2:
            raise ValueError(f"class {label} has fewer than two usable events")
        split_at = max(1, min(len(indices) - 1, int(math.floor(len(indices) * train_ratio))))
        train.extend(indices[:split_at])
        test.extend(indices[split_at:])
    return np.asarray(sorted(train), dtype=int), np.asarray(sorted(test), dtype=int)


def feature_names(points_per_phase: int) -> list[str]:
    names = [
        f"{phase}\u76f8\u5dee\u5206\u7535\u6d41\u5f52\u4e00\u5316\u70b9_{point:03d}"
        for phase in ("A", "B", "C")
        for point in range(points_per_phase)
    ]
    return names + ["\u4e8b\u4ef6\u5dee\u5206\u6ce2\u5f62RMS\u5bf9\u6570"]


def prediction_rows(
    details: list[dict[str, object]],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detail, predicted, probability in zip(details, predictions, probabilities):
        row = dict(detail)
        row["\u9884\u6d4b\u5206\u8868"] = str(predicted)
        row["\u9884\u6d4b\u7ed3\u679c"] = f"{predicted}__{detail['\u52a8\u4f5c']}"
        row["\u7f6e\u4fe1\u5ea6"] = round(float(np.max(probability)), 6)
        for label, value in zip(classes, probability):
            row[f"\u5c5e\u4e8e{label}\u7684\u6982\u7387"] = round(float(value), 6)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF from phase-aligned steady event waveforms.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("E:/\u534e\u9e4f\u6ce2\u5f62\u6570\u636e"))
    parser.add_argument("--features", type=Path, default=Path("outputs/features/minute_features.csv"))
    parser.add_argument("--target-category", default="\u914d\u7535\u623f\uff08\u603b\uff09")
    parser.add_argument("--train-min-score", type=float, default=100.0)
    parser.add_argument("--target-min-score", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--steady-cycles", type=int, default=20)
    parser.add_argument("--previous-tail-cycles", type=int, default=200)
    parser.add_argument("--points-per-phase", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--max-thresholds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/model_reports/steady_waveform_rf"))
    parser.add_argument("--model-output", type=Path, default=Path("models/steady_waveform_rf.json"))
    args = parser.parse_args()

    if not 0.5 <= args.train_ratio < 1.0:
        raise SystemExit("train-ratio must be in [0.5, 1.0)")
    rows = load_rows(args.features)
    categories = {str(row.get("category", "")) for row in rows if row.get("category")}
    submeter_categories = categories - {args.target_category}
    train_events = candidate_events(rows, submeter_categories, args.train_min_score)
    target_events = candidate_events(rows, {args.target_category}, args.target_min_score)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    x, y, details, failures = build_samples(
        args.raw_data_dir,
        train_events,
        args.steady_cycles,
        args.previous_tail_cycles,
        args.points_per_phase,
    )
    write_csv(details, args.report_dir / "\u5206\u8868\u4e8b\u4ef6\u6ce2\u5f62\u6837\u672c.csv")
    write_csv(failures, args.report_dir / "\u6837\u672c\u6784\u5efa\u5931\u8d25\u8bb0\u5f55.csv")
    if len(details) < 2 or len(set(y.tolist())) < 2:
        raise SystemExit("not enough usable submeter waveform events")

    train_idx, test_idx = chronological_stratified_split(details, args.train_ratio)
    x_train, x_test, mean, std = standardize_train_test(x[train_idx], x[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    model = RandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_thresholds=args.max_thresholds,
        seed=args.seed,
    )
    model.fit(x_train, y_train)
    test_probability = model.predict_proba(x_test)
    y_pred = np.asarray([model.class_names[int(np.argmax(row))] for row in test_probability], dtype=object)
    accuracy = float(np.mean(y_pred == y_test))

    split_rows = []
    train_set = set(train_idx.tolist())
    for index, detail in enumerate(details):
        row = dict(detail)
        row["\u6570\u636e\u96c6\u7528\u9014"] = "\u8bad\u7ec3" if index in train_set else "\u6d4b\u8bd5"
        split_rows.append(row)
    write_csv(split_rows, args.report_dir / "\u8bad\u7ec3\u6d4b\u8bd5\u5212\u5206.csv")
    write_csv(classification_rows(y_test, y_pred, model.class_names), args.report_dir / "\u5206\u7c7b\u62a5\u544a.csv")
    write_csv(confusion_rows(y_test, y_pred, model.class_names), args.report_dir / "\u6df7\u6dc6\u77e9\u9635.csv")

    target_x, _, target_details, target_failures = build_samples(
        args.raw_data_dir,
        target_events,
        args.steady_cycles,
        args.previous_tail_cycles,
        args.points_per_phase,
    )
    write_csv(target_failures, args.report_dir / "\u603b\u8868\u6837\u672c\u6784\u5efa\u5931\u8d25\u8bb0\u5f55.csv")
    target_predictions: list[dict[str, object]] = []
    if len(target_details):
        target_scaled = (target_x - mean) / std
        probabilities = model.predict_proba(target_scaled)
        predictions = np.asarray(
            [model.class_names[int(np.argmax(row))] for row in probabilities],
            dtype=object,
        )
        target_predictions = prediction_rows(target_details, predictions, probabilities, model.class_names)
    write_csv(target_predictions, args.report_dir / "\u603b\u8868\u4e8b\u4ef6\u8bc6\u522b\u7ed3\u679c.csv")

    names = feature_names(args.points_per_phase)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict(names)
    payload["standardization"] = {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    payload["workflow"] = {
        "method": "phase-aligned steady waveform difference",
        "steady_cycles": args.steady_cycles,
        "points_per_phase": args.points_per_phase,
        "train_ratio": args.train_ratio,
        "split": "chronological within each class",
    }
    args.model_output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    metrics = {
        "\u65b9\u6cd5": "\u4e8b\u4ef6\u524d\u540e\u7a33\u6001\u5468\u671f\u76f8\u4f4d\u5bf9\u9f50\u5dee\u5206 + \u968f\u673a\u68ee\u6797",
        "\u6807\u7b7e\u6027\u8d28": "\u5206\u8868\u6587\u4ef6\u5939\u540d\u5f31\u6807\u7b7e",
        "\u5212\u5206\u65b9\u5f0f": "\u6bcf\u4e2a\u7c7b\u522b\u5185\u6309\u65f6\u95f4\u5148\u540e\u5212\u5206",
        "\u8bad\u7ec3\u6bd4\u4f8b": args.train_ratio,
        "\u5019\u9009\u5206\u8868\u4e8b\u4ef6\u6570": len(train_events),
        "\u53ef\u7528\u5206\u8868\u6837\u672c\u6570": len(details),
        "\u8bad\u7ec3\u6837\u672c\u6570": int(len(train_idx)),
        "\u6d4b\u8bd5\u6837\u672c\u6570": int(len(test_idx)),
        "\u8bad\u7ec3\u96c6\u7c7b\u522b\u5206\u5e03": dict(Counter(y_train.tolist())),
        "\u6d4b\u8bd5\u96c6\u7c7b\u522b\u5206\u5e03": dict(Counter(y_test.tolist())),
        "\u6d4b\u8bd5\u51c6\u786e\u7387": round(accuracy, 6),
        "\u603b\u8868\u5019\u9009\u4e8b\u4ef6\u6570": len(target_events),
        "\u603b\u8868\u6210\u529f\u8bc6\u522b\u6570": len(target_predictions),
        "\u7a33\u6001\u5468\u671f\u6570": args.steady_cycles,
        "\u6bcf\u76f8\u964d\u91c7\u6837\u70b9\u6570": args.points_per_phase,
        "\u8bf4\u660e": "\u4ec5\u4e3a\u540c\u4e00\u5929\u6570\u636e\u7684\u521d\u6b65\u9a8c\u8bc1\uff0c\u4e0d\u4ee3\u8868\u8de8\u5929\u6cdb\u5316\u51c6\u786e\u7387\u3002",
    }
    (args.report_dir / "\u8bc4\u4f30\u6307\u6807.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
