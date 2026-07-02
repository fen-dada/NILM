"""Detect candidate load events from minute-level waveform features.

This is an unsupervised first pass. It compares each minute with the previous
minute in the same category and flags unusually large changes in current and
power-like features. Output rows are candidates for later waveform review and
manual labeling, not final appliance IDs.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_FEATURES = (
    "Ia_rms",
    "Ib_rms",
    "Ic_rms",
    "I0_rms",
    "current_rms_avg",
    "P_total",
    "S_total",
    "power_factor_like",
    "current_unbalance",
)


def parse_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def mad(values: list[float], center: float) -> float:
    return median([abs(value - center) for value in values])


def robust_stats(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    center = median(finite)
    scale = 1.4826 * mad(finite, center)
    if scale == 0:
        nonzero = [abs(value - center) for value in finite if value != center]
        scale = median(nonzero) if nonzero else 1.0
    return center, scale


def row_time_key(row: dict[str, str]) -> tuple[str, str, str]:
    timestamp = row.get("first_timestamp", "")
    if timestamp:
        return timestamp, row.get("relative_path", ""), row.get("file_name", "")
    return row.get("date", ""), row.get("hour", ""), row.get("file_name", "")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def grouped_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("category", "")].append(row)
    for category in groups:
        groups[category].sort(key=row_time_key)
    return groups


def time_gap_seconds(prev: dict[str, str], curr: dict[str, str]) -> float:
    a = prev.get("first_timestamp", "")
    b = curr.get("first_timestamp", "")
    if not a or not b:
        return math.nan
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except ValueError:
        return math.nan


def compute_deltas(groups: dict[str, list[dict[str, str]]], features: tuple[str, ...]) -> list[dict[str, object]]:
    deltas: list[dict[str, object]] = []
    for category, rows in groups.items():
        for prev, curr in zip(rows, rows[1:]):
            row: dict[str, object] = {
                "category": category,
                "timestamp": curr.get("first_timestamp", ""),
                "relative_path": curr.get("relative_path", ""),
                "previous_relative_path": prev.get("relative_path", ""),
                "gap_seconds": round(time_gap_seconds(prev, curr), 8),
            }
            for feature in features:
                current_value = parse_float(curr.get(feature))
                previous_value = parse_float(prev.get(feature))
                delta = current_value - previous_value
                row[f"{feature}_prev"] = previous_value
                row[f"{feature}_curr"] = current_value
                row[f"{feature}_delta"] = delta
                row[f"{feature}_abs_delta"] = abs(delta) if math.isfinite(delta) else math.nan
            deltas.append(row)
    return deltas


def score_events(deltas: list[dict[str, object]], features: tuple[str, ...]) -> list[dict[str, object]]:
    stats = {
        feature: robust_stats([parse_float(row.get(f"{feature}_abs_delta")) for row in deltas])
        for feature in features
    }

    scored: list[dict[str, object]] = []
    for row in deltas:
        score = 0.0
        triggered: list[str] = []
        for feature in features:
            value = parse_float(row.get(f"{feature}_abs_delta"))
            center, scale = stats[feature]
            if not math.isfinite(value) or not math.isfinite(center) or not math.isfinite(scale) or scale == 0:
                continue
            z = max((value - center) / scale, 0.0)
            row[f"{feature}_robust_z"] = round(z, 6)
            if z > 0:
                score += z
            if z >= 6.0:
                triggered.append(feature)
        row["event_score"] = round(score, 6)
        row["triggered_features"] = ";".join(triggered)
        scored.append(row)
    return scored


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return
    with output.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect candidate NILM events from minute features.")
    parser.add_argument("--features", type=Path, default=Path("outputs/features/minute_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--top-n", type=int, default=300, help="Number of highest-scoring candidates to write")
    parser.add_argument("--min-score", type=float, default=20.0, help="Minimum aggregate robust score")
    parser.add_argument("--feature", action="append", dest="features_to_use", help="Feature to compare; can be repeated")
    args = parser.parse_args()

    features = tuple(args.features_to_use) if args.features_to_use else DEFAULT_FEATURES
    rows = load_rows(args.features)
    deltas = compute_deltas(grouped_rows(rows), features)
    scored = score_events(deltas, features)
    candidates = [
        row
        for row in scored
        if parse_float(row.get("event_score")) >= args.min_score and row.get("triggered_features")
    ]
    candidates.sort(key=lambda row: parse_float(row.get("event_score")), reverse=True)
    if args.top_n is not None:
        candidates = candidates[: args.top_n]

    write_csv(candidates, args.output)
    print(f"read {len(rows)} feature rows")
    print(f"scored {len(scored)} adjacent-minute changes")
    print(f"wrote {len(candidates)} candidate events to {args.output}")


if __name__ == "__main__":
    main()
