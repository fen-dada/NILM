"""Create rule-based pseudo labels for candidate NILM events.

These labels are derived only from voltage/current features. They are useful as
bootstrap labels when no external operation log is available, but they are not
ground-truth equipment labels.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def parse_float(value: object) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator


def safe_image_name(index: int, category: str, relative_path: str) -> str:
    rel = re.sub(r'[<>:\\"/|?*\s]+', "_", relative_path).strip("_")
    cat = re.sub(r'[<>:\\"/|?*\s]+', "_", category).strip("_")
    return f"{index:03d}_{cat}_{rel}.png"


def classify_event(row: dict[str, str]) -> dict[str, object]:
    prev_current = parse_float(row.get("current_rms_avg_prev"))
    curr_current = parse_float(row.get("current_rms_avg_curr"))
    delta_current = parse_float(row.get("current_rms_avg_delta"))
    abs_delta = abs(delta_current) if math.isfinite(delta_current) else math.nan
    larger_current = max(prev_current, curr_current)
    smaller_current = min(prev_current, curr_current)
    relative_change = safe_divide(abs_delta, larger_current)
    on_off_ratio = safe_divide(larger_current, max(smaller_current, 1.0))

    label = "不确定"
    confidence = 0.5
    reason = "电流变化方向或幅度不足以稳定判断"

    if math.isfinite(delta_current) and delta_current > 0:
        if on_off_ratio >= 5.0 and relative_change >= 0.70:
            label = "启动"
            confidence = min(0.98, 0.75 + min(on_off_ratio, 20.0) / 100.0)
            reason = "平均三相电流由低到高，且前后电流差异很大"
        elif relative_change >= 0.20:
            label = "状态切换"
            confidence = min(0.90, 0.60 + relative_change / 2.0)
            reason = "平均三相电流明显上升，但事件前已经存在负荷"
    elif math.isfinite(delta_current) and delta_current < 0:
        if on_off_ratio >= 5.0 and relative_change >= 0.70:
            label = "停止"
            confidence = min(0.98, 0.75 + min(on_off_ratio, 20.0) / 100.0)
            reason = "平均三相电流由高到低，且后续电流接近事件前低负荷水平"
        elif relative_change >= 0.20:
            label = "状态切换"
            confidence = min(0.90, 0.60 + relative_change / 2.0)
            reason = "平均三相电流明显下降，但事件后仍存在负荷"

    if label == "不确定":
        confidence = 0.45

    direction = "上升" if delta_current > 0 else "下降" if delta_current < 0 else "无明显方向"
    return {
        "自动判断": label,
        "置信度": round(confidence, 4),
        "电流变化方向": direction,
        "电流相对变化": round(relative_change, 6) if math.isfinite(relative_change) else "",
        "前后电流倍数": round(on_off_ratio, 6) if math.isfinite(on_off_ratio) else "",
        "标注依据": reason,
    }


def load_rows(path: Path) -> list[dict[str, str]]:
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


def build_label_rows(events: list[dict[str, str]], project_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        label_info = classify_event(event)
        category = event.get("category", "")
        relative_path = event.get("relative_path", "")
        image_path = project_dir / "outputs" / "event_review" / safe_image_name(index, category, relative_path)
        rows.append(
            {
                "编号": index,
                "区域": category,
                "事件时间": event.get("timestamp", ""),
                **label_info,
                "可能设备": f"{category}负荷" if label_info["自动判断"] != "不确定" else "不确定",
                "备注": "自动伪标注：仅基于电压电流特征，未用真实启停记录验证",
                "事件分数": event.get("event_score", ""),
                "前平均电流": event.get("current_rms_avg_prev", ""),
                "后平均电流": event.get("current_rms_avg_curr", ""),
                "平均电流变化": event.get("current_rms_avg_delta", ""),
                "前总功率特征": event.get("P_total_prev", ""),
                "后总功率特征": event.get("P_total_curr", ""),
                "总功率特征变化": event.get("P_total_delta", ""),
                "触发特征": event.get("triggered_features", ""),
                "事件前文件": event.get("previous_relative_path", ""),
                "事件后文件": relative_path,
                "复核图片": str(image_path).replace("\\", "/"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rule-based pseudo labels for candidate events.")
    parser.add_argument("--events", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/labels/auto_event_labels.csv"))
    parser.add_argument("--project-dir", type=Path, default=Path(r"E:/NILM_Project"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    events = load_rows(args.events)
    if args.limit is not None:
        events = events[: args.limit]

    rows = build_label_rows(events, args.project_dir)
    write_csv(rows, args.output)

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["自动判断"])
        counts[label] = counts.get(label, 0) + 1

    print(f"read {len(events)} candidate events")
    print(f"wrote {len(rows)} pseudo labels to {args.output}")
    print(f"label counts: {counts}")


if __name__ == "__main__":
    main()
