"""Plot before/after waveform samples for candidate NILM events."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parse_waveform import DEFAULT_FORMAT, inspect_waveform_file, parse_waveform_file


CANVAS_WIDTH = 1800
ROW_HEIGHT = 210
LEFT_MARGIN = 90
RIGHT_MARGIN = 40
TOP_MARGIN = 115
BOTTOM_MARGIN = 60
PANEL_GAP = 40
BACKGROUND = (250, 250, 248)
GRID = (220, 224, 228)
AXIS = (94, 103, 113)
TEXT = (30, 35, 42)
MUTED = (95, 103, 112)
BEFORE = (31, 119, 180)
AFTER = (214, 90, 49)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def safe_stem(value: str) -> str:
    return re.sub(r'[<>:\\"/|?*\s]+', "_", value).strip("_")


def scale_points(values: np.ndarray, x0: int, y0: int, width: int, height: int, vmin: float, vmax: float) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    xs = np.linspace(x0, x0 + width, values.size)
    ys = y0 + height - ((values.astype(np.float64) - vmin) / (vmax - vmin) * height)
    return [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]


def draw_grid(draw: ImageDraw.ImageDraw, x0: int, y0: int, width: int, height: int, records: int) -> None:
    draw.rectangle([x0, y0, x0 + width, y0 + height], outline=AXIS, width=1)
    for tick in range(records + 1):
        x = x0 + int(round(width * tick / max(records, 1)))
        draw.line([x, y0, x, y0 + height], fill=GRID, width=1)
    for tick in range(1, 4):
        y = y0 + int(round(height * tick / 4))
        draw.line([x0, y, x0 + width, y], fill=GRID, width=1)


def choose_middle_start(record_count: int, records: int) -> int:
    if record_count <= records:
        return 0
    return max((record_count - records) // 2, 0)


def read_event_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.DictReader(file_obj))
    return rows[:limit]


def read_waveform_window(path: Path, records: int) -> tuple[np.ndarray, int]:
    info = inspect_waveform_file(path)
    start = choose_middle_start(info.record_count, records)
    return parse_waveform_file(path, start_frame=start, max_frames=records), start


def plot_event(
    event: dict[str, str],
    index: int,
    raw_data_dir: Path,
    output_dir: Path,
    records: int,
) -> Path:
    before_path = raw_data_dir / event["previous_relative_path"]
    after_path = raw_data_dir / event["relative_path"]
    before_waveforms, before_start = read_waveform_window(before_path, records)
    after_waveforms, after_start = read_waveform_window(after_path, records)

    channels = len(DEFAULT_FORMAT.channel_names)
    panel_width = (CANVAS_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - PANEL_GAP) // 2
    plot_height = ROW_HEIGHT - 70
    canvas_height = TOP_MARGIN + channels * ROW_HEIGHT + BOTTOM_MARGIN
    image = Image.new("RGB", (CANVAS_WIDTH, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = load_font(26)
    label_font = load_font(17)
    small_font = load_font(14)

    title = f"Candidate event #{index:03d}  score={event.get('event_score', '')}"
    subtitle = f"{event.get('category', '')}  {event.get('timestamp', '')}  features={event.get('triggered_features', '')}"
    draw.text((LEFT_MARGIN, 24), title, fill=TEXT, font=title_font)
    draw.text((LEFT_MARGIN, 58), subtitle[:190], fill=MUTED, font=small_font)
    draw.text((LEFT_MARGIN, 88), f"before: {event['previous_relative_path']}", fill=BEFORE, font=small_font)
    draw.text((LEFT_MARGIN + panel_width + PANEL_GAP, 88), f"after: {event['relative_path']}", fill=AFTER, font=small_font)

    x_before = LEFT_MARGIN
    x_after = LEFT_MARGIN + panel_width + PANEL_GAP

    for channel, name in enumerate(DEFAULT_FORMAT.channel_names):
        row_y = TOP_MARGIN + channel * ROW_HEIGHT
        y0 = row_y + 38
        before_values = before_waveforms[:, channel, :].reshape(-1)
        after_values = after_waveforms[:, channel, :].reshape(-1)
        combined = np.concatenate([before_values, after_values])
        vmin = float(np.min(combined))
        vmax = float(np.max(combined))

        draw.text((18, y0 + plot_height // 2 - 10), name, fill=TEXT, font=label_font)
        label = f"{name}  before_rms={np.sqrt(np.mean(before_values.astype(np.float64) ** 2)):.1f}  after_rms={np.sqrt(np.mean(after_values.astype(np.float64) ** 2)):.1f}"
        draw.text((x_before, row_y + 8), label, fill=TEXT, font=label_font)

        draw_grid(draw, x_before, y0, panel_width, plot_height, records)
        draw_grid(draw, x_after, y0, panel_width, plot_height, records)
        before_points = scale_points(before_values, x_before, y0, panel_width, plot_height, vmin, vmax)
        after_points = scale_points(after_values, x_after, y0, panel_width, plot_height, vmin, vmax)
        if len(before_points) >= 2:
            draw.line(before_points, fill=BEFORE, width=2)
        if len(after_points) >= 2:
            draw.line(after_points, fill=AFTER, width=2)

        draw.text((x_before, y0 + plot_height + 8), f"records {before_start}-{before_start + records - 1}", fill=MUTED, font=small_font)
        draw.text((x_after, y0 + plot_height + 8), f"records {after_start}-{after_start + records - 1}", fill=MUTED, font=small_font)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(f"{index:03d}_{event.get('category', '')}_{event.get('relative_path', '')}")
    output_path = output_dir / f"{stem}.png"
    image.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot before/after waveforms for candidate events.")
    parser.add_argument("--events", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path(r"E:/华鹏波形数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/event_review"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--records", type=int, default=3)
    args = parser.parse_args()

    rows = read_event_rows(args.events, args.limit)
    outputs: list[Path] = []
    for index, row in enumerate(rows, start=1):
        output = plot_event(row, index, args.raw_data_dir, args.output_dir, args.records)
        outputs.append(output)
        print(f"[{index}/{len(rows)}] {output}")

    print(f"wrote {len(outputs)} review plots to {args.output_dir}")


if __name__ == "__main__":
    main()
