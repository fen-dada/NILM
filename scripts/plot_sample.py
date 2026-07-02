"""Plot sample records from one parsed NILM waveform file.

This script intentionally uses Pillow instead of matplotlib so the first visual
validation step has very few dependencies. It reads a small number of records via
`parse_waveform.py` and writes a PNG with one row per channel.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parse_waveform import DEFAULT_FORMAT, channel_summary, inspect_waveform_file, parse_waveform_file


CANVAS_WIDTH = 1600
ROW_HEIGHT = 260
LEFT_MARGIN = 90
RIGHT_MARGIN = 35
TOP_MARGIN = 90
BOTTOM_MARGIN = 70
BACKGROUND = (250, 250, 248)
GRID = (218, 222, 226)
AXIS = (95, 103, 112)
TEXT = (30, 35, 42)
CHANNEL_COLORS = [(31, 119, 180), (214, 90, 49), (45, 158, 95)]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _safe_output_path(input_path: Path, output_dir: Path, start_frame: int, cycles: int) -> Path:
    category = input_path.parents[2].name if len(input_path.parents) >= 3 else "sample"
    date = input_path.parents[1].name if len(input_path.parents) >= 2 else "unknown_date"
    hour = input_path.parents[0].name if len(input_path.parents) >= 1 else "unknown_hour"
    stem = f"{category}_{date}_{hour}_{input_path.name}"
    stem = re.sub(r'[<>:\\"/|?*]+', "_", stem)
    return output_dir / f"{stem}_frame{start_frame}_records{cycles}.png"


def _scale_points(values: np.ndarray, x0: int, y0: int, width: int, height: int) -> list[tuple[int, int]]:
    if values.size == 0:
        return []

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0

    xs = np.linspace(x0, x0 + width, values.size)
    ys = y0 + height - ((values.astype(np.float64) - vmin) / (vmax - vmin) * height)
    return [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]


def _draw_grid(draw: ImageDraw.ImageDraw, x0: int, y0: int, width: int, height: int, cycles: int) -> None:
    draw.rectangle([x0, y0, x0 + width, y0 + height], outline=AXIS, width=1)
    for tick in range(cycles + 1):
        x = x0 + int(round(width * tick / max(cycles, 1)))
        draw.line([x, y0, x, y0 + height], fill=GRID, width=1)
    for tick in range(1, 4):
        y = y0 + int(round(height * tick / 4))
        draw.line([x0, y, x0 + width, y], fill=GRID, width=1)


def plot_waveforms(
    waveforms: np.ndarray,
    input_path: Path,
    output_path: Path,
    *,
    start_frame: int,
    dpi_note: str = "256 samples/record",
) -> Path:
    """Write a PNG showing parsed waveforms in three stacked channel rows."""

    if waveforms.ndim != 3:
        raise ValueError("waveforms must have shape (frames, channels, samples)")
    if waveforms.shape[0] == 0:
        raise ValueError("no frames available to plot")

    cycles, channels, samples = waveforms.shape
    plot_width = CANVAS_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    plot_height = ROW_HEIGHT - 70
    canvas_height = TOP_MARGIN + channels * ROW_HEIGHT + BOTTOM_MARGIN

    image = Image.new("RGB", (CANVAS_WIDTH, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _load_font(26)
    label_font = _load_font(18)
    small_font = _load_font(15)

    title = f"Waveform sample: {input_path.name}"
    subtitle = f"start_frame={start_frame}, records={cycles}, shape={tuple(waveforms.shape)}, {dpi_note}"
    draw.text((LEFT_MARGIN, 24), title, fill=TEXT, font=title_font)
    draw.text((LEFT_MARGIN, 58), subtitle, fill=(82, 90, 99), font=small_font)

    for channel in range(channels):
        row_y = TOP_MARGIN + channel * ROW_HEIGHT
        x0 = LEFT_MARGIN
        y0 = row_y + 36
        values = waveforms[:, channel, :].reshape(-1)
        color = CHANNEL_COLORS[channel % len(CHANNEL_COLORS)]

        _draw_grid(draw, x0, y0, plot_width, plot_height, cycles)
        points = _scale_points(values, x0, y0, plot_width, plot_height)
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)

        zero_y = None
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin <= 0 <= vmax and not math.isclose(vmin, vmax):
            zero_y = y0 + plot_height - int(round((0 - vmin) / (vmax - vmin) * plot_height))
            draw.line([x0, zero_y, x0 + plot_width, zero_y], fill=(140, 146, 153), width=1)

        rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
        label = f"{DEFAULT_FORMAT.channel_names[channel]}  min={int(vmin)}  max={int(vmax)}  rms={rms:.1f}"
        draw.text((x0, row_y + 8), label, fill=TEXT, font=label_font)
        draw.text((18, y0 + plot_height // 2 - 10), DEFAULT_FORMAT.channel_names[channel], fill=color, font=label_font)
        draw.text((x0, y0 + plot_height + 8), "record boundaries", fill=(104, 112, 122), font=small_font)
        if zero_y is not None:
            draw.text((x0 + plot_width - 45, zero_y - 18), "0", fill=(104, 112, 122), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a few parsed records from one waveform file.")
    parser.add_argument("path", type=Path, help="Path to one binary waveform file")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--records", "--cycles", dest="records", type=int, default=3, help="Number of records to plot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "sample_plots",
        help="Directory for the generated PNG",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional explicit output PNG path")
    args = parser.parse_args()

    if args.records <= 0:
        raise ValueError("--records must be positive")

    info = inspect_waveform_file(args.path)
    waveforms = parse_waveform_file(args.path, start_frame=args.start_frame, max_frames=args.records)
    output_path = args.output or _safe_output_path(args.path, args.output_dir, args.start_frame, args.records)
    output_path = plot_waveforms(waveforms, args.path, output_path, start_frame=args.start_frame)

    print(f"input: {args.path}")
    print(f"file_records: {info.frame_count}")
    print(f"trailing_bytes: {info.trailing_bytes}")
    print(f"parsed_shape: {tuple(waveforms.shape)}")
    print(f"summary: {channel_summary(waveforms)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()


