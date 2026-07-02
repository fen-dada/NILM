"""Scan waveform files and write a QC report.

The report is one row per raw waveform file. It checks the confirmed vendor
record layer and computes lightweight channel statistics from a configurable
number of records per file.

By default, the script samples records instead of reading every record payload,
so it is suitable for a first project-wide pass. Use `--full-stats` when you
want channel statistics over every record in every file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from parse_waveform import DEFAULT_FORMAT, RecordHeader, inspect_waveform_file, parse_waveform_file, read_record_headers


RAW_CATEGORY_NAMES = {"波峰", "老化房", "选波", "配电房（总）", "隧道炉"}


def iter_waveform_files(raw_data_dir: Path) -> Iterable[Path]:
    """Yield raw waveform files below known category folders."""

    for category_dir in sorted(raw_data_dir.iterdir(), key=lambda p: p.name):
        if not category_dir.is_dir() or category_dir.name not in RAW_CATEGORY_NAMES:
            continue
        yield from sorted(path for path in category_dir.rglob("*") if path.is_file())


def parse_path_parts(path: Path, raw_data_dir: Path) -> dict[str, str]:
    rel = path.relative_to(raw_data_dir)
    parts = rel.parts
    return {
        "category": parts[0] if len(parts) > 0 else "",
        "date": parts[1] if len(parts) > 1 else "",
        "hour": parts[2] if len(parts) > 2 else "",
        "file_name": path.name,
        "relative_path": rel.as_posix(),
        "file_path": str(path),
    }


def choose_sample_start(record_count: int, max_records: int) -> tuple[int, int]:
    """Choose a middle-ish contiguous sample window."""

    if record_count <= 0 or max_records <= 0:
        return 0, 0
    count = min(record_count, max_records)
    start = max((record_count - count) // 2, 0)
    return start, count


def channel_stats(waveforms: np.ndarray) -> dict[str, float | int]:
    stats: dict[str, float | int] = {}
    names = DEFAULT_FORMAT.channel_names
    if waveforms.size == 0:
        for name in names:
            stats[f"{name}_rms"] = math.nan
            stats[f"{name}_mean"] = math.nan
            stats[f"{name}_min"] = math.nan
            stats[f"{name}_max"] = math.nan
            stats[f"{name}_p2p"] = math.nan
        return stats

    values = waveforms.astype(np.float64)
    axes = (0, 2)
    rms = np.sqrt(np.mean(values * values, axis=axes))
    mean = np.mean(values, axis=axes)
    mins = np.min(waveforms, axis=axes)
    maxs = np.max(waveforms, axis=axes)

    for idx, name in enumerate(names):
        stats[f"{name}_rms"] = round(float(rms[idx]), 6)
        stats[f"{name}_mean"] = round(float(mean[idx]), 6)
        stats[f"{name}_min"] = int(mins[idx])
        stats[f"{name}_max"] = int(maxs[idx])
        stats[f"{name}_p2p"] = int(maxs[idx] - mins[idx])
    return stats


def seq_status(headers: list[RecordHeader]) -> tuple[str, str]:
    if len(headers) < 2:
        return "unknown", "not enough headers"
    deltas = [(b.seq - a.seq) % 256 for a, b in zip(headers, headers[1:])]
    counts = Counter(deltas)
    if set(counts) == {1}:
        return "ok", "seq increments by 1 modulo 256"
    return "warn", f"seq modulo deltas {dict(counts.most_common(5))}"


def build_status(row: dict[str, object]) -> tuple[str, str]:
    notes: list[str] = []
    status = "ok"

    if row["trailing_bytes"] != 0:
        status = "error"
        notes.append("trailing bytes")
    if row["unexpected_data_len_count"] != 0:
        status = "error"
        notes.append("unexpected data_len")
    if row["record_count"] == 0:
        status = "error"
        notes.append("empty file")
    if row["record_count"] and abs(float(row["duration_seconds_at_50hz"]) - 60.0) > 2.0:
        if status == "ok":
            status = "warn"
        notes.append("duration differs from about 60s")
    if row["seq_status"] != "ok":
        if status == "ok":
            status = "warn"
        notes.append(str(row["seq_note"]))

    return status, "; ".join(notes)


def scan_file(path: Path, raw_data_dir: Path, *, max_records_per_file: int, full_stats: bool) -> dict[str, object]:
    row: dict[str, object] = parse_path_parts(path, raw_data_dir)
    info = inspect_waveform_file(path)
    data_len_counts = dict(info.data_len_counts)
    unexpected_count = sum(count for data_len, count in data_len_counts.items() if data_len != DEFAULT_FORMAT.expected_content_bytes)

    row.update(
        {
            "file_bytes": info.file_bytes,
            "record_count": info.record_count,
            "trailing_bytes": info.trailing_bytes,
            "record_aligned": info.is_frame_aligned,
            "data_len_counts": json.dumps(data_len_counts, ensure_ascii=False, sort_keys=True),
            "unexpected_data_len_count": unexpected_count,
            "first_timestamp": info.first_timestamp.isoformat() if info.first_timestamp else "",
            "last_timestamp": info.last_timestamp.isoformat() if info.last_timestamp else "",
            "duration_seconds_at_50hz": round(info.duration_seconds_at_50hz, 6),
        }
    )

    headers = read_record_headers(path, max_records=min(info.record_count, 200))
    seq_state, seq_note = seq_status(headers)
    row["seq_status"] = seq_state
    row["seq_note"] = seq_note

    if full_stats:
        sample_start, sample_count = 0, info.record_count
    else:
        sample_start, sample_count = choose_sample_start(info.record_count, max_records_per_file)
    row["stats_start_record"] = sample_start
    row["stats_record_count"] = sample_count

    if sample_count > 0 and unexpected_count == 0:
        waveforms = parse_waveform_file(path, start_frame=sample_start, max_frames=sample_count)
    else:
        waveforms = np.empty((0, DEFAULT_FORMAT.channels, DEFAULT_FORMAT.samples_per_record), dtype=np.int32)
    row.update(channel_stats(waveforms))

    status, notes = build_status(row)
    row["status"] = status
    row["notes"] = notes
    return row


def write_report(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan raw waveform files and write a QC CSV.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path(r"E:/华鹏波形数据"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qc/waveform_qc.csv"))
    parser.add_argument("--max-records-per-file", type=int, default=100)
    parser.add_argument("--full-stats", action="store_true", help="Compute channel stats over every record payload")
    parser.add_argument("--limit", type=int, default=None, help="Scan only the first N files, for testing")
    args = parser.parse_args()

    files = list(iter_waveform_files(args.raw_data_dir))
    if args.limit is not None:
        files = files[: args.limit]

    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        rows.append(
            scan_file(
                path,
                args.raw_data_dir,
                max_records_per_file=args.max_records_per_file,
                full_stats=args.full_stats,
            )
        )

    write_report(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

