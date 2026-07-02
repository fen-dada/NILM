"""Extract minute-level features from Huapeng waveform files.

Each raw file is treated as one time window, usually about one minute. The output
CSV has one row per file and contains lightweight electrical features derived
from the confirmed 7-channel content layout:

Ua, Ub, Uc, Ia, Ib, Ic, I0

Values are still raw ADC/int24 units. Power-related features are therefore
relative features until voltage/current scaling factors are known.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from parse_waveform import DEFAULT_FORMAT, inspect_waveform_file, parse_waveform_file
from qc_scan import iter_waveform_files, parse_path_parts


VOLTAGE_CHANNELS = ("Ua", "Ub", "Uc")
CURRENT_CHANNELS = ("Ia", "Ib", "Ic")
ALL_CHANNELS = DEFAULT_FORMAT.channel_names
CHANNEL_INDEX = {name: index for index, name in enumerate(ALL_CHANNELS)}


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return math.nan
    return numerator / denominator


def rms(values: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=axis))


def crest_factor(values: np.ndarray) -> float:
    value_rms = float(rms(values))
    return safe_divide(float(np.max(np.abs(values))), value_rms)


def unbalance_ratio(phase_values: list[float]) -> float:
    avg = sum(phase_values) / len(phase_values)
    if avg == 0:
        return math.nan
    return max(abs(value - avg) for value in phase_values) / avg


def harmonic_features(samples_by_record: np.ndarray, max_harmonic: int = 15) -> dict[str, float]:
    """Return simple FFT ratios assuming each record spans one fundamental cycle."""

    if samples_by_record.size == 0:
        return {"thd_like": math.nan, "h3_ratio": math.nan, "h5_ratio": math.nan, "h7_ratio": math.nan}

    values = samples_by_record.astype(np.float64)
    values = values - np.mean(values, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(values, axis=1))
    if spectrum.shape[1] <= 1:
        return {"thd_like": math.nan, "h3_ratio": math.nan, "h5_ratio": math.nan, "h7_ratio": math.nan}

    fundamental = spectrum[:, 1]
    valid = fundamental > 0
    if not np.any(valid):
        return {"thd_like": math.nan, "h3_ratio": math.nan, "h5_ratio": math.nan, "h7_ratio": math.nan}

    limit = min(max_harmonic, spectrum.shape[1] - 1)
    harmonics = spectrum[:, 2 : limit + 1]
    thd = np.sqrt(np.sum(harmonics * harmonics, axis=1)) / fundamental

    result = {"thd_like": round(float(np.mean(thd[valid])), 8)}
    for harmonic in (3, 5, 7):
        if harmonic < spectrum.shape[1]:
            ratio = spectrum[:, harmonic] / fundamental
            result[f"h{harmonic}_ratio"] = round(float(np.mean(ratio[valid])), 8)
        else:
            result[f"h{harmonic}_ratio"] = math.nan
    return result


def choose_center_records(waveforms: np.ndarray, max_records: int | None) -> np.ndarray:
    if max_records is None or waveforms.shape[0] <= max_records:
        return waveforms
    start = max((waveforms.shape[0] - max_records) // 2, 0)
    return waveforms[start : start + max_records]


def channel_features(waveforms: np.ndarray, max_fft_records: int | None) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    if waveforms.size == 0:
        for channel in ALL_CHANNELS:
            for suffix in ("rms", "mean", "std", "min", "max", "p2p", "crest_factor", "thd_like", "h3_ratio", "h5_ratio", "h7_ratio"):
                features[f"{channel}_{suffix}"] = math.nan
        return features

    for channel in ALL_CHANNELS:
        idx = CHANNEL_INDEX[channel]
        values = waveforms[:, idx, :]
        flat = values.reshape(-1)
        features[f"{channel}_rms"] = round(float(rms(flat)), 8)
        features[f"{channel}_mean"] = round(float(np.mean(flat)), 8)
        features[f"{channel}_std"] = round(float(np.std(flat)), 8)
        features[f"{channel}_min"] = int(np.min(flat))
        features[f"{channel}_max"] = int(np.max(flat))
        features[f"{channel}_p2p"] = int(np.max(flat) - np.min(flat))
        features[f"{channel}_crest_factor"] = round(float(crest_factor(flat)), 8)
        h = harmonic_features(choose_center_records(waveforms, max_fft_records)[:, idx, :])
        for key, value in h.items():
            features[f"{channel}_{key}"] = value
    return features


def power_features(waveforms: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    if waveforms.size == 0:
        for name in ("Pa", "Pb", "Pc", "P_total", "Sa", "Sb", "Sc", "S_total", "power_factor_like"):
            features[name] = math.nan
        return features

    phase_pairs = (("a", "Ua", "Ia"), ("b", "Ub", "Ib"), ("c", "Uc", "Ic"))
    p_total = 0.0
    s_total = 0.0
    for suffix, voltage, current in phase_pairs:
        u = waveforms[:, CHANNEL_INDEX[voltage], :].astype(np.float64)
        i = waveforms[:, CHANNEL_INDEX[current], :].astype(np.float64)
        p = float(np.mean(u * i))
        s = float(rms(u) * rms(i))
        features[f"P{suffix}"] = round(p, 8)
        features[f"S{suffix}"] = round(s, 8)
        features[f"pf_{suffix}_like"] = round(float(safe_divide(p, s)), 8)
        p_total += p
        s_total += s

    features["P_total"] = round(p_total, 8)
    features["S_total"] = round(s_total, 8)
    features["power_factor_like"] = round(float(safe_divide(p_total, s_total)), 8)
    return features


def aggregate_features(waveforms: np.ndarray, max_fft_records: int | None) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    features.update(channel_features(waveforms, max_fft_records))
    features.update(power_features(waveforms))

    voltage_rms = [float(features[f"{channel}_rms"]) for channel in VOLTAGE_CHANNELS]
    current_rms = [float(features[f"{channel}_rms"]) for channel in CURRENT_CHANNELS]
    avg_voltage_rms = sum(voltage_rms) / len(voltage_rms)
    avg_current_rms = sum(current_rms) / len(current_rms)

    features["voltage_rms_avg"] = round(avg_voltage_rms, 8)
    features["current_rms_avg"] = round(avg_current_rms, 8)
    features["voltage_unbalance"] = round(float(unbalance_ratio(voltage_rms)), 8)
    features["current_unbalance"] = round(float(unbalance_ratio(current_rms)), 8)
    features["i0_to_phase_current_ratio"] = round(float(safe_divide(float(features["I0_rms"]), avg_current_rms)), 8)
    return features


def scan_file(path: Path, raw_data_dir: Path, max_records_per_file: int | None, max_fft_records: int | None) -> dict[str, object]:
    row: dict[str, object] = parse_path_parts(path, raw_data_dir)
    info = inspect_waveform_file(path)
    row.update(
        {
            "file_bytes": info.file_bytes,
            "record_count": info.record_count,
            "first_timestamp": info.first_timestamp.isoformat() if info.first_timestamp else "",
            "last_timestamp": info.last_timestamp.isoformat() if info.last_timestamp else "",
            "duration_seconds_at_50hz": round(info.duration_seconds_at_50hz, 8),
        }
    )

    waveforms = parse_waveform_file(path, max_frames=max_records_per_file)
    row["feature_record_count"] = int(waveforms.shape[0])
    row["fft_record_count"] = int(choose_center_records(waveforms, max_fft_records).shape[0])
    row.update(aggregate_features(waveforms, max_fft_records))
    return row


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
    parser = argparse.ArgumentParser(description="Extract minute-level waveform features.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path(r"E:/华鹏波形数据"))
    parser.add_argument("--output", type=Path, default=Path("outputs/features/minute_features.csv"))
    parser.add_argument("--max-records-per-file", type=int, default=None, help="Optional cap for faster exploratory runs")
    parser.add_argument("--max-fft-records", type=int, default=200, help="Record cap for FFT-based harmonic features")
    parser.add_argument("--limit", type=int, default=None, help="Extract only the first N files, for testing")
    args = parser.parse_args()

    files = list(iter_waveform_files(args.raw_data_dir))
    if args.limit is not None:
        files = files[: args.limit]

    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        rows.append(scan_file(path, args.raw_data_dir, args.max_records_per_file, args.max_fft_records))

    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

