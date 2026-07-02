"""Parse Huapeng NILM binary waveform files.

The current project data appears to be organized as repeated cycle frames:

- 5394 bytes per frame
- 18-byte frame header
- 5376-byte payload = 896 samples x 3 channels x int16

The parser returns waveform arrays in shape `(frames, channels, samples)` so that
one complete grid cycle can be addressed as `waveforms[frame_index, channel]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class WaveformFormat:
    """Binary frame layout for the waveform files."""

    frame_bytes: int = 5394
    header_bytes: int = 18
    samples_per_cycle: int = 896
    channels: int = 3
    dtype: str = "<i2"

    @property
    def payload_bytes(self) -> int:
        return self.frame_bytes - self.header_bytes

    @property
    def sample_bytes(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def expected_payload_bytes(self) -> int:
        return self.samples_per_cycle * self.channels * self.sample_bytes

    def validate(self) -> None:
        if self.header_bytes < 0:
            raise ValueError("header_bytes must be non-negative")
        if self.frame_bytes <= self.header_bytes:
            raise ValueError("frame_bytes must be larger than header_bytes")
        if self.payload_bytes != self.expected_payload_bytes:
            raise ValueError(
                "payload size mismatch: "
                f"frame payload is {self.payload_bytes} bytes, "
                f"but samples_per_cycle x channels x dtype is "
                f"{self.expected_payload_bytes} bytes"
            )


DEFAULT_FORMAT = WaveformFormat()


@dataclass(frozen=True)
class WaveformFileInfo:
    """Basic metadata inferred from a waveform file."""

    path: Path
    file_bytes: int
    frame_count: int
    trailing_bytes: int
    format: WaveformFormat = DEFAULT_FORMAT

    @property
    def is_frame_aligned(self) -> bool:
        return self.trailing_bytes == 0

    @property
    def duration_seconds_at_50hz(self) -> float:
        return self.frame_count / 50.0


def inspect_waveform_file(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
) -> WaveformFileInfo:
    """Return size and frame-count metadata without loading the full file."""

    fmt.validate()
    file_path = Path(path)
    file_bytes = file_path.stat().st_size
    frame_count, trailing_bytes = divmod(file_bytes, fmt.frame_bytes)
    return WaveformFileInfo(
        path=file_path,
        file_bytes=file_bytes,
        frame_count=frame_count,
        trailing_bytes=trailing_bytes,
        format=fmt,
    )


def _read_exact_frames(
    file_obj: BinaryIO,
    frame_count: int,
    fmt: WaveformFormat,
) -> bytes:
    byte_count = frame_count * fmt.frame_bytes
    raw = file_obj.read(byte_count)
    if len(raw) != byte_count:
        raise EOFError(f"expected {byte_count} bytes, got {len(raw)} bytes")
    return raw


def parse_waveform_bytes(
    raw: bytes | bytearray | memoryview,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Parse raw bytes into an array shaped `(frames, channels, samples)`.

    Args:
        raw: Binary data containing one or more complete frames.
        fmt: Frame layout definition.
        strict: If true, reject bytes whose length is not frame-aligned. If false,
            ignore trailing bytes that do not make a complete frame.
    """

    fmt.validate()
    raw_view = memoryview(raw)
    frame_count, trailing_bytes = divmod(len(raw_view), fmt.frame_bytes)
    if strict and trailing_bytes:
        raise ValueError(
            f"input has {len(raw_view)} bytes, leaving {trailing_bytes} trailing "
            f"bytes after {frame_count} complete frames"
        )
    if frame_count == 0:
        return np.empty((0, fmt.channels, fmt.samples_per_cycle), dtype=np.dtype(fmt.dtype))

    usable = raw_view[: frame_count * fmt.frame_bytes]
    frames = np.frombuffer(usable, dtype=np.uint8).reshape(frame_count, fmt.frame_bytes)
    payload = frames[:, fmt.header_bytes :]

    waveform = np.frombuffer(payload.tobytes(), dtype=np.dtype(fmt.dtype))
    waveform = waveform.reshape(frame_count, fmt.samples_per_cycle, fmt.channels)
    return waveform.transpose(0, 2, 1).copy()


def parse_waveform_file(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    strict: bool = True,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    """Parse a waveform file into `(frames, channels, samples)`.

    `max_frames` and `start_frame` allow quick sampling without loading the full
    minute file. This is useful for plotting and parser validation.
    """

    fmt.validate()
    file_path = Path(path)
    info = inspect_waveform_file(file_path, fmt)

    if strict and info.trailing_bytes:
        raise ValueError(
            f"{file_path} is not frame-aligned: {info.trailing_bytes} trailing bytes"
        )
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if start_frame >= info.frame_count:
        return np.empty((0, fmt.channels, fmt.samples_per_cycle), dtype=np.dtype(fmt.dtype))

    available_frames = info.frame_count - start_frame
    frame_count = available_frames if max_frames is None else min(max_frames, available_frames)
    if frame_count < 0:
        raise ValueError("max_frames must be non-negative or None")

    with file_path.open("rb") as file_obj:
        file_obj.seek(start_frame * fmt.frame_bytes)
        raw = _read_exact_frames(file_obj, frame_count, fmt)

    return parse_waveform_bytes(raw, fmt, strict=True)


def read_frame_headers(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    """Read frame headers as uint8 array shaped `(frames, header_bytes)`."""

    fmt.validate()
    file_path = Path(path)
    info = inspect_waveform_file(file_path, fmt)
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if start_frame >= info.frame_count:
        return np.empty((0, fmt.header_bytes), dtype=np.uint8)

    available_frames = info.frame_count - start_frame
    frame_count = available_frames if max_frames is None else min(max_frames, available_frames)

    headers = np.empty((frame_count, fmt.header_bytes), dtype=np.uint8)
    with file_path.open("rb") as file_obj:
        file_obj.seek(start_frame * fmt.frame_bytes)
        for index in range(frame_count):
            header = file_obj.read(fmt.header_bytes)
            if len(header) != fmt.header_bytes:
                raise EOFError(f"could not read header for frame {start_frame + index}")
            headers[index] = np.frombuffer(header, dtype=np.uint8)
            file_obj.seek(fmt.payload_bytes, 1)
    return headers


def channel_summary(waveforms: np.ndarray) -> dict[str, list[float | int]]:
    """Compute simple per-channel statistics for parsed waveforms."""

    if waveforms.ndim != 3:
        raise ValueError("waveforms must have shape (frames, channels, samples)")
    if waveforms.size == 0:
        return {"rms": [], "mean": [], "min": [], "max": []}

    values = waveforms.astype(np.float64)
    axes = (0, 2)
    return {
        "rms": np.sqrt(np.mean(values * values, axis=axes)).round(3).tolist(),
        "mean": np.mean(values, axis=axes).round(3).tolist(),
        "min": np.min(waveforms, axis=axes).astype(int).tolist(),
        "max": np.max(waveforms, axis=axes).astype(int).tolist(),
    }


def _format_info(info: WaveformFileInfo) -> str:
    return "\n".join(
        [
            f"path: {info.path}",
            f"file_bytes: {info.file_bytes}",
            f"frame_count: {info.frame_count}",
            f"trailing_bytes: {info.trailing_bytes}",
            f"frame_aligned: {info.is_frame_aligned}",
            f"duration_seconds_at_50hz: {info.duration_seconds_at_50hz:.3f}",
        ]
    )


def main() -> None:
    """Small command-line smoke test for one waveform file."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse one NILM waveform file.")
    parser.add_argument("path", type=Path, help="Path to one binary waveform file")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--non-strict", action="store_true", help="Ignore trailing bytes")
    args = parser.parse_args()

    info = inspect_waveform_file(args.path)
    waveforms = parse_waveform_file(
        args.path,
        strict=not args.non_strict,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    print(_format_info(info))
    print(f"parsed_shape: {tuple(waveforms.shape)}")
    print(json.dumps(channel_summary(waveforms), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
