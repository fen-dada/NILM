"""Parse Huapeng NILM binary waveform records.

Confirmed record layer from vendor support:

- Each file is a sequence of records with no global file header.
- Each record has an 18-byte header followed by `data_len` bytes of content.
- Header layout, little-endian unless noted:
  - offset 0:  seq, uint32 little-endian
  - offset 4:  data_len, uint32 little-endian
  - offset 8:  timestamp, 10 bytes
- Timestamp layout:
  - byte 0: year offset from 2000
  - byte 1: month
  - byte 2: day
  - byte 3: hour
  - byte 4: minute
  - byte 5: second
  - byte 6-9: microsecond, uint32 big-endian

Confirmed content layout from vendor support:

- data_len is normally 5376 bytes.
- Each record contains 256 sample groups.
- Each group contains 7 data items:
  Ua, Ub, Uc, Ia, Ib, Ic, I0.
- Each item is 3 bytes, interpreted here as signed little-endian int24 until
  vendor confirms otherwise.

The returned waveform array has shape `(records, channels, samples_per_record)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable
import struct

import numpy as np


DEFAULT_CHANNEL_NAMES = ("Ua", "Ub", "Uc", "Ia", "Ib", "Ic", "I0")


@dataclass(frozen=True)
class WaveformFormat:
    """Content interpretation for one waveform record."""

    header_bytes: int = 18
    samples_per_record: int = 256
    channels: int = 7
    bytes_per_item: int = 3
    signed: bool = True
    item_byteorder: str = "little"
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES

    @property
    def expected_content_bytes(self) -> int:
        return self.samples_per_record * self.channels * self.bytes_per_item

    @property
    def nominal_record_bytes(self) -> int:
        return self.header_bytes + self.expected_content_bytes

    # Backward-compatible aliases used by early scripts.
    @property
    def samples_per_cycle(self) -> int:
        return self.samples_per_record

    @property
    def frame_bytes(self) -> int:
        return self.nominal_record_bytes

    @property
    def payload_bytes(self) -> int:
        return self.expected_content_bytes

    @property
    def dtype(self) -> str:
        return "int24"

    def validate(self) -> None:
        if self.header_bytes != 18:
            raise ValueError("vendor record header is expected to be 18 bytes")
        if self.samples_per_record <= 0:
            raise ValueError("samples_per_record must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.bytes_per_item != 3:
            raise ValueError("current content parser expects 3-byte items")
        if self.item_byteorder not in {"little", "big"}:
            raise ValueError("item_byteorder must be 'little' or 'big'")
        if len(self.channel_names) != self.channels:
            raise ValueError("channel_names length must match channels")


DEFAULT_FORMAT = WaveformFormat()


@dataclass(frozen=True)
class RecordTimestamp:
    """Timestamp decoded from the 10-byte vendor timestamp field."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int

    def to_datetime(self) -> datetime:
        return datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
        )

    def isoformat(self) -> str:
        return self.to_datetime().isoformat(timespec="microseconds")


@dataclass(frozen=True)
class RecordHeader:
    """Decoded 18-byte record header."""

    seq: int
    data_len: int
    timestamp: RecordTimestamp
    offset: int

    @property
    def content_offset(self) -> int:
        return self.offset + 18

    @property
    def next_offset(self) -> int:
        return self.content_offset + self.data_len


@dataclass(frozen=True)
class WaveformFileInfo:
    """Basic metadata inferred from a waveform file."""

    path: Path
    file_bytes: int
    record_count: int
    trailing_bytes: int
    data_len_counts: tuple[tuple[int, int], ...]
    first_timestamp: RecordTimestamp | None = None
    last_timestamp: RecordTimestamp | None = None
    format: WaveformFormat = DEFAULT_FORMAT

    @property
    def frame_count(self) -> int:
        return self.record_count

    @property
    def is_frame_aligned(self) -> bool:
        return self.trailing_bytes == 0

    @property
    def duration_seconds_at_50hz(self) -> float:
        return self.record_count / 50.0


def parse_timestamp(raw: bytes) -> RecordTimestamp:
    """Parse the 10-byte timestamp field from one record header."""

    if len(raw) != 10:
        raise ValueError(f"timestamp must be 10 bytes, got {len(raw)}")
    return RecordTimestamp(
        year=2000 + raw[0],
        month=raw[1],
        day=raw[2],
        hour=raw[3],
        minute=raw[4],
        second=raw[5],
        microsecond=int.from_bytes(raw[6:10], byteorder="big", signed=False),
    )


def parse_record_header(raw: bytes, *, offset: int = 0) -> RecordHeader:
    """Decode an 18-byte vendor record header."""

    if len(raw) != 18:
        raise ValueError(f"record header must be 18 bytes, got {len(raw)}")
    seq, data_len = struct.unpack_from("<II", raw, 0)
    timestamp = parse_timestamp(raw[8:18])
    return RecordHeader(seq=seq, data_len=data_len, timestamp=timestamp, offset=offset)


def decode_int24(raw_content: bytes, fmt: WaveformFormat = DEFAULT_FORMAT) -> np.ndarray:
    """Decode packed 3-byte integer content into int32 values.

    The vendor says each data item is 3 bytes. Until confirmed otherwise, this
    treats those 3-byte items as signed little-endian 24-bit integers.
    """

    fmt.validate()
    raw = np.frombuffer(raw_content, dtype=np.uint8)
    if raw.size % fmt.bytes_per_item:
        raise ValueError(f"content byte count {raw.size} is not divisible by 3")

    triples = raw.reshape(-1, 3).astype(np.int32)
    if fmt.item_byteorder == "little":
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    else:
        values = triples[:, 2] | (triples[:, 1] << 8) | (triples[:, 0] << 16)

    if fmt.signed:
        sign_bit = 1 << 23
        values = (values ^ sign_bit) - sign_bit
    return values.astype(np.int32, copy=False)


def _read_header(file_obj: BinaryIO, offset: int) -> RecordHeader | None:
    raw = file_obj.read(18)
    if not raw:
        return None
    if len(raw) != 18:
        raise EOFError(f"partial record header at byte offset {offset}: {len(raw)} bytes")
    return parse_record_header(raw, offset=offset)


def iter_record_headers(path: str | Path) -> Iterable[RecordHeader]:
    """Yield decoded record headers by following each header's data_len."""

    file_path = Path(path)
    with file_path.open("rb") as file_obj:
        offset = 0
        while True:
            header = _read_header(file_obj, offset)
            if header is None:
                break
            yield header
            file_obj.seek(header.data_len, 1)
            offset = header.next_offset


def inspect_waveform_file(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
) -> WaveformFileInfo:
    """Return record-count metadata without loading record contents."""

    fmt.validate()
    file_path = Path(path)
    file_bytes = file_path.stat().st_size
    counts: dict[int, int] = {}
    record_count = 0
    first_timestamp: RecordTimestamp | None = None
    last_timestamp: RecordTimestamp | None = None
    offset = 0

    with file_path.open("rb") as file_obj:
        while offset < file_bytes:
            remaining = file_bytes - offset
            if remaining < fmt.header_bytes:
                return WaveformFileInfo(
                    path=file_path,
                    file_bytes=file_bytes,
                    record_count=record_count,
                    trailing_bytes=remaining,
                    data_len_counts=tuple(sorted(counts.items())),
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp,
                    format=fmt,
                )

            header = _read_header(file_obj, offset)
            if header is None:
                break
            if header.data_len > file_bytes - header.content_offset:
                trailing = file_bytes - offset
                return WaveformFileInfo(
                    path=file_path,
                    file_bytes=file_bytes,
                    record_count=record_count,
                    trailing_bytes=trailing,
                    data_len_counts=tuple(sorted(counts.items())),
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp,
                    format=fmt,
                )

            record_count += 1
            counts[header.data_len] = counts.get(header.data_len, 0) + 1
            first_timestamp = first_timestamp or header.timestamp
            last_timestamp = header.timestamp
            file_obj.seek(header.data_len, 1)
            offset = header.next_offset

    return WaveformFileInfo(
        path=file_path,
        file_bytes=file_bytes,
        record_count=record_count,
        trailing_bytes=file_bytes - offset,
        data_len_counts=tuple(sorted(counts.items())),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        format=fmt,
    )


def _parse_contents_to_waveforms(contents: list[bytes], fmt: WaveformFormat) -> np.ndarray:
    if not contents:
        return np.empty((0, fmt.channels, fmt.samples_per_record), dtype=np.int32)
    raw_content = b"".join(contents)
    values = decode_int24(raw_content, fmt)
    waveforms = values.reshape(len(contents), fmt.samples_per_record, fmt.channels)
    return waveforms.transpose(0, 2, 1).copy()


def parse_waveform_bytes(
    raw: bytes | bytearray | memoryview,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Parse complete vendor records from bytes into `(records, channels, samples)`.

    The content is decoded as packed int24 and returned as int32.
    """

    fmt.validate()
    raw_view = memoryview(raw)
    contents: list[bytes] = []
    offset = 0
    total = len(raw_view)

    while offset < total:
        remaining = total - offset
        if remaining < fmt.header_bytes:
            if strict:
                raise ValueError(f"partial record header at byte offset {offset}")
            break
        header = parse_record_header(raw_view[offset : offset + fmt.header_bytes].tobytes(), offset=offset)
        content_start = offset + fmt.header_bytes
        content_end = content_start + header.data_len
        if content_end > total:
            if strict:
                raise ValueError(f"partial record content at byte offset {content_start}")
            break
        if header.data_len != fmt.expected_content_bytes:
            if strict:
                raise ValueError(
                    f"record at offset {offset} has data_len={header.data_len}, "
                    f"expected {fmt.expected_content_bytes}"
                )
            offset = content_end
            continue
        contents.append(raw_view[content_start:content_end].tobytes())
        offset = content_end

    return _parse_contents_to_waveforms(contents, fmt)


def parse_waveform_file(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    strict: bool = True,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    """Parse waveform content records into `(records, channels, samples)`.

    `start_frame` and `max_frames` refer to vendor records. Only records whose
    `data_len` matches the configured int24 waveform content size are converted.
    """

    fmt.validate()
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative or None")

    file_path = Path(path)
    contents: list[bytes] = []
    record_index = 0

    with file_path.open("rb") as file_obj:
        offset = 0
        while True:
            header = _read_header(file_obj, offset)
            if header is None:
                break
            content = file_obj.read(header.data_len)
            if len(content) != header.data_len:
                if strict:
                    raise EOFError(
                        f"record {record_index} expected {header.data_len} content bytes, "
                        f"got {len(content)}"
                    )
                break

            if record_index >= start_frame:
                if header.data_len == fmt.expected_content_bytes:
                    contents.append(content)
                    if max_frames is not None and len(contents) >= max_frames:
                        break
                elif strict:
                    raise ValueError(
                        f"record {record_index} has data_len={header.data_len}, "
                        f"expected {fmt.expected_content_bytes}"
                    )

            record_index += 1
            offset = header.next_offset

    return _parse_contents_to_waveforms(contents, fmt)


def read_frame_headers(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    """Read record headers as uint8 array shaped `(records, 18)`."""

    fmt.validate()
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative or None")

    headers: list[bytes] = []
    with Path(path).open("rb") as file_obj:
        offset = 0
        record_index = 0
        while True:
            raw = file_obj.read(fmt.header_bytes)
            if not raw:
                break
            if len(raw) != fmt.header_bytes:
                raise EOFError(f"partial record header at byte offset {offset}")
            header = parse_record_header(raw, offset=offset)
            if record_index >= start_frame:
                headers.append(raw)
                if max_frames is not None and len(headers) >= max_frames:
                    break
            file_obj.seek(header.data_len, 1)
            record_index += 1
            offset = header.next_offset

    if not headers:
        return np.empty((0, fmt.header_bytes), dtype=np.uint8)
    return np.frombuffer(b"".join(headers), dtype=np.uint8).reshape(len(headers), fmt.header_bytes)


def read_record_headers(
    path: str | Path,
    fmt: WaveformFormat = DEFAULT_FORMAT,
    *,
    max_records: int | None = None,
    start_record: int = 0,
) -> list[RecordHeader]:
    """Read decoded record headers."""

    fmt.validate()
    result: list[RecordHeader] = []
    for index, header in enumerate(iter_record_headers(path)):
        if index < start_record:
            continue
        result.append(header)
        if max_records is not None and len(result) >= max_records:
            break
    return result


def channel_summary(waveforms: np.ndarray) -> dict[str, list[float | int | str]]:
    """Compute simple per-channel statistics for parsed waveforms."""

    if waveforms.ndim != 3:
        raise ValueError("waveforms must have shape (records, channels, samples)")
    if waveforms.size == 0:
        return {"channel": [], "rms": [], "mean": [], "min": [], "max": []}

    values = waveforms.astype(np.float64)
    axes = (0, 2)
    return {
        "channel": list(DEFAULT_FORMAT.channel_names[: waveforms.shape[1]]),
        "rms": np.sqrt(np.mean(values * values, axis=axes)).round(3).tolist(),
        "mean": np.mean(values, axis=axes).round(3).tolist(),
        "min": np.min(waveforms, axis=axes).astype(int).tolist(),
        "max": np.max(waveforms, axis=axes).astype(int).tolist(),
    }


def _format_info(info: WaveformFileInfo) -> str:
    first = info.first_timestamp.isoformat() if info.first_timestamp else "None"
    last = info.last_timestamp.isoformat() if info.last_timestamp else "None"
    return "\n".join(
        [
            f"path: {info.path}",
            f"file_bytes: {info.file_bytes}",
            f"record_count: {info.record_count}",
            f"trailing_bytes: {info.trailing_bytes}",
            f"record_aligned: {info.is_frame_aligned}",
            f"data_len_counts: {dict(info.data_len_counts)}",
            f"first_timestamp: {first}",
            f"last_timestamp: {last}",
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
    parser.add_argument("--non-strict", action="store_true", help="Skip non-waveform records")
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
