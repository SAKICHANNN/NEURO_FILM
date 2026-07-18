"""Exact two-pass percentile reduction for repeatable finite float32 streams."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


STREAMING_PERCENTILE_VERSION = "float32-radix-percentile-v1"
_SIGN_BIT = np.uint32(0x80000000)
_LOW_MASK = np.uint32(0x0000FFFF)
_BIN_COUNT = 1 << 16

Float32ChunkFactory = Callable[[], Iterable[np.ndarray]]


@dataclass(frozen=True)
class StreamingPercentileResult:
    version: str
    count: int
    percentiles: tuple[float, ...]
    values: tuple[float, ...]
    rank_pairs: tuple[tuple[int, int], ...]
    stream_sha256: str
    histogram_bytes: int
    passes: int = 2


def _percentiles(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("percentiles must be numeric")
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("percentiles must be a non-empty numeric iterable") from error
    if not result or any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in result):
        raise ValueError("percentiles must be finite values in [0, 100]")
    return result


def _ordered_keys(flat: np.ndarray) -> np.ndarray:
    # NumPy canonicalizes signed zero in percentile results. Do the same before
    # converting IEEE-754 bits into monotonically ordered unsigned keys.
    canonical = flat.copy()
    canonical[canonical == 0.0] = np.float32(0.0)
    bits = canonical.view(np.uint32)
    negative = (bits & _SIGN_BIT) != 0
    return np.where(negative, ~bits, bits ^ _SIGN_BIT).astype(np.uint32, copy=False)


def _key_to_float(key: int) -> float:
    ordered = np.uint32(key)
    if ordered & _SIGN_BIT:
        bits = ordered ^ _SIGN_BIT
    else:
        bits = ~ordered
    return float(np.asarray(bits, dtype=np.uint32).view(np.float32))


def _chunks(factory: Float32ChunkFactory) -> Iterable[np.ndarray]:
    try:
        produced = factory()
        iterator = iter(produced)
    except Exception as error:
        raise ValueError("chunk factory must return an iterable") from error
    for chunk in iterator:
        if not isinstance(chunk, np.ndarray) or chunk.dtype != np.float32:
            raise ValueError("every chunk must be a float32 numpy array")
        if not np.isfinite(chunk).all():
            raise ValueError("chunks must contain only finite values")
        yield np.ascontiguousarray(chunk).reshape(-1)


def _rank_definition(count: int, percentiles: tuple[float, ...]) -> tuple[tuple[int, int, float], ...]:
    definitions = []
    for percentile in percentiles:
        # NumPy normalizes q to [0, 1] before computing the virtual index.
        # Preserve that operation order for bit-exact interpolation weights.
        position = (count - 1) * (percentile / 100.0)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        definitions.append((lower, upper, float(position - lower)))
    return tuple(definitions)


def _numpy_linear_interpolate(low: float, high: float, fraction: float) -> float:
    """Reproduce NumPy's float32-bound `_lerp` operation order."""

    left = np.asarray(low, dtype=np.float32)
    right = np.asarray(high, dtype=np.float32)
    weight = np.asarray(fraction, dtype=np.float64)
    difference = np.subtract(right, left)
    result = np.add(left, difference * weight)
    if bool(weight >= 0.5):
        result = np.subtract(right, difference * (1.0 - weight), dtype=result.dtype)
    return float(result)


def exact_streaming_percentiles(
    chunk_factory: Float32ChunkFactory,
    *,
    count: int,
    percentiles: Iterable[float],
) -> StreamingPercentileResult:
    """Recover NumPy-linear percentiles using two bounded radix passes."""

    if not callable(chunk_factory):
        raise ValueError("chunk_factory must be callable")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    requested = _percentiles(percentiles)
    ranks = _rank_definition(count, requested)

    high_counts = np.zeros(_BIN_COUNT, dtype=np.uint64)
    first_hash = hashlib.sha256()
    first_count = 0
    for flat in _chunks(chunk_factory):
        first_hash.update(flat.tobytes(order="C"))
        first_count += int(flat.size)
        if flat.size:
            high = (_ordered_keys(flat) >> np.uint32(16)).astype(np.int64, copy=False)
            high_counts += np.bincount(high, minlength=_BIN_COUNT).astype(np.uint64, copy=False)
    if first_count != count:
        raise ValueError(f"first pass produced {first_count} values, expected {count}")

    cumulative_high = np.cumsum(high_counts, dtype=np.uint64)
    selected_ranks = sorted({rank for lower, upper, _ in ranks for rank in (lower, upper)})
    rank_high: dict[int, tuple[int, int]] = {}
    selected_high: set[int] = set()
    for rank in selected_ranks:
        high_bucket = int(np.searchsorted(cumulative_high, np.uint64(rank), side="right"))
        before = int(cumulative_high[high_bucket - 1]) if high_bucket else 0
        rank_high[rank] = (high_bucket, rank - before)
        selected_high.add(high_bucket)

    low_counts = {bucket: np.zeros(_BIN_COUNT, dtype=np.uint64) for bucket in sorted(selected_high)}
    second_hash = hashlib.sha256()
    second_count = 0
    for flat in _chunks(chunk_factory):
        second_hash.update(flat.tobytes(order="C"))
        second_count += int(flat.size)
        if not flat.size:
            continue
        keys = _ordered_keys(flat)
        high = keys >> np.uint32(16)
        for bucket, histogram in low_counts.items():
            lows = (keys[high == bucket] & _LOW_MASK).astype(np.int64, copy=False)
            if lows.size:
                histogram += np.bincount(lows, minlength=_BIN_COUNT).astype(np.uint64, copy=False)
    if second_count != count:
        raise ValueError(f"second pass produced {second_count} values, expected {count}")
    if second_hash.digest() != first_hash.digest():
        raise ValueError("chunk factory changed its float32 byte stream between passes")

    rank_values: dict[int, float] = {}
    for rank, (high_bucket, rank_in_bucket) in rank_high.items():
        cumulative_low = np.cumsum(low_counts[high_bucket], dtype=np.uint64)
        low_bucket = int(np.searchsorted(cumulative_low, np.uint64(rank_in_bucket), side="right"))
        rank_values[rank] = _key_to_float((high_bucket << 16) | low_bucket)

    resolved = []
    for lower, upper, fraction in ranks:
        low_value = rank_values[lower]
        high_value = rank_values[upper]
        resolved.append(_numpy_linear_interpolate(low_value, high_value, fraction))
    histogram_bytes = int(high_counts.nbytes + sum(histogram.nbytes for histogram in low_counts.values()))
    return StreamingPercentileResult(
        version=STREAMING_PERCENTILE_VERSION,
        count=count,
        percentiles=requested,
        values=tuple(float(value) for value in resolved),
        rank_pairs=tuple((lower, upper) for lower, upper, _ in ranks),
        stream_sha256=first_hash.hexdigest(),
        histogram_bytes=histogram_bytes,
    )
