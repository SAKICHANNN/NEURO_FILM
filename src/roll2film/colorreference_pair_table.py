"""Deterministic ColorReference recorder/measurement table primitives.

This module parses already-audited local IT8/CGATS payloads and computes
repeated-target-set diagnostics. It deliberately contains no operator fitter.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import re
import zipfile

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ReferenceTable:
    member_name: str
    fields: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def parse_reference_table(
    archive_payload: bytes, member_name: str
) -> ReferenceTable:
    """Parse one declared CGATS table, including wrapped spectral rows."""

    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        payload = archive.read(member_name)
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        text = payload.decode("latin-1", errors="strict")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    upper = [line.upper() for line in lines]
    required = (
        "BEGIN_DATA_FORMAT",
        "END_DATA_FORMAT",
        "BEGIN_DATA",
        "END_DATA",
    )
    if any(token not in upper for token in required):
        raise ValueError("reference table lacks required data blocks")
    field_count = _declared_integer(lines, "NUMBER_OF_FIELDS")
    row_count = _declared_integer(lines, "NUMBER_OF_SETS")
    format_begin = upper.index("BEGIN_DATA_FORMAT")
    format_end = upper.index("END_DATA_FORMAT")
    fields = tuple(
        " ".join(lines[format_begin + 1 : format_end]).split()
    )
    if len(fields) != field_count:
        raise ValueError("reference field count mismatch")
    data_begin = upper.index("BEGIN_DATA")
    data_end = upper.index("END_DATA")
    tokens = " ".join(lines[data_begin + 1 : data_end]).split()
    if len(tokens) != field_count * row_count:
        raise ValueError("reference token count mismatch")
    rows = tuple(
        tuple(tokens[offset : offset + field_count])
        for offset in range(0, len(tokens), field_count)
    )
    return ReferenceTable(member_name, fields, rows)


def _declared_integer(lines: list[str], key: str) -> int:
    pattern = re.compile(rf"^{re.escape(key)}\s+\"?(\d+)\"?$", re.I)
    for line in lines:
        match = pattern.match(line)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"reference table lacks {key}")


def sample_source_patches(
    tiff_payload: bytes,
    *,
    main_x: list[int],
    main_y: list[int],
    gray_x: list[int],
    gray_y: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return channelwise median RGB codes and local channel ranges."""

    with Image.open(io.BytesIO(tiff_payload)) as image:
        image.load()
        if image.mode != "RGB" or image.n_frames != 1:
            raise ValueError("source TIFF must be one RGB frame")
        pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3:
        raise ValueError("source TIFF sample contract mismatch")
    centers = [(x, y) for y in main_y for x in main_x]
    centers.extend((x, gray_y) for x in gray_x)
    medians: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    height, width, _ = pixels.shape
    for x, y in centers:
        if not (
            radius <= x < width - radius
            and radius <= y < height - radius
        ):
            raise ValueError("source patch center outside valid image")
        patch = pixels[
            y - radius : y + radius + 1,
            x - radius : x + radius + 1,
            :,
        ]
        medians.append(np.median(patch, axis=(0, 1)))
        ranges.append(
            patch.max(axis=(0, 1)).astype(np.int16)
            - patch.min(axis=(0, 1)).astype(np.int16)
        )
    return (
        np.asarray(medians, dtype=np.float64),
        np.asarray(ranges, dtype=np.int16),
    )


def repeated_set_metrics(
    labs: np.ndarray,
) -> dict[str, float | int]:
    """Measure six-set target stability without fitting an operator.

    ``labs`` has shape ``[set, slide, sample, Lab]``.
    """

    values = np.asarray(labs, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or values.shape[0] < 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("invalid repeated-set Lab tensor")
    consensus = np.median(values, axis=0)
    radii = np.linalg.norm(values - consensus[None, ...], axis=-1)
    leave_one = []
    for set_index in range(values.shape[0]):
        other = np.delete(values, set_index, axis=0)
        center = np.median(other, axis=0)
        leave_one.append(
            np.linalg.norm(values[set_index] - center, axis=-1)
        )
    leave_one_distances = np.stack(leave_one, axis=0)

    correlations: list[float] = []
    for set_index in range(values.shape[0]):
        for slide_index in range(values.shape[1]):
            actual = _condensed_distances(
                values[set_index, slide_index]
            )
            expected = _condensed_distances(consensus[slide_index])
            correlations.append(_spearman(actual, expected))

    flattened = values.reshape(values.shape[0], -1, 3)
    overall = flattened.reshape(-1, 3).mean(axis=0)
    set_centroids = flattened.mean(axis=1)
    numerator = float(
        np.mean(np.sum((set_centroids - overall) ** 2, axis=1))
    )
    denominator = float(
        np.mean(
            np.sum(
                (flattened.reshape(-1, 3) - overall) ** 2,
                axis=1,
            )
        )
    )
    variance_fraction = (
        numerator / denominator if denominator > 0.0 else float("inf")
    )
    return {
        "observation_count": int(radii.size),
        "median_patch_set_radius_deltae76": float(np.median(radii)),
        "p95_patch_set_radius_deltae76": float(
            np.percentile(radii, 95.0)
        ),
        "maximum_patch_set_radius_deltae76": float(radii.max()),
        "fraction_patch_set_radius_above_10": float(
            np.mean(radii > 10.0)
        ),
        "median_slide_distance_structure_spearman": float(
            np.median(correlations)
        ),
        "minimum_slide_distance_structure_spearman": float(
            np.min(correlations)
        ),
        "between_set_centroid_variance_fraction": variance_fraction,
        "leave_one_set_consensus_rmse_deltae76": float(
            np.sqrt(np.mean(leave_one_distances**2))
        ),
    }


def _condensed_distances(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("distance input must be Nx3")
    indices = np.triu_indices(matrix.shape[0], k=1)
    delta = matrix[indices[0]] - matrix[indices[1]]
    return np.linalg.norm(delta, axis=1)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Spearman inputs must be equal nontrivial vectors")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    correlation = np.corrcoef(left_rank, right_rank)[0, 1]
    if not np.isfinite(correlation):
        raise ValueError("undefined Spearman correlation")
    return float(correlation)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while (
            end < values.size
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks
