"""Strict metadata-only parsing for the P242 DNG ProfileHueSatMap audit."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


class DngProfileHueSatMapAuditError(ValueError):
    """Raised when a retained ProfileHueSatMap payload is ambiguous or invalid."""


_TAG_LINE = re.compile(r"^Exif\.Image\.([A-Za-z0-9]+)\s+(.+?)\s*$")


@dataclass(frozen=True)
class ProfileHueSatMapAudit:
    """Validated metadata structure for one DNG profile hue/saturation map pair."""

    dimensions: tuple[int, int, int]
    encoding: int
    dynamic_range: int
    data1: np.ndarray
    data2: np.ndarray


def _single(tags: dict[str, list[str]], name: str, *, required: bool) -> str | None:
    values = tags.get(name, [])
    if not values:
        if required:
            raise DngProfileHueSatMapAuditError(f"missing {name}")
        return None
    unique = sorted(set(values))
    if len(unique) != 1:
        raise DngProfileHueSatMapAuditError(f"conflicting {name}")
    return unique[0]


def _integer_tokens(value: str, name: str, count: int) -> tuple[int, ...]:
    tokens = value.split()
    if len(tokens) != count or any(not re.fullmatch(r"\d+", token) for token in tokens):
        raise DngProfileHueSatMapAuditError(f"invalid {name}")
    return tuple(int(token) for token in tokens)


def _data(value: str, name: str, dimensions: tuple[int, int, int]) -> np.ndarray:
    try:
        values = np.asarray([float(token) for token in value.split()], dtype=np.float64)
    except ValueError as exc:
        raise DngProfileHueSatMapAuditError(f"invalid {name}") from exc
    expected = math.prod(dimensions) * 3
    if values.shape != (expected,):
        raise DngProfileHueSatMapAuditError(f"{name} count mismatch")
    if not np.all(np.isfinite(values)):
        raise DngProfileHueSatMapAuditError(f"{name} contains nonfinite values")
    return values.reshape(dimensions[2], dimensions[0], dimensions[1], 3)


def parse_profile_huesatmap_exif(text: str) -> ProfileHueSatMapAudit:
    """Parse and validate the exact DNG table tags used by P242."""

    if not isinstance(text, str) or not text.strip():
        raise DngProfileHueSatMapAuditError("EXIF text must be non-empty")
    tags: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = _TAG_LINE.fullmatch(line)
        if match is not None:
            tags.setdefault(match.group(1), []).append(match.group(2))

    dims_value = _single(tags, "ProfileHueSatMapDims", required=True)
    assert dims_value is not None
    dimensions = _integer_tokens(dims_value, "ProfileHueSatMapDims", 3)
    hue, saturation, value = dimensions
    if hue < 1 or saturation < 2 or value < 1:
        raise DngProfileHueSatMapAuditError("ProfileHueSatMapDims are out of range")

    data1_value = _single(tags, "ProfileHueSatMapData1", required=True)
    data2_value = _single(tags, "ProfileHueSatMapData2", required=True)
    assert data1_value is not None and data2_value is not None
    data1 = _data(data1_value, "ProfileHueSatMapData1", dimensions)
    data2 = _data(data2_value, "ProfileHueSatMapData2", dimensions)

    encoding_value = _single(tags, "ProfileHueSatMapEncoding", required=False)
    encoding = 0 if encoding_value is None else _integer_tokens(
        encoding_value, "ProfileHueSatMapEncoding", 1
    )[0]
    if encoding not in (0, 1):
        raise DngProfileHueSatMapAuditError("unsupported ProfileHueSatMapEncoding")
    dynamic_value = _single(tags, "ProfileDynamicRange", required=False)
    dynamic_range = 0 if dynamic_value is None else _integer_tokens(
        dynamic_value, "ProfileDynamicRange", 1
    )[0]
    if dynamic_range not in (0, 1):
        raise DngProfileHueSatMapAuditError("unsupported ProfileDynamicRange")

    for name, table in (("Data1", data1), ("Data2", data2)):
        zero_saturation_value_scale = table[:, :, 0, 2]
        if not np.array_equal(
            zero_saturation_value_scale,
            np.ones_like(zero_saturation_value_scale),
        ):
            raise DngProfileHueSatMapAuditError(
                f"{name} zero-saturation value scale must equal one"
            )
    return ProfileHueSatMapAudit(
        dimensions=dimensions,
        encoding=encoding,
        dynamic_range=dynamic_range,
        data1=data1,
        data2=data2,
    )


def table_summary(table: np.ndarray) -> dict[str, float | int]:
    """Return deterministic descriptive statistics without applying the table."""

    values = np.asarray(table, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise DngProfileHueSatMapAuditError("table structure is invalid")
    identity = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    changed = np.any(values != identity, axis=-1)
    return {
        "entry_count": int(np.prod(values.shape[:-1])),
        "hue_shift_min_degrees": float(np.min(values[..., 0])),
        "hue_shift_max_degrees": float(np.max(values[..., 0])),
        "saturation_scale_min": float(np.min(values[..., 1])),
        "saturation_scale_max": float(np.max(values[..., 1])),
        "value_scale_min": float(np.min(values[..., 2])),
        "value_scale_max": float(np.max(values[..., 2])),
        "nonidentity_fraction": float(np.mean(changed)),
    }


__all__ = [
    "DngProfileHueSatMapAuditError",
    "ProfileHueSatMapAudit",
    "parse_profile_huesatmap_exif",
    "table_summary",
]
