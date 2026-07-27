"""Strict parser and inventory helpers for the CAVE DoRF response archive.

DoRF contains normalized input/output response curves, not complete film colour
operators.  This module deliberately preserves source names and scale labels;
it does not infer process, scanner, spectral sensitivity, or stock identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np


class DorfArchiveError(ValueError):
    """Raised when the frozen archive or response format is invalid."""


@dataclass(frozen=True)
class DorfCurve:
    name: str
    scale: str
    irradiance: np.ndarray
    brightness: np.ndarray


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive(path: Path, expected_member: str = "dorfCurves.txt") -> tuple[bytes, dict[str, Any]]:
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != expected_member:
            raise DorfArchiveError("DoRF archive must contain exactly the frozen text member")
        member_path = PurePosixPath(members[0].filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise DorfArchiveError("unsafe DoRF archive member")
        payload = archive.read(members[0])
        metadata = {
            "member": members[0].filename,
            "member_bytes": members[0].file_size,
            "member_crc32": members[0].CRC,
            "member_sha256": sha256(payload).hexdigest(),
        }
    return payload, metadata


def parse_curves(payload: bytes, *, expected_samples: int = 1024) -> list[DorfCurve]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise DorfArchiveError("DoRF text is not ASCII") from exc
    if len(lines) % 6:
        raise DorfArchiveError("DoRF text is not a sequence of six-line records")

    curves: list[DorfCurve] = []
    for offset in range(0, len(lines), 6):
        name, scale, i_label, i_values, b_label, b_values = lines[offset : offset + 6]
        if not name:
            raise DorfArchiveError("empty curve name")
        if i_label.strip() != "I =" or b_label.strip() != "B =":
            raise DorfArchiveError(f"invalid DoRF labels for {name}")
        try:
            irradiance = np.fromstring(i_values, sep=" ", dtype=np.float64)
            brightness = np.fromstring(b_values, sep=" ", dtype=np.float64)
        except ValueError as exc:
            raise DorfArchiveError(f"invalid numbers for {name}") from exc
        if len(irradiance) != expected_samples or len(brightness) != expected_samples:
            raise DorfArchiveError(f"unexpected sample count for {name}")
        if not np.all(np.isfinite(irradiance)) or not np.all(np.isfinite(brightness)):
            raise DorfArchiveError(f"non-finite curve for {name}")
        if np.any(np.diff(irradiance) <= 0):
            raise DorfArchiveError(f"irradiance is not strictly increasing for {name}")
        if np.any(np.diff(brightness) < 0):
            raise DorfArchiveError(f"brightness is not monotone for {name}")
        if (
            irradiance[0] != 0.0
            or irradiance[-1] != 1.0
            or brightness[0] != 0.0
            or brightness[-1] != 1.0
            or np.min(brightness) < 0.0
            or np.max(brightness) > 1.0
        ):
            raise DorfArchiveError(f"curve endpoints or range invalid for {name}")
        curves.append(DorfCurve(name, scale, irradiance, brightness))
    return curves


def _channel_name(name: str) -> tuple[str, str] | None:
    for channel in ("Red", "Green", "Blue"):
        if name.endswith(channel):
            return name[: -len(channel)], channel.lower()
    return None


def strict_rgb_triplets(
    curves: list[DorfCurve],
    *,
    eligible_scales: set[str] | None = None,
) -> dict[str, dict[str, DorfCurve]]:
    groups: dict[str, dict[str, DorfCurve]] = defaultdict(dict)
    for curve in curves:
        parsed = _channel_name(curve.name)
        if parsed is None:
            continue
        base, channel = parsed
        if channel in groups[base]:
            raise DorfArchiveError(f"duplicate strict channel for {base}/{channel}")
        groups[base][channel] = curve
    result: dict[str, dict[str, DorfCurve]] = {}
    for base, channels in groups.items():
        if set(channels) != {"red", "green", "blue"}:
            continue
        scales = {curve.scale for curve in channels.values()}
        if len(scales) != 1:
            continue
        if eligible_scales is not None and next(iter(scales)) not in eligible_scales:
            continue
        result[base] = channels
    return dict(sorted(result.items()))


def inventory(curves: list[DorfCurve]) -> dict[str, Any]:
    groups: dict[str, dict[str, DorfCurve]] = defaultdict(dict)
    unmatched: list[str] = []
    for curve in curves:
        parsed = _channel_name(curve.name)
        if parsed is None:
            unmatched.append(curve.name)
            continue
        base, channel = parsed
        groups[base][channel] = curve
    complete = {
        base: channels
        for base, channels in groups.items()
        if set(channels) == {"red", "green", "blue"}
    }
    incomplete = {
        base: sorted(channels)
        for base, channels in groups.items()
        if set(channels) != {"red", "green", "blue"}
    }
    scale_mismatch = {
        base: sorted({curve.scale for curve in channels.values()})
        for base, channels in complete.items()
        if len({curve.scale for curve in channels.values()}) != 1
    }
    exact_curve_groups: dict[str, list[str]] = defaultdict(list)
    for curve in curves:
        key = sha256(curve.brightness.astype("<f8").tobytes()).hexdigest()
        exact_curve_groups[key].append(curve.name)
    duplicates = sorted(
        sorted(names) for names in exact_curve_groups.values() if len(names) > 1
    )
    duplicate_names = {
        name: count
        for name, count in sorted(Counter(curve.name for curve in curves).items())
        if count > 1
    }
    return {
        "record_count": len(curves),
        "samples_per_curve": sorted({len(curve.irradiance) for curve in curves}),
        "scale_counts": dict(sorted(Counter(curve.scale for curve in curves).items())),
        "complete_rgb_triplet_count": len(complete),
        "complete_rgb_triplets": sorted(complete),
        "incomplete_rgb_groups": dict(sorted(incomplete.items())),
        "unmatched_names": sorted(unmatched),
        "complete_triplet_scale_mismatches": scale_mismatch,
        "exact_duplicate_curve_groups": duplicates,
        "duplicate_source_names": duplicate_names,
    }


def audit_archive(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    if path.stat().st_size != source["zip_bytes"]:
        raise DorfArchiveError("DoRF zip size mismatch")
    if sha256_file(path) != source["zip_sha256"]:
        raise DorfArchiveError("DoRF zip hash mismatch")
    payload, archive = read_archive(path, source["member"])
    if archive["member_bytes"] != source["member_bytes"]:
        raise DorfArchiveError("DoRF member size mismatch")
    if archive["member_sha256"] != source["member_sha256"]:
        raise DorfArchiveError("DoRF member hash mismatch")
    if archive["member_crc32"] != source["member_crc32"]:
        raise DorfArchiveError("DoRF member CRC mismatch")
    curves = parse_curves(payload, expected_samples=config["expected"]["samples_per_curve"])
    result = inventory(curves)
    if result["record_count"] != config["expected"]["record_count"]:
        raise DorfArchiveError("DoRF record count mismatch")
    if result["scale_counts"] != config["expected"]["scale_counts"]:
        raise DorfArchiveError("DoRF scale inventory mismatch")
    return {"archive": archive, "inventory": result}
