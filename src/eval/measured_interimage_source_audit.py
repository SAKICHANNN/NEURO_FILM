"""Bounded row-level data audit for measured photographic interimage sources."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


class MeasuredInterimageSourceError(ValueError):
    """Raised when a retained source drifts or cannot be audited exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(reader: PdfReader) -> str:
    raw = "\n".join(
        (page.extract_text() or "").replace("\r\n", "\n") for page in reader.pages
    )
    dehyphenated = re.sub(r"(?<=\w)-\n(?=\w)", "", raw)
    return "\n".join(" ".join(line.split()) for line in dehyphenated.splitlines())


def _candidate_six_numeric_rows(text: str) -> list[str]:
    """Find possible RGB-input plus three-output rows without interpreting figures."""

    rows: list[str] = []
    number = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if stripped and len(number.findall(stripped)) >= 6:
            rows.append(stripped)
    return rows


def audit_companion_pdf(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise MeasuredInterimageSourceError(f"missing companion PDF: {path}")
    if path.stat().st_size != expected_bytes:
        raise MeasuredInterimageSourceError("companion PDF byte count drift")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise MeasuredInterimageSourceError("companion PDF SHA-256 drift")

    reader = PdfReader(path)
    text = _normalized_text(reader)
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    required_anchors = {
        "film": "Kodak Ektachrome 100 Plus Professional film",
        "recorder": "A Solitaire 8xp image recorder was used",
        "measurement": "spectral transmittances of each of these samples were measured",
        "interimage": "inter-image effects",
        "operator": "matrix transformation on the dye concentrations",
        "row_sum": "constraining each row to a sum of 1.0",
        "reported_dataset": "data set consisting of cyan, magenta, and yellow",
    }
    search_text = " ".join(text.split())
    anchor_presence = {
        key: value in search_text for key, value in required_anchors.items()
    }
    if not all(anchor_presence.values()):
        missing = [key for key, present in anchor_presence.items() if not present]
        raise MeasuredInterimageSourceError(f"companion PDF anchor drift: {missing}")

    metadata = reader.metadata or {}
    title = str(metadata.get("/Title", ""))
    author = str(metadata.get("/Author", ""))
    if "Accurate Color Reproduction" not in title or "Hae Kyung Shin" not in author:
        raise MeasuredInterimageSourceError("companion PDF metadata drift")

    candidate_rows = _candidate_six_numeric_rows(text)
    return {
        "path": path.as_posix(),
        "bytes": expected_bytes,
        "sha256": digest,
        "page_count": len(reader.pages),
        "title": title,
        "author": author,
        "text_sha256": text_sha256,
        "anchor_presence": anchor_presence,
        "reported_measurement_design": {
            "film": "Kodak Ektachrome 100 Plus Professional",
            "film_recorder": "Solitaire 8xp CRT film recorder",
            "spectral_absorptivity_exposures": 60,
            "single_channel_ramp_steps": 11,
            "modeling_exposures_described": 21,
            "reported_rgb_to_cmy_dataset_colors": 36,
            "operator_family": "three 1-D LUTs followed by a row-sum-one 3x3 matrix",
        },
        "candidate_lines_with_at_least_six_numeric_values": candidate_rows,
        "complete_machine_readable_rgb_to_cmy_rows": 0,
        "raw_row_level_measurements_published": False,
    }
