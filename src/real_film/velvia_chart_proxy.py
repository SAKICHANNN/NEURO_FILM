"""Bounded acquisition and exact patch extraction for the AO0 chart proxy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_exact_asset(
    url: str,
    destination: Path,
    *,
    maximum_bytes: int,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download one bounded public asset without retaining response metadata."""

    if (
        not url.startswith("https://")
        or maximum_bytes < 1
        or timeout_seconds <= 0.0
    ):
        raise ValueError("invalid bounded asset download controls")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    total = 0
    digest = hashlib.sha256()
    try:
        with requests.get(
            url,
            stream=True,
            timeout=timeout_seconds,
            headers={"User-Agent": "neuro-film-research/1.0"},
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > maximum_bytes:
                raise ValueError("asset Content-Length exceeds the frozen maximum")
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=64 * 1024):
                    if not block:
                        continue
                    total += len(block)
                    if total > maximum_bytes:
                        raise ValueError("asset body exceeds the frozen maximum")
                    handle.write(block)
                    digest.update(block)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _grid_centres(
    horizontal_start: int,
    horizontal_end_inclusive: int,
    vertical_start: int,
    vertical_end_inclusive: int,
    *,
    rows: int,
    columns: int,
) -> list[tuple[int, int]]:
    width = horizontal_end_inclusive - horizontal_start + 1
    height = vertical_end_inclusive - vertical_start + 1
    if width < columns or height < rows:
        raise ValueError("invalid chart grid bounds")
    return [
        (
            round(horizontal_start + (column + 0.5) * width / columns),
            round(vertical_start + (row + 0.5) * height / rows),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def _extract_grid(
    pixels: np.ndarray,
    centres: list[tuple[int, int]],
    *,
    half_width: int,
    half_height: int,
) -> tuple[np.ndarray, int]:
    values: list[np.ndarray] = []
    maximum_range = 0
    for centre_x, centre_y in centres:
        block = pixels[
            centre_y - half_height : centre_y + half_height + 1,
            centre_x - half_width : centre_x + half_width + 1,
        ]
        if block.shape != (2 * half_height + 1, 2 * half_width + 1, 3):
            raise ValueError("chart sample block escapes the image")
        maximum_range = max(
            maximum_range,
            int(np.max(np.ptp(block.astype(np.int16), axis=(0, 1)))),
        )
        values.append(np.median(block, axis=(0, 1)).astype(np.uint8))
    return np.asarray(values, dtype=np.uint8), maximum_range


def extract_chart_proxy(
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    asset = config["asset"]
    extraction = config["extraction"]
    if path.stat().st_size != int(asset["expected_bytes"]):
        raise ValueError("AO0 asset byte count mismatch")
    file_sha256 = sha256_file(path)
    if file_sha256 != asset["expected_sha256"]:
        raise ValueError("AO0 asset SHA-256 mismatch")
    with Image.open(path) as image:
        if (
            image.format != asset["expected_format"]
            or image.mode != asset["expected_mode"]
            or image.size
            != (int(asset["expected_width"]), int(asset["expected_height"]))
        ):
            raise ValueError("AO0 image format, mode or dimensions mismatch")
        pixels = np.asarray(image, dtype=np.uint8)

    common = {
        "vertical_start": int(extraction["vertical_start"]),
        "vertical_end_inclusive": int(extraction["vertical_end_inclusive"]),
        "rows": int(extraction["rows"]),
        "columns": int(extraction["columns"]),
    }
    film_centres = _grid_centres(
        int(extraction["film_horizontal_start"]),
        int(extraction["film_horizontal_end_inclusive"]),
        **common,
    )
    reference_centres = _grid_centres(
        int(extraction["reference_horizontal_start"]),
        int(extraction["reference_horizontal_end_inclusive"]),
        **common,
    )
    sample = {
        "half_width": int(extraction["centre_half_width"]),
        "half_height": int(extraction["centre_half_height"]),
    }
    film, film_range = _extract_grid(pixels, film_centres, **sample)
    reference, reference_range = _extract_grid(pixels, reference_centres, **sample)
    # The frozen wire order is target film RGB followed by source/reference RGB.
    paired = np.concatenate((film, reference), axis=1)
    hashes = {
        "film_patch_u8_sha256": hashlib.sha256(film.tobytes()).hexdigest(),
        "reference_patch_u8_sha256": hashlib.sha256(
            reference.tobytes()
        ).hexdigest(),
        "paired_patch_u8_sha256": hashlib.sha256(paired.tobytes()).hexdigest(),
    }
    checks = [
        {
            "name": "film_patch_hash",
            "passed": hashes["film_patch_u8_sha256"]
            == extraction["expected_film_patch_u8_sha256"],
        },
        {
            "name": "reference_patch_hash",
            "passed": hashes["reference_patch_u8_sha256"]
            == extraction["expected_reference_patch_u8_sha256"],
        },
        {
            "name": "paired_patch_hash",
            "passed": hashes["paired_patch_u8_sha256"]
            == extraction["expected_paired_patch_u8_sha256"],
        },
        {
            "name": "uniform_centres",
            "passed": max(film_range, reference_range)
            <= int(extraction["maximum_centre_channel_range"]),
        },
        {
            "name": "film_patches_unique",
            "passed": np.unique(film, axis=0).shape[0] == film.shape[0],
        },
        {
            "name": "reference_patches_unique",
            "passed": np.unique(reference, axis=0).shape[0] == reference.shape[0],
        },
    ]
    return {
        "asset_bytes": int(path.stat().st_size),
        "asset_sha256": file_sha256,
        "patch_count": int(film.shape[0]),
        "paired_patch_channel_order": ["film_rgb", "reference_rgb"],
        "film_patch_rgb_u8": film.tolist(),
        "reference_patch_rgb_u8": reference.tolist(),
        "film_centres_xy": [list(value) for value in film_centres],
        "reference_centres_xy": [list(value) for value in reference_centres],
        "film_maximum_centre_channel_range": film_range,
        "reference_maximum_centre_channel_range": reference_range,
        **hashes,
        "automatic_checks": checks,
        "automatic_pass": all(bool(check["passed"]) for check in checks),
    }


__all__ = [
    "download_exact_asset",
    "extract_chart_proxy",
    "sha256_file",
]
