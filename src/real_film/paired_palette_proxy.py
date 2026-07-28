"""Exact triangular-swatch extraction for the AO4P palette display proxy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _sample_triangle(
    pixels: np.ndarray,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    fraction_x: float,
    fraction_y: float,
    half_width: int,
    half_height: int,
) -> tuple[np.ndarray, int]:
    x = round(x0 + fraction_x * (x1 - x0))
    y = round(y0 + fraction_y * (y1 - y0))
    block = pixels[
        y - half_height : y + half_height + 1,
        x - half_width : x + half_width + 1,
    ]
    expected = (2 * half_height + 1, 2 * half_width + 1, 3)
    if block.shape != expected:
        raise ValueError("palette sample block escapes the image")
    channel_range = int(
        np.max(np.ptp(block.astype(np.int16), axis=(0, 1)))
    )
    value = np.median(block, axis=(0, 1)).astype(np.uint8)
    return value, channel_range


def extract_palette_groups(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Extract and verify all frozen film/reference palette pairs."""

    asset = config["asset"]
    raw = path.read_bytes()
    if (
        len(raw) != int(asset["expected_bytes"])
        or hashlib.sha256(raw).hexdigest() != asset["expected_sha256"]
    ):
        raise ValueError("AO4P asset identity mismatch")
    with Image.open(path) as image:
        if (
            image.format != asset["expected_format"]
            or image.mode != asset["expected_mode"]
            or image.size
            != (int(asset["expected_width"]), int(asset["expected_height"]))
        ):
            raise ValueError("AO4P image format, mode or dimensions mismatch")
        pixels = np.asarray(image, dtype=np.uint8)

    sample = config["sample"]
    film_position = sample["film_triangle"]
    reference_position = sample["reference_triangle"]
    half_width = int(sample["centre_half_width"])
    half_height = int(sample["centre_half_height"])
    maximum_allowed_range = int(sample["maximum_centre_channel_range"])
    results: list[dict[str, Any]] = []
    all_checks: list[dict[str, Any]] = []

    for group in config["groups"]:
        xs = [int(value) for value in group["column_boundaries"]]
        ys = [int(value) for value in group["row_boundaries"]]
        counts = [int(value) for value in group["row_cell_counts"]]
        if len(ys) != len(counts) + 1:
            raise ValueError("AO4P row geometry mismatch")
        film_values: list[np.ndarray] = []
        reference_values: list[np.ndarray] = []
        maximum_range = 0
        for row, count in enumerate(counts):
            if count > len(xs) - 1:
                raise ValueError("AO4P column geometry mismatch")
            for column in range(count):
                geometry = {
                    "x0": xs[column],
                    "x1": xs[column + 1],
                    "y0": ys[row],
                    "y1": ys[row + 1],
                    "half_width": half_width,
                    "half_height": half_height,
                }
                reference, reference_range = _sample_triangle(
                    pixels,
                    fraction_x=float(reference_position["fraction_x"]),
                    fraction_y=float(reference_position["fraction_y"]),
                    **geometry,
                )
                film, film_range = _sample_triangle(
                    pixels,
                    fraction_x=float(film_position["fraction_x"]),
                    fraction_y=float(film_position["fraction_y"]),
                    **geometry,
                )
                reference_values.append(reference)
                film_values.append(film)
                maximum_range = max(
                    maximum_range, reference_range, film_range
                )
        film_array = np.asarray(film_values, dtype=np.uint8)
        reference_array = np.asarray(reference_values, dtype=np.uint8)
        paired_array = np.concatenate((film_array, reference_array), axis=1)
        hashes = {
            "film_u8_sha256": _sha256(film_array),
            "reference_u8_sha256": _sha256(reference_array),
            "paired_u8_sha256": _sha256(paired_array),
        }
        unique_film = int(np.unique(film_array, axis=0).shape[0])
        unique_reference = int(np.unique(reference_array, axis=0).shape[0])
        checks = [
            {
                "name": "pair_count",
                "passed": len(film_array) == int(group["expected_pair_count"]),
            },
            {
                "name": "sample_blocks_uniform",
                "passed": maximum_range <= maximum_allowed_range,
            },
            {
                "name": "unique_film_count",
                "passed": unique_film
                == int(group["expected_unique_film_rgb_count"]),
            },
            {
                "name": "unique_reference_count",
                "passed": unique_reference
                == int(group["expected_unique_reference_rgb_count"]),
            },
            {
                "name": "film_hash",
                "passed": hashes["film_u8_sha256"]
                == group["expected_film_u8_sha256"],
            },
            {
                "name": "reference_hash",
                "passed": hashes["reference_u8_sha256"]
                == group["expected_reference_u8_sha256"],
            },
            {
                "name": "paired_hash",
                "passed": hashes["paired_u8_sha256"]
                == group["expected_paired_u8_sha256"],
            },
        ]
        all_checks.extend(
            {
                "name": f"{group['group_id']}::{check['name']}",
                "passed": check["passed"],
            }
            for check in checks
        )
        results.append(
            {
                "group_id": group["group_id"],
                "film_stock_id": group["film_stock_id"],
                "pair_count": len(film_array),
                "unique_film_rgb_count": unique_film,
                "unique_reference_rgb_count": unique_reference,
                "maximum_sample_channel_range": maximum_range,
                **hashes,
                "film_rgb_u8": film_array.tolist(),
                "reference_rgb_u8": reference_array.tolist(),
                "checks": checks,
            }
        )

    automatic_pass = all(bool(check["passed"]) for check in all_checks)
    return {
        "asset_sha256": asset["expected_sha256"],
        "paired_channel_order": sample["paired_channel_order"],
        "groups": results,
        "checks": all_checks,
        "automatic_pass": automatic_pass,
    }
