"""Shared SDR destination and encoding helpers for staging transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.preprocess import save_srgb8, save_srgb16_png, save_srgb16_tiff

from .contracts import ReferenceMatchContractError


SDR_OUTPUT_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
)
SDR16_OUTPUT_EXTENSIONS = frozenset({".png", ".tif", ".tiff"})


def _bounded_output_paths(
    values: Iterable[Path | str],
    *,
    count: int,
) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes, Path)):
        raise ReferenceMatchContractError(
            "output_paths must be an iterable of paths"
        )
    paths: list[Path] = []
    try:
        for value in values:
            if len(paths) >= count:
                raise ReferenceMatchContractError(
                    "output path count must match authorized sources"
                )
            paths.append(Path(value))
    except ReferenceMatchContractError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReferenceMatchContractError(
            "output_paths must be an iterable of paths"
        ) from exc
    if len(paths) != count:
        raise ReferenceMatchContractError(
            "output path count must match authorized sources"
        )
    return tuple(paths)


def staging_output_paths(
    values: Iterable[Path | str],
    *,
    count: int,
) -> tuple[Path, ...]:
    paths = _bounded_output_paths(values, count=count)
    keys = [str(path.resolve(strict=False)).casefold() for path in paths]
    if len(set(keys)) != len(keys):
        raise ReferenceMatchContractError("output paths must be unique")
    return paths


def validate_sdr_staging_destinations(
    outputs: tuple[Path, ...],
    report: Path,
    output_bit_depth: int,
    *,
    label: str,
) -> None:
    if output_bit_depth not in {8, 16}:
        raise ReferenceMatchContractError(
            "output_bit_depth must be 8 or 16"
        )
    allowed = (
        SDR_OUTPUT_EXTENSIONS
        if output_bit_depth == 8
        else SDR16_OUTPUT_EXTENSIONS
    )
    for path in outputs:
        if path.suffix.casefold() not in allowed:
            raise ReferenceMatchContractError(
                f"unsupported {label} output extension"
            )
        if path.exists() and path.is_dir():
            raise ReferenceMatchContractError(
                f"{label} output must not be a directory"
            )
    if report.suffix.casefold() != ".json":
        raise ReferenceMatchContractError(
            f"{label} report must use .json"
        )
    if report.exists() and report.is_dir():
        raise ReferenceMatchContractError(
            f"{label} report must not be a directory"
        )
    report_key = str(report.resolve(strict=False)).casefold()
    if report_key in {
        str(path.resolve(strict=False)).casefold() for path in outputs
    }:
        raise ReferenceMatchContractError(
            f"{label} report must not overwrite an output"
        )


def display_srgb(linear: np.ndarray, *, label: str) -> tuple[np.ndarray, float]:
    if (
        float(np.min(linear)) < -2e-6
        or float(np.max(linear)) > 1.0 + 2e-6
    ):
        raise ReferenceMatchContractError(
            f"{label} output exceeds the bounded sRGB encoding tolerance"
        )
    clipped = np.any((linear < 0.0) | (linear > 1.0), axis=-1)
    bounded = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        bounded <= 0.0031308,
        bounded * 12.92,
        1.055 * np.power(bounded, 1.0 / 2.4) - 0.055,
    )
    return (
        np.asarray(np.clip(encoded, 0.0, 1.0), dtype=np.float32),
        float(np.mean(clipped, dtype=np.float64)),
    )


def encode_sdr_staging_output(
    pixels: np.ndarray,
    destination: Path,
    *,
    output_bit_depth: int,
    label: str,
) -> tuple[str, float]:
    encoded, clipped_fraction = display_srgb(pixels, label=label)
    if output_bit_depth == 8:
        output_format = save_srgb8(encoded, destination)
    elif destination.suffix.casefold() == ".png":
        output_format = save_srgb16_png(encoded, destination)
    else:
        output_format = save_srgb16_tiff(encoded, destination)
    return output_format, clipped_fraction


__all__ = [
    "SDR16_OUTPUT_EXTENSIONS",
    "SDR_OUTPUT_EXTENSIONS",
    "display_srgb",
    "encode_sdr_staging_output",
    "staging_output_paths",
    "validate_sdr_staging_destinations",
]
