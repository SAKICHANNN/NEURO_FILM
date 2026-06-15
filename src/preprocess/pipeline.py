"""Shared preprocessing entrypoints."""

from __future__ import annotations

from pathlib import Path

from .raster_decode import inspect_raster, load_raster_working_image
from .raw_decode import inspect_raw, is_raw_path, load_raw_working_image
from .types import DecodeWarning, InputInspection, SourceProfile, WorkingImage


def inspect_input(path: Path | str) -> InputInspection:
    resolved = Path(path)
    if not resolved.exists():
        return InputInspection(
            path=resolved,
            exists=False,
            source_kind="unknown",
            source_profile=SourceProfile("unknown", "missing file"),
            warnings=[DecodeWarning("missing_file", "Input path does not exist.")],
        )
    if is_raw_path(resolved):
        return inspect_raw(resolved)
    return inspect_raster(resolved)


def load_working_image(path: Path | str) -> WorkingImage:
    resolved = Path(path)
    if is_raw_path(resolved):
        return load_raw_working_image(resolved)
    return load_raster_working_image(resolved)
