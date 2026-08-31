"""Predecode guards for the private product primary-image transaction."""

from __future__ import annotations

import os
from pathlib import Path


class ProductRenderTransactionError(ValueError):
    """Reject an unsafe product primary-image destination."""


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path.resolve(strict=False))))


def preflight_product_primary_output(input_path: Path, output_path: Path) -> None:
    """Reject an existing or input-alias destination without reading input pixels."""

    source = Path(input_path)
    destination = Path(output_path)
    if _normalized_path(source) == _normalized_path(destination):
        raise ProductRenderTransactionError(
            "product output must not identify the input path"
        )
    try:
        destination.lstat()
    except FileNotFoundError:
        return
    raise ProductRenderTransactionError(
        "product output destination must not already exist"
    )


__all__ = [
    "ProductRenderTransactionError",
    "preflight_product_primary_output",
]
