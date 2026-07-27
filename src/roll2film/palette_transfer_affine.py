"""Analytic collapse of affine-barycentric palette transfer.

The automatic/global path in Zhao et al. (JCST 2025) keeps affine
generalized barycentric weights fixed while replacing every source-palette
vertex by a transport-weighted target-palette vertex. Once the palettes and
transport are fixed, that construction is exactly one affine RGB operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TOLERANCE = 1e-10


def _validate_palette(name: str, palette: np.ndarray) -> np.ndarray:
    values = np.asarray(palette, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"{name} must be finite [0,1] rows of RGB")
    return values


def _validate_system(
    source_palette: np.ndarray,
    target_palette: np.ndarray,
    transport: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _validate_palette("source_palette", source_palette)
    target = _validate_palette("target_palette", target_palette)
    if len(source) < 4:
        raise ValueError("source palette needs at least four affine-spanning colours")
    centered = source - np.mean(source, axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-12) != 3:
        raise ValueError("source palette must affinely span RGB")
    mapping = np.asarray(transport, dtype=np.float64)
    if (
        mapping.shape != (len(source), len(target))
        or not np.all(np.isfinite(mapping))
        or np.any(mapping < 0.0)
        or np.any(mapping > 1.0)
        or not np.allclose(
            np.sum(mapping, axis=1), 1.0, atol=_TOLERANCE, rtol=0.0
        )
        or not np.allclose(
            np.sum(mapping, axis=0),
            len(source) / len(target),
            atol=_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise ValueError(
            "transport must satisfy the paper's nonnegative row/column sums"
        )
    return source, target, mapping


def _validate_colours(colours: np.ndarray) -> np.ndarray:
    values = np.asarray(colours, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("colours must be finite RGB rows")
    return values


@dataclass(frozen=True)
class AffinePaletteOperator:
    """Row-vector affine RGB operator ``output = input @ matrix + bias``."""

    matrix: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or bias.shape != (3,)
            or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(bias))
        ):
            raise ValueError("affine palette operator is invalid")
        matrix = matrix.copy()
        bias = bias.copy()
        matrix.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)

    def apply(self, colours: np.ndarray) -> np.ndarray:
        values = _validate_colours(colours)
        return values @ self.matrix + self.bias


def generalized_affine_barycentric_weights(
    source_palette: np.ndarray, colours: np.ndarray
) -> np.ndarray:
    """Evaluate the paper's affine generalized barycentric coordinates."""

    source = _validate_palette("source_palette", source_palette)
    if len(source) < 4:
        raise ValueError("source palette needs at least four affine-spanning colours")
    values = _validate_colours(colours)
    centre = np.mean(source, axis=0)
    centered = source - centre
    if np.linalg.matrix_rank(centered, tol=1e-12) != 3:
        raise ValueError("source palette must affinely span RGB")
    inverse_gram = np.linalg.inv(centered.T @ centered)
    return (
        (values - centre) @ inverse_gram @ centered.T
        + 1.0 / len(source)
    )


def collapse_palette_transfer_affine(
    source_palette: np.ndarray,
    target_palette: np.ndarray,
    transport: np.ndarray,
) -> AffinePaletteOperator:
    """Return the exact affine operator for fixed palettes and transport."""

    source, target, mapping = _validate_system(
        source_palette, target_palette, transport
    )
    centre = np.mean(source, axis=0)
    centered = source - centre
    mapped_palette = mapping @ target
    inverse_gram = np.linalg.inv(centered.T @ centered)
    matrix = inverse_gram @ centered.T @ mapped_palette
    bias = np.mean(mapped_palette, axis=0) - centre @ matrix
    return AffinePaletteOperator(matrix=matrix, bias=bias)


def apply_palette_transfer_direct(
    source_palette: np.ndarray,
    target_palette: np.ndarray,
    transport: np.ndarray,
    colours: np.ndarray,
) -> np.ndarray:
    """Evaluate the published barycentric construction without collapsing it."""

    source, target, mapping = _validate_system(
        source_palette, target_palette, transport
    )
    weights = generalized_affine_barycentric_weights(source, colours)
    return weights @ (mapping @ target)
