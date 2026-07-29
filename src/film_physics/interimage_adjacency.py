"""Bounded cross-layer interimage adjacency in developed-density space."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class InterimageAdjacencyProfile:
    pixel_pitch_um: float
    diffusion_sigma_um: float
    coupling_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    maximum_absolute_transmittance_delta: float
    maximum_absolute_density_delta: float
    black_reference_density: tuple[float, float, float]
    white_reference_density: tuple[float, float, float]
    gaussian_truncate: float = 4.0

    def __post_init__(self) -> None:
        scalars = (
            self.pixel_pitch_um,
            self.diffusion_sigma_um,
            self.maximum_absolute_transmittance_delta,
            self.maximum_absolute_density_delta,
            self.gaussian_truncate,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in scalars):
            raise ValueError("interimage scalar parameters must be finite and positive")
        if self.maximum_absolute_transmittance_delta >= 1.0:
            raise ValueError("transmittance bound must be inside (0, 1)")
        matrix = np.asarray(self.coupling_matrix, dtype=np.float64)
        black = np.asarray(self.black_reference_density, dtype=np.float64)
        white = np.asarray(self.white_reference_density, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("coupling matrix must be finite 3x3")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("coupling matrix must be exactly symmetric")
        if np.any(np.diag(matrix) < 0.0):
            raise ValueError("coupling diagonal must be nonnegative")
        off_diagonal = matrix[~np.eye(3, dtype=bool)]
        if np.any(off_diagonal > 0.0):
            raise ValueError("coupling off-diagonal must be nonpositive")
        if not np.allclose(np.sum(matrix, axis=1), 0.0, atol=1e-15, rtol=0.0):
            raise ValueError("coupling rows must sum to zero")
        if (
            black.shape != (3,)
            or white.shape != (3,)
            or not np.all(np.isfinite(black))
            or not np.all(np.isfinite(white))
            or np.any(black < 0.0)
            or np.any(white <= black)
        ):
            raise ValueError("density references must be finite ordered RGB")


def interimage_adjacency_profile_from_contract(
    contract: dict,
) -> InterimageAdjacencyProfile:
    row = contract["candidate"]
    matrix = tuple(
        tuple(float(value) for value in values)
        for values in row["coupling_matrix"]
    )
    if len(matrix) != 3 or any(len(values) != 3 for values in matrix):
        raise ValueError("contract coupling matrix must be 3x3")
    return InterimageAdjacencyProfile(
        pixel_pitch_um=float(row["pixel_pitch_um"]),
        diffusion_sigma_um=float(row["diffusion_sigma_um"]),
        coupling_matrix=matrix,  # type: ignore[arg-type]
        maximum_absolute_transmittance_delta=float(
            row["maximum_absolute_transmittance_delta"]
        ),
        maximum_absolute_density_delta=float(
            row["maximum_absolute_density_delta"]
        ),
        black_reference_density=tuple(
            float(value) for value in row["black_reference_density"]
        ),
        white_reference_density=tuple(
            float(value) for value in row["white_reference_density"]
        ),
        gaussian_truncate=float(row["gaussian_truncate"]),
    )


def required_interimage_adjacency_halo(
    profile: InterimageAdjacencyProfile,
) -> int:
    sigma_pixels = profile.diffusion_sigma_um / profile.pixel_pitch_um
    return int(profile.gaussian_truncate * sigma_pixels + 0.5)


def _validate_density(
    developed_density: np.ndarray,
    profile: InterimageAdjacencyProfile,
) -> np.ndarray:
    density = np.asarray(developed_density, dtype=np.float64)
    black = np.asarray(profile.black_reference_density, dtype=np.float64)
    white = np.asarray(profile.white_reference_density, dtype=np.float64)
    if density.ndim != 3 or density.shape[-1] != 3:
        raise ValueError("interimage adjacency requires a HxWx3 density array")
    if (
        not np.all(np.isfinite(density))
        or np.any(density < black.reshape(1, 1, 3))
        or np.any(density > white.reshape(1, 1, 3))
    ):
        raise ValueError("developed density is outside the profile domain")
    return density


def apply_interimage_adjacency(
    developed_density: np.ndarray,
    profile: InterimageAdjacencyProfile,
) -> np.ndarray:
    """Apply a smooth cross-layer correction with density/transmittance bounds."""

    density = _validate_density(developed_density, profile)
    matrix = np.asarray(profile.coupling_matrix, dtype=np.float64)
    if not np.any(matrix):
        return density.copy()
    neutral_axis = np.array_equal(
        density[..., 0], density[..., 1]
    ) and np.array_equal(density[..., 1], density[..., 2])
    if np.all(density == density[0, 0]) or neutral_axis:
        return density.copy()

    sigma_pixels = profile.diffusion_sigma_um / profile.pixel_pitch_um
    blurred = np.empty_like(density)
    for channel in range(3):
        blurred[..., channel] = gaussian_filter(
            density[..., channel],
            sigma=sigma_pixels,
            order=0,
            mode="nearest",
            truncate=profile.gaussian_truncate,
        )
    raw = (density - blurred) @ matrix.T

    black = np.asarray(profile.black_reference_density, dtype=np.float64)
    white = np.asarray(profile.white_reference_density, dtype=np.float64)
    transmittance = np.power(10.0, -density)
    lower_transmittance = np.maximum(
        transmittance - profile.maximum_absolute_transmittance_delta,
        np.finfo(np.float64).tiny,
    )
    upper_transmittance = np.minimum(
        transmittance + profile.maximum_absolute_transmittance_delta,
        1.0,
    )
    positive_limit = np.minimum.reduce(
        (
            -np.log10(lower_transmittance) - density,
            np.full_like(density, profile.maximum_absolute_density_delta),
            white.reshape(1, 1, 3) - density,
        )
    )
    negative_limit = np.minimum.reduce(
        (
            density + np.log10(upper_transmittance),
            np.full_like(density, profile.maximum_absolute_density_delta),
            density - black.reshape(1, 1, 3),
        )
    )
    limit = np.where(raw >= 0.0, positive_limit, negative_limit)
    correction = np.zeros_like(raw)
    active = limit > 0.0
    correction[active] = (
        np.sign(raw[active])
        * limit[active]
        * np.tanh(np.abs(raw[active]) / limit[active])
    )
    output = density + correction
    if (
        not np.all(np.isfinite(output))
        or np.any(output < black.reshape(1, 1, 3) - 1e-12)
        or np.any(output > white.reshape(1, 1, 3) + 1e-12)
    ):
        raise RuntimeError("interimage adjacency escaped developed-density domain")
    output_transmittance = np.power(10.0, -output)
    if (
        np.max(np.abs(output_transmittance - transmittance))
        > profile.maximum_absolute_transmittance_delta + 1e-12
    ):
        raise RuntimeError("interimage adjacency violated transmittance bound")
    return output


def apply_interimage_adjacency_row_tiled(
    developed_density: np.ndarray,
    profile: InterimageAdjacencyProfile,
    *,
    tile_rows: int,
) -> np.ndarray:
    """Apply exact row partitions using the finite Gaussian support."""

    density = _validate_density(developed_density, profile)
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows <= 0:
        raise ValueError("tile_rows must be a positive integer")
    halo = required_interimage_adjacency_halo(profile)
    output = np.empty_like(density)
    for y0 in range(0, density.shape[0], tile_rows):
        y1 = min(density.shape[0], y0 + tile_rows)
        source_y0 = max(0, y0 - halo)
        source_y1 = min(density.shape[0], y1 + halo)
        rendered = apply_interimage_adjacency(
            density[source_y0:source_y1],
            profile,
        )
        output[y0:y1] = rendered[y0 - source_y0 : y1 - source_y0]
    return output


__all__ = [
    "InterimageAdjacencyProfile",
    "apply_interimage_adjacency",
    "apply_interimage_adjacency_row_tiled",
    "interimage_adjacency_profile_from_contract",
    "required_interimage_adjacency_halo",
]
