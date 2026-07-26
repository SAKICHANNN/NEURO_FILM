"""Clean-room structural diagnostics for fixed published D-LUT assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


class PublishedDLUTError(ValueError):
    """Raised when a published D-LUT asset violates its frozen contract."""


@dataclass(frozen=True)
class CubeAsset:
    """A strict single 3D `.cube` LUT."""

    values: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        minimum = np.asarray(self.domain_min, dtype=np.float64)
        maximum = np.asarray(self.domain_max, dtype=np.float64)
        if (
            values.ndim != 4
            or values.shape[-1] != 3
            or len(set(values.shape[:3])) != 1
            or values.shape[0] < 2
        ):
            raise PublishedDLUTError("cube values must have shape (N,N,N,3)")
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise PublishedDLUTError("cube domain must contain three channels")
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
            or np.any(maximum <= minimum)
        ):
            raise PublishedDLUTError("cube values/domain must be finite and ordered")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "domain_min", minimum)
        object.__setattr__(self, "domain_max", maximum)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cube(path: Path) -> CubeAsset:
    """Parse standard red-fastest `.cube` rows into R-G-B array axes."""

    dimension: int | None = None
    domain_min = np.zeros(3, dtype=np.float64)
    domain_max = np.ones(3, dtype=np.float64)
    rows: list[list[float]] = []
    seen_data = False
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        key = fields[0].upper()
        if key == "TITLE":
            if seen_data:
                raise PublishedDLUTError(f"TITLE after data at line {line_number}")
            continue
        if key == "LUT_3D_SIZE":
            if len(fields) != 2 or dimension is not None:
                raise PublishedDLUTError("cube must declare one LUT_3D_SIZE")
            dimension = int(fields[1])
            continue
        if key in {"DOMAIN_MIN", "DOMAIN_MAX"}:
            if len(fields) != 4:
                raise PublishedDLUTError(f"invalid {key} declaration")
            value = np.asarray([float(item) for item in fields[1:]], dtype=np.float64)
            if key == "DOMAIN_MIN":
                domain_min = value
            else:
                domain_max = value
            continue
        if len(fields) != 3:
            raise PublishedDLUTError(f"invalid cube row at line {line_number}")
        try:
            rows.append([float(item) for item in fields])
        except ValueError as error:
            raise PublishedDLUTError(
                f"non-numeric cube row at line {line_number}"
            ) from error
        seen_data = True
    if dimension is None or dimension < 2:
        raise PublishedDLUTError("cube dimension is missing or invalid")
    if len(rows) != dimension**3:
        raise PublishedDLUTError(
            f"cube row count mismatch: {len(rows)} != {dimension**3}"
        )
    # `.cube` rows vary red fastest, then green, then blue. Reshape therefore
    # produces B-G-R axes and is transposed into the project's R-G-B order.
    values = np.asarray(rows, dtype=np.float64).reshape(
        dimension, dimension, dimension, 3
    )
    return CubeAsset(
        values=np.transpose(values, (2, 1, 0, 3)),
        domain_min=domain_min,
        domain_max=domain_max,
    )


def canonical_lut_manifest(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], str]:
    ordered = sorted(paths, key=_step_from_path)
    rows = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in ordered
    ]
    payload = "".join(
        f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return rows, hashlib.sha256(payload).hexdigest()


def _step_from_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("LUT_"):
        raise PublishedDLUTError(f"unexpected LUT path: {path.name}")
    try:
        return int(stem.removeprefix("LUT_"))
    except ValueError as error:
        raise PublishedDLUTError(f"invalid LUT step: {path.name}") from error


def identity_grid(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def tetrahedral_jacobians(asset: CubeAsset) -> np.ndarray:
    """Return exact affine Jacobians for all six tetrahedra in every cell."""

    values = asset.values
    c000 = values[:-1, :-1, :-1]
    c100 = values[1:, :-1, :-1]
    c010 = values[:-1, 1:, :-1]
    c001 = values[:-1, :-1, 1:]
    c110 = values[1:, 1:, :-1]
    c101 = values[1:, :-1, 1:]
    c011 = values[:-1, 1:, 1:]
    c111 = values[1:, 1:, 1:]
    steps = (asset.domain_max - asset.domain_min) / (values.shape[0] - 1)
    columns = (
        (c100 - c000, c110 - c100, c111 - c110),
        (c100 - c000, c111 - c101, c101 - c100),
        (c101 - c001, c111 - c101, c001 - c000),
        (c110 - c010, c010 - c000, c111 - c110),
        (c111 - c011, c010 - c000, c011 - c010),
        (c111 - c011, c011 - c001, c001 - c000),
    )
    return np.stack(
        [
            np.stack(
                (
                    red / steps[0],
                    green / steps[1],
                    blue / steps[2],
                ),
                axis=-1,
            )
            for red, green, blue in columns
        ],
        axis=-3,
    ).reshape(-1, 3, 3)


def trilinear_jacobians(
    asset: CubeAsset,
    subcell_axis: Iterable[float],
) -> np.ndarray:
    """Return analytic trilinear Jacobians on a fixed subcell grid."""

    values = asset.values
    c000 = values[:-1, :-1, :-1]
    c100 = values[1:, :-1, :-1]
    c010 = values[:-1, 1:, :-1]
    c001 = values[:-1, :-1, 1:]
    c110 = values[1:, 1:, :-1]
    c101 = values[1:, :-1, 1:]
    c011 = values[:-1, 1:, 1:]
    c111 = values[1:, 1:, 1:]
    steps = (asset.domain_max - asset.domain_min) / (values.shape[0] - 1)
    coordinates = np.asarray(list(subcell_axis), dtype=np.float64)
    if (
        coordinates.ndim != 1
        or len(coordinates) < 2
        or not np.all(np.isfinite(coordinates))
        or np.any(coordinates < 0.0)
        or np.any(coordinates > 1.0)
    ):
        raise PublishedDLUTError("subcell coordinates must lie in [0,1]")
    results = []
    for red in coordinates:
        for green in coordinates:
            for blue in coordinates:
                d_red = (
                    (1.0 - green) * (1.0 - blue) * (c100 - c000)
                    + green * (1.0 - blue) * (c110 - c010)
                    + (1.0 - green) * blue * (c101 - c001)
                    + green * blue * (c111 - c011)
                ) / steps[0]
                d_green = (
                    (1.0 - red) * (1.0 - blue) * (c010 - c000)
                    + red * (1.0 - blue) * (c110 - c100)
                    + (1.0 - red) * blue * (c011 - c001)
                    + red * blue * (c111 - c101)
                ) / steps[1]
                d_blue = (
                    (1.0 - red) * (1.0 - green) * (c001 - c000)
                    + red * (1.0 - green) * (c101 - c100)
                    + (1.0 - red) * green * (c011 - c010)
                    + red * green * (c111 - c110)
                ) / steps[2]
                results.append(np.stack((d_red, d_green, d_blue), axis=-1))
    return np.stack(results, axis=0).reshape(-1, 3, 3)


def _jacobian_summary(jacobians: np.ndarray) -> dict[str, float | int]:
    determinants = np.linalg.det(jacobians)
    spectral_norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    return {
        "samples": int(len(jacobians)),
        "minimum_determinant": float(np.min(determinants)),
        "p01_determinant": float(np.quantile(determinants, 0.01)),
        "nonpositive_determinant_count": int(np.sum(determinants <= 0.0)),
        "nonpositive_determinant_fraction": float(
            np.mean(determinants <= 0.0)
        ),
        "maximum_spectral_norm": float(np.max(spectral_norms)),
    }


def analyze_asset(
    asset: CubeAsset,
    gates: Mapping[str, Any],
    subcell_axis: Iterable[float],
) -> dict[str, Any]:
    values = asset.values
    identity = identity_grid(values.shape[0])
    displacement = values - identity
    design = np.column_stack(
        (np.ones(values.shape[0] ** 3), identity.reshape(-1, 3))
    )
    coefficients = np.linalg.lstsq(
        design, values.reshape(-1, 3), rcond=None
    )[0]
    affine = design @ coefficients
    tetrahedral = _jacobian_summary(tetrahedral_jacobians(asset))
    trilinear = _jacobian_summary(trilinear_jacobians(asset, subcell_axis))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    range_pass = (
        minimum >= float(gates["minimum_output_node"])
        and maximum <= float(gates["maximum_output_node"])
    )
    orientation_pass = (
        float(tetrahedral["minimum_determinant"])
        > float(gates["minimum_jacobian_determinant_exclusive"])
        and float(trilinear["minimum_determinant"])
        > float(gates["minimum_jacobian_determinant_exclusive"])
    )
    norm_pass = (
        float(tetrahedral["maximum_spectral_norm"])
        <= float(gates["maximum_jacobian_spectral_norm"])
        and float(trilinear["maximum_spectral_norm"])
        <= float(gates["maximum_jacobian_spectral_norm"])
    )
    return {
        "dimension": int(values.shape[0]),
        "minimum_node": minimum,
        "maximum_node": maximum,
        "identity_maximum_absolute_error": float(np.max(np.abs(displacement))),
        "identity_rmse": float(np.sqrt(np.mean(displacement**2))),
        "best_affine_residual_rmse": float(
            np.sqrt(np.mean((affine - values.reshape(-1, 3)) ** 2))
        ),
        "tetrahedral": tetrahedral,
        "trilinear": trilinear,
        "range_pass": bool(range_pass),
        "orientation_pass": bool(orientation_pass),
        "jacobian_norm_pass": bool(norm_pass),
        "structural_pass": bool(range_pass and orientation_pass and norm_pass),
    }


__all__ = [
    "CubeAsset",
    "PublishedDLUTError",
    "analyze_asset",
    "canonical_lut_manifest",
    "identity_grid",
    "parse_cube",
    "sha256_file",
    "tetrahedral_jacobians",
    "trilinear_jacobians",
]
