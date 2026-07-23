"""Generated bounded colour operators for method-identifiability research."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np


SYNTHETIC_OPERATOR_SCHEMA = "roll2film.synthetic_bounded_operator.v1"
_IDENTITY_TONE = np.array([0.25, 0.5, 0.75], dtype=np.float64)
_PALETTES = (
    "balanced",
    "warm",
    "cool",
    "foliage_like",
    "skin_like",
    "low_key",
    "red_omitted_narrow",
    "blue_omitted_narrow",
    "green_omitted_narrow",
)


def _as_parameters(value: np.ndarray | Iterable[float]) -> np.ndarray:
    parameters = np.asarray(value, dtype=np.float64)
    if parameters.shape != (9,) or not np.all(np.isfinite(parameters)):
        raise ValueError("operator parameters must be nine finite values")
    return parameters


def project_operator_parameters(
    parameters: np.ndarray | Iterable[float],
    *,
    maximum_off_diagonal_sum: float = 0.3,
    minimum_tone_increment: float = 0.08,
) -> np.ndarray:
    """Deterministically map arbitrary predictions into the feasible family."""

    if not 0.0 < maximum_off_diagonal_sum < 1.0:
        raise ValueError("maximum_off_diagonal_sum must be in (0, 1)")
    if not 0.0 < minimum_tone_increment < 0.25:
        raise ValueError("minimum_tone_increment must be in (0, 0.25)")
    raw = _as_parameters(parameters)
    result = raw.copy()
    for start in (0, 2, 4):
        pair = np.maximum(result[start : start + 2], 0.0)
        total = float(pair.sum())
        if total > maximum_off_diagonal_sum:
            pair *= maximum_off_diagonal_sum / total
        result[start : start + 2] = pair

    tone = np.concatenate(([0.0], result[6:9], [1.0]))
    raw_increments = np.maximum(np.diff(tone), 0.0)
    remaining = 1.0 - 4.0 * minimum_tone_increment
    excess = np.maximum(raw_increments - minimum_tone_increment, 0.0)
    if float(excess.sum()) <= 1e-15:
        weights = np.full(4, 0.25, dtype=np.float64)
    else:
        weights = excess / excess.sum()
    increments = minimum_tone_increment + remaining * weights
    result[6:9] = np.cumsum(increments)[:3]
    return result


def parameters_to_matrix(parameters: np.ndarray | Iterable[float]) -> np.ndarray:
    values = _as_parameters(parameters)
    matrix = np.zeros((3, 3), dtype=np.float64)
    matrix[0] = [1.0 - values[0] - values[1], values[0], values[1]]
    matrix[1] = [values[2], 1.0 - values[2] - values[3], values[3]]
    matrix[2] = [values[4], values[5], 1.0 - values[4] - values[5]]
    return matrix


@dataclass(frozen=True)
class SyntheticBoundedOperator:
    parameters: np.ndarray
    family: str
    operator_id: str

    def __post_init__(self) -> None:
        parameters = _as_parameters(self.parameters)
        if self.family not in {"matrix_only", "tone_only", "combined"}:
            raise ValueError(f"unsupported operator family: {self.family!r}")
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty")
        parameters = parameters.copy()
        parameters.setflags(write=False)
        object.__setattr__(self, "parameters", parameters)
        report = audit_synthetic_operator(self)
        if not report["valid"]:
            raise ValueError(f"operator violates the synthetic contract: {report}")

    @property
    def matrix(self) -> np.ndarray:
        return parameters_to_matrix(self.parameters)

    @property
    def tone_knots(self) -> np.ndarray:
        return np.concatenate(([0.0], self.parameters[6:9], [1.0]))

    def apply(self, encoded_srgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(encoded_srgb, dtype=np.float64)
        if (
            rgb.ndim < 2
            or rgb.shape[-1] != 3
            or not np.all(np.isfinite(rgb))
            or np.any(rgb < 0.0)
            or np.any(rgb > 1.0)
        ):
            raise ValueError("encoded_srgb must be finite [0, 1] data with shape (..., 3)")
        mixed = rgb @ self.matrix.T
        x_knots = np.linspace(0.0, 1.0, 5)
        return np.stack(
            [np.interp(mixed[..., channel], x_knots, self.tone_knots) for channel in range(3)],
            axis=-1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SYNTHETIC_OPERATOR_SCHEMA,
            "operator_id": self.operator_id,
            "family": self.family,
            "parameters": self.parameters.tolist(),
            "matrix_determinant": float(np.linalg.det(self.matrix)),
        }


def audit_synthetic_operator(
    operator: SyntheticBoundedOperator,
    *,
    minimum_determinant: float = 0.2,
    maximum_off_diagonal_sum: float = 0.3,
    minimum_tone_increment: float = 0.08,
    neutral_axis_tolerance: float = 1e-10,
) -> dict[str, Any]:
    parameters = _as_parameters(operator.parameters)
    matrix = parameters_to_matrix(parameters)
    off_diagonal_sums = np.array(
        [parameters[0:2].sum(), parameters[2:4].sum(), parameters[4:6].sum()]
    )
    determinant = float(np.linalg.det(matrix))
    tone_knots = np.concatenate(([0.0], parameters[6:9], [1.0]))
    tone_increments = np.diff(tone_knots)
    neutral = np.linspace(0.0, 1.0, 33)
    neutral_rgb = np.repeat(neutral[:, None], 3, axis=1)
    output = operator.apply(neutral_rgb)
    neutral_error = float(np.max(np.ptp(output, axis=1)))
    row_sum_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    valid = bool(
        np.all(parameters[:6] >= 0.0)
        and np.all(off_diagonal_sums <= maximum_off_diagonal_sum + 1e-12)
        and determinant >= minimum_determinant
        and np.all(tone_increments >= minimum_tone_increment - 1e-12)
        and row_sum_error <= 1e-12
        and neutral_error <= neutral_axis_tolerance
        and np.min(output) >= 0.0
        and np.max(output) <= 1.0
    )
    return {
        "valid": valid,
        "matrix_determinant": determinant,
        "maximum_off_diagonal_sum": float(off_diagonal_sums.max()),
        "minimum_tone_increment": float(tone_increments.min()),
        "row_sum_error": row_sum_error,
        "neutral_axis_error": neutral_error,
        "neutral_output_minimum": float(output.min()),
        "neutral_output_maximum": float(output.max()),
    }


def _sample_matrix_parameters(rng: np.random.Generator, active: bool) -> np.ndarray:
    if not active:
        return np.zeros(6, dtype=np.float64)
    result = np.empty(6, dtype=np.float64)
    for start in (0, 2, 4):
        total = rng.uniform(0.06, 0.3)
        fraction = rng.uniform(0.12, 0.88)
        result[start : start + 2] = [total * fraction, total * (1.0 - fraction)]
    return result


def _sample_tone_parameters(rng: np.random.Generator, active: bool) -> np.ndarray:
    if not active:
        return _IDENTITY_TONE.copy()
    increments = 0.08 + 0.68 * rng.dirichlet(np.full(4, 2.0))
    tone = np.cumsum(increments)[:3]
    if np.max(np.abs(tone - _IDENTITY_TONE)) < 0.04:
        # Deterministic anti-trivial reflection around the identity midpoints.
        tone = np.array([1.0 - tone[2], 1.0 - tone[1], 1.0 - tone[0]])
    return tone


def generate_operator_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    specification = config["operator"]
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    counts = (
        ("fit", int(specification["fit_per_family"])),
        ("validation", int(specification["validation_per_family"])),
        ("confirmation", int(specification["confirmation_per_family"])),
        ("stress", int(specification["stress_per_family"])),
    )
    rows: list[dict[str, Any]] = []
    for family in specification["families"]:
        family_parameters = []
        for _ in range(int(specification["operators_per_family"])):
            matrix = _sample_matrix_parameters(rng, family != "tone_only")
            tone = _sample_tone_parameters(rng, family != "matrix_only")
            family_parameters.append(np.concatenate((matrix, tone)))
        cursor = 0
        for split, count in counts:
            for within_split in range(count):
                parameters = family_parameters[cursor]
                operator_id = f"{family}-{cursor:04d}"
                operator = SyntheticBoundedOperator(parameters, family, operator_id)
                rows.append(
                    {
                        **operator.to_dict(),
                        "split": split,
                        "family_index": cursor,
                        "within_split_index": within_split,
                    }
                )
                cursor += 1
        if cursor != len(family_parameters):
            raise ValueError("configured split counts do not exhaust each operator family")
    return rows


def operator_from_manifest(row: dict[str, Any]) -> SyntheticBoundedOperator:
    if row.get("schema") != SYNTHETIC_OPERATOR_SCHEMA:
        raise ValueError("unsupported synthetic operator manifest schema")
    return SyntheticBoundedOperator(
        parameters=np.asarray(row["parameters"], dtype=np.float64),
        family=str(row["family"]),
        operator_id=str(row["operator_id"]),
    )


def palette_cloud(name: str, pixel_count: int, seed: int) -> np.ndarray:
    if name not in _PALETTES:
        raise ValueError(f"unsupported palette: {name!r}")
    if pixel_count <= 0:
        raise ValueError("pixel_count must be positive")
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    if name == "balanced":
        cloud = rng.uniform(0.0, 1.0, size=(pixel_count, 3))
    elif name == "warm":
        cloud = np.column_stack(
            (rng.beta(4, 2, pixel_count), rng.beta(2.5, 2.5, pixel_count), rng.beta(2, 5, pixel_count))
        )
    elif name == "cool":
        cloud = np.column_stack(
            (rng.beta(2, 5, pixel_count), rng.beta(2.5, 2.5, pixel_count), rng.beta(4, 2, pixel_count))
        )
    elif name == "foliage_like":
        cloud = np.column_stack(
            (rng.beta(2.5, 4, pixel_count), rng.beta(4, 2, pixel_count), rng.beta(2, 5, pixel_count))
        )
    elif name == "skin_like":
        base = rng.normal([0.68, 0.45, 0.34], [0.13, 0.10, 0.09], size=(pixel_count, 3))
        cloud = np.clip(base, 0.0, 1.0)
    elif name == "low_key":
        cloud = rng.beta(1.2, 5.5, size=(pixel_count, 3))
    else:
        cloud = rng.beta(2.2, 2.2, size=(pixel_count, 3))
        omitted = {"red_omitted_narrow": 0, "green_omitted_narrow": 1, "blue_omitted_narrow": 2}[name]
        cloud[:, omitted] *= 0.18
        cloud = 0.15 + 0.7 * cloud
    return np.asarray(cloud, dtype=np.float64)


def palette_pairs(
    names: list[str],
    *,
    operator_index: int,
    observations: int = 2,
) -> list[tuple[str, str]]:
    if len(names) < 2 or observations <= 0:
        raise ValueError("at least two palette names and one observation are required")
    pairs = []
    for observation in range(observations):
        query_index = (operator_index + observation) % len(names)
        reference_index = (operator_index + observation + 1) % len(names)
        query = names[query_index]
        reference = names[reference_index]
        if query == reference:
            raise RuntimeError("query and reference palettes must differ")
        pairs.append((query, reference))
    return pairs


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uniform_probe_grid(size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("probe size must be at least two")
    axis = np.linspace(0.0, 1.0, size)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
