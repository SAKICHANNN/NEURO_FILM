"""U6.P2M synthetic recovery of density-dependent interimage development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.film_physics.interimage_development import (
    InterimageDevelopmentOperator,
    independent_development_operator,
)


SCHEMA = "neuro_film.u6_p2m_density_interimage_recovery_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2M contract")
    return payload


def _operator(row: dict[str, Any]) -> InterimageDevelopmentOperator:
    return InterimageDevelopmentOperator(
        tuple(row["density_min"]),
        tuple(row["density_max"]),
        tuple(row["slope"]),
        tuple(row["midpoint"]),
        tuple(tuple(value) for value in row["coupling"]),
    )


def _sample_group(
    rng: np.random.Generator,
    group: str,
    rows: int,
    low: float,
    high: float,
) -> np.ndarray:
    if group == "uniform":
        return rng.uniform(low, high, size=(rows, 3))
    if group == "warm-biased":
        values = rng.uniform(low, high, size=(rows, 3))
        values[:, 0] = np.clip(values[:, 0] + 1.1, low, high)
        values[:, 2] = np.clip(values[:, 2] - 0.8, low, high)
        return values
    if group == "cool-biased":
        values = rng.uniform(low, high, size=(rows, 3))
        values[:, 0] = np.clip(values[:, 0] - 0.8, low, high)
        values[:, 2] = np.clip(values[:, 2] + 1.1, low, high)
        return values
    if group == "diagonal-ramp":
        base = np.linspace(low, high, rows, dtype=np.float64)
        offsets = rng.uniform(-0.9, 0.9, size=(rows, 3))
        return np.clip(base[:, None] + offsets, low, high)
    if group == "corner-mixture":
        corners = rng.integers(0, 2, size=(rows, 3))
        values = np.where(corners == 0, low + 0.35, high - 0.35)
        return np.clip(values + rng.normal(0.0, 0.3, size=(rows, 3)), low, high)
    raise ValueError(f"unknown exposure group: {group}")


def _dataset(
    operator: InterimageDevelopmentOperator,
    *,
    seed: int,
    groups: list[str],
    rows_per_group: int,
    exposure_range: tuple[float, float],
    noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    exposure_rows = []
    group_ids = []
    for index, group in enumerate(groups):
        exposure_rows.append(
            _sample_group(
                rng,
                group,
                rows_per_group,
                exposure_range[0],
                exposure_range[1],
            )
        )
        group_ids.extend([index] * rows_per_group)
    exposure = np.concatenate(exposure_rows, axis=0)
    density = operator.apply_log2_exposure(exposure)
    density = density + rng.normal(0.0, noise_sigma, size=density.shape)
    return exposure, density, np.asarray(group_ids, dtype=np.int64)


def _rmse(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(expected - actual))))


def _fit_coupling(
    base: InterimageDevelopmentOperator,
    exposure: np.ndarray,
    density: np.ndarray,
) -> InterimageDevelopmentOperator:
    result = least_squares(
        lambda values: (
            base.with_coupling_vector(values).apply_log2_exposure(exposure)
            - density
        ).ravel(),
        x0=np.full(6, 0.1, dtype=np.float64),
        bounds=(0.0, base.maximum_coupling),
        method="trf",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=800,
    )
    if not result.success:
        raise RuntimeError(f"coupling fit failed: {result.message}")
    return base.with_coupling_vector(result.x)


def _fit_affine(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate(
        [source, np.ones((source.shape[0], 1), dtype=np.float64)], axis=1
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients[:3, :], coefficients[3, :]


def _apply_affine(
    source: np.ndarray, matrix: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    return source @ matrix + offset


def evaluate_interimage_recovery(contract: dict[str, Any]) -> dict[str, Any]:
    sampling = contract["sampling"]
    gates = contract["automatic_gates"]
    exposure_range = tuple(float(value) for value in sampling["log2_exposure_range"])
    witness_rows = []
    for witness_index, witness in enumerate(contract["synthetic_witnesses"]):
        truth = _operator(witness)
        independent = independent_development_operator(truth)
        development = _dataset(
            truth,
            seed=int(sampling["development_seed"]) + witness_index * 1009,
            groups=list(sampling["development_groups"]),
            rows_per_group=int(sampling["development_rows_per_group"]),
            exposure_range=exposure_range,
            noise_sigma=float(sampling["measurement_noise_sigma_density"]),
        )
        confirmation = _dataset(
            truth,
            seed=int(sampling["confirmation_seed"]) + witness_index * 1013,
            groups=list(sampling["confirmation_groups"]),
            rows_per_group=int(sampling["confirmation_rows_per_group"]),
            exposure_range=exposure_range,
            noise_sigma=float(sampling["measurement_noise_sigma_density"]),
        )
        development_x, development_y, _ = development
        confirmation_x, confirmation_y, confirmation_group = confirmation
        fitted = _fit_coupling(independent, development_x, development_y)

        independent_development = independent.apply_log2_exposure(development_x)
        matrix, offset = _fit_affine(independent_development, development_y)
        candidate_y = fitted.apply_log2_exposure(confirmation_x)
        independent_y = independent.apply_log2_exposure(confirmation_x)
        affine_y = _apply_affine(independent_y, matrix, offset)
        candidate_rmse = _rmse(confirmation_y, candidate_y)
        independent_rmse = _rmse(confirmation_y, independent_y)
        affine_rmse = _rmse(confirmation_y, affine_y)

        derivative_points = np.concatenate(
            [
                confirmation_x,
                np.asarray(
                    [
                        [a, b, c]
                        for a in np.linspace(exposure_range[0], exposure_range[1], 9)
                        for b in np.linspace(exposure_range[0], exposure_range[1], 9)
                        for c in np.linspace(exposure_range[0], exposure_range[1], 9)
                    ],
                    dtype=np.float64,
                ),
            ],
            axis=0,
        )
        jacobian = fitted.derivative_log2_exposure(derivative_points)
        own = np.stack([jacobian[:, i, i] for i in range(3)], axis=-1)
        cross = np.stack(
            [
                jacobian[:, row, column]
                for row in range(3)
                for column in range(3)
                if row != column
            ],
            axis=-1,
        )
        minimum = np.asarray(fitted.density_min)
        maximum = np.asarray(fitted.density_max)
        domain_violation = max(
            0.0,
            float(np.max(minimum - candidate_y)),
            float(np.max(candidate_y - maximum)),
        )
        group_rmse = {
            str(group): _rmse(
                confirmation_y[confirmation_group == group],
                candidate_y[confirmation_group == group],
            )
            for group in np.unique(confirmation_group)
        }
        coupling_error = float(
            np.max(np.abs(fitted.coupling_vector() - truth.coupling_vector()))
        )
        zero_exact = np.array_equal(
            independent.apply_log2_exposure(confirmation_x),
            truth.with_coupling_vector(np.zeros(6)).apply_log2_exposure(
                confirmation_x
            ),
        )
        neutral_spread = None
        if witness["id"] == "symmetric-moderate":
            neutral_x = np.repeat(
                np.linspace(exposure_range[0], exposure_range[1], 257)[:, None],
                3,
                axis=1,
            )
            neutral_y = fitted.apply_log2_exposure(neutral_x)
            neutral_spread = float(
                np.max(np.ptp(neutral_y, axis=-1))
            )
        decisions = {
            "absolute_rmse": candidate_rmse
            <= float(gates["maximum_confirmation_rmse_density"]),
            "gain_over_independent": 1.0 - candidate_rmse / independent_rmse
            >= float(gates["minimum_gain_over_independent_fraction"]),
            "gain_over_affine": 1.0 - candidate_rmse / affine_rmse
            >= float(gates["minimum_gain_over_affine_fraction"]),
            "coupling_recovery": coupling_error
            <= float(gates["maximum_coupling_absolute_error"]),
            "own_monotonicity": float(np.min(own))
            >= float(gates["minimum_own_log_exposure_derivative"]),
            "cross_suppression": float(np.max(cross))
            <= float(gates["maximum_cross_log_exposure_derivative"]),
            "density_domain": domain_violation
            <= float(gates["maximum_density_domain_violation"]),
            "zero_coupling": zero_exact,
        }
        if neutral_spread is not None:
            decisions["symmetric_neutral"] = neutral_spread <= float(
                gates["maximum_symmetric_neutral_channel_spread"]
            )
        witness_rows.append(
            {
                "id": witness["id"],
                "metrics": {
                    "candidate_confirmation_rmse_density": candidate_rmse,
                    "independent_confirmation_rmse_density": independent_rmse,
                    "affine_confirmation_rmse_density": affine_rmse,
                    "gain_over_independent_fraction": (
                        1.0 - candidate_rmse / independent_rmse
                    ),
                    "gain_over_affine_fraction": 1.0 - candidate_rmse / affine_rmse,
                    "maximum_coupling_absolute_error": coupling_error,
                    "minimum_own_log_exposure_derivative": float(np.min(own)),
                    "maximum_cross_log_exposure_derivative": float(np.max(cross)),
                    "density_domain_violation": domain_violation,
                    "symmetric_neutral_channel_spread": neutral_spread,
                    "confirmation_group_rmse_density": group_rmse,
                    "truth_coupling": truth.coupling_vector().tolist(),
                    "fitted_coupling": fitted.coupling_vector().tolist(),
                },
                "decisions": decisions,
                "passed": all(decisions.values()),
            }
        )

    automatic_pass = all(row["passed"] for row in witness_rows)
    core = {
        "schema": "neuro_film.u6_p2m_density_interimage_recovery_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "witnesses": witness_rows,
        "automatic_pass": automatic_pass,
        "branch": contract["branch_rule"]["pass" if automatic_pass else "fail"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()

