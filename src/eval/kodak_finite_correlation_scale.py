"""U6.P4BI cross-density finite-correlation-scale evaluator."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.gaussian_aperture_variance import (
    gaussian_covariance_aperture_rms_ratio,
)

SCHEMA = "neuro_film.u6_p4bi_kodak_finite_correlation_scale_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bi_kodak_finite_correlation_scale_report.v1"


class KodakFiniteCorrelationScaleError(RuntimeError):
    """Raised when frozen P4BI evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise KodakFiniteCorrelationScaleError("P4BI paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    controls = payload.get("controls", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or candidate.get("covariance")
        != "C(r) = variance * exp(-r^2 / (2 * correlation_scale_um^2))"
        or candidate.get("measurement")
        != "continuous circular aperture area average"
        or candidate.get("normalization_aperture_micrometres") != 48.0
        or candidate.get("scale_grid_micrometres")
        != {"minimum": 0.1, "maximum": 24.0, "count": 481, "spacing": "geometric"}
        or candidate.get("quadrature")
        != {
            "method": "Gauss-Legendre on normalized disk-overlap integral",
            "order": 512,
        }
        or candidate.get("fit_loss")
        != "mean_squared_log_ratio_on_development_groups"
        or candidate.get("parameters_per_film") != 1
        or candidate.get("density_specific_parameters_allowed") is not False
        or candidate.get("confirmation_refit_allowed") is not False
        or controls.get("selwyn_zero_scale")
        != "sigma(d) = sigma(48um) * 48um / d"
        or controls.get("wrong_film")
        != "cyclic permutation of frozen development-fitted film scales"
        or controls.get("density_specific_oracle")
        != "diagnostic only; fit each confirmation density independently"
        or controls.get("development_leave_one_group_out") is not True
        or evaluation.get("required_development_group_count") != 12
        or evaluation.get("required_confirmation_group_count") != 11
        or evaluation.get("required_confirmation_comparison_count") != 66
        or evaluation.get("maximum_confirmation_median_relative_error") != 0.045
        or evaluation.get("maximum_confirmation_p90_relative_error") != 0.2
        or evaluation.get("maximum_confirmation_worst_relative_error") != 0.65
        or evaluation.get("minimum_median_improvement_over_selwyn") != 0.1
        or evaluation.get("minimum_per_film_improvement_over_selwyn") != 0.0
        or evaluation.get("minimum_median_improvement_over_wrong_film") != 0.05
        or evaluation.get("maximum_leave_one_group_out_scale_relative_span") != 0.5
        or evaluation.get("require_grid_interior_scale") is not True
        or evaluation.get("require_reversed_development_order_exact_scale") is not True
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise KodakFiniteCorrelationScaleError("P4BI frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise KodakFiniteCorrelationScaleError(
            f"P4BI parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _group_ratios(group: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    measurements = {
        float(row["aperture_diameter_micrometres"]): float(
            row["density_standard_deviation"]
        )
        for row in group["measurements"]
    }
    base = measurements[48.0]
    apertures = np.asarray(sorted(value for value in measurements if value != 48.0))
    ratios = np.asarray([measurements[float(value)] / base for value in apertures])
    return apertures, ratios


def _fit_scale(
    groups: Sequence[Mapping[str, Any]],
    scale_grid: np.ndarray,
    model_ratios: np.ndarray,
) -> tuple[float, int, float]:
    ordered = sorted(groups, key=lambda group: float(group["density"]))
    observed = np.stack([_group_ratios(group)[1] for group in ordered])
    residual = np.log(model_ratios[:, None, :] / observed[None, :, :])
    losses = np.mean(np.square(residual), axis=(1, 2), dtype=np.float64)
    index = int(np.argmin(losses))
    return float(scale_grid[index]), index, float(losses[index])


def _relative_errors(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.abs(predicted / observed - 1.0)


def evaluate_scale(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    p4be_decision = _load_parent(parents, root, "p4be_decision")
    source = _load_parent(parents, root, "p4be_report")
    p4bf_decision = _load_parent(parents, root, "p4bf_decision")
    p4bf_report = _load_parent(parents, root, "p4bf_report")
    if (
        p4be_decision.get("source_pass") is not True
        or source.get("stable_evidence_id") != parents["p4be_stable_evidence_id"]
        or p4bf_decision.get("automatic_pass") is not True
        or p4bf_report.get("stable_evidence_id")
        != parents["p4bf_stable_evidence_id"]
        or p4bf_report.get("automatic_pass") is not True
    ):
        raise KodakFiniteCorrelationScaleError("P4BI parent decision mismatch")

    roles = contract["roles"]
    by_film_density = {
        (str(group["film"]), float(group["density"])): group
        for group in source["density_groups"]
    }
    development: dict[str, list[dict[str, Any]]] = {}
    confirmation: dict[str, list[dict[str, Any]]] = {}
    consumed: set[tuple[str, float]] = set()
    for film, split in roles.items():
        development[film] = []
        confirmation[film] = []
        for role, target in (
            ("development_density", development[film]),
            ("confirmation_density", confirmation[film]),
        ):
            for density in split[role]:
                key = (film, float(density))
                if key not in by_film_density or key in consumed:
                    raise KodakFiniteCorrelationScaleError("P4BI role inventory drift")
                consumed.add(key)
                target.append(by_film_density[key])
    if consumed != set(by_film_density):
        raise KodakFiniteCorrelationScaleError("P4BI role coverage drift")

    grid_spec = contract["candidate"]["scale_grid_micrometres"]
    scale_grid = np.geomspace(
        float(grid_spec["minimum"]),
        float(grid_spec["maximum"]),
        int(grid_spec["count"]),
    )
    apertures, _ = _group_ratios(source["density_groups"][0])
    order = int(contract["candidate"]["quadrature"]["order"])
    model_ratios = gaussian_covariance_aperture_rms_ratio(
        apertures[None, :],
        reference_aperture_micrometres=48.0,
        correlation_scale_micrometres=scale_grid[:, None],
        quadrature_order=order,
    )
    selwyn_ratios = 48.0 / apertures

    scales: dict[str, float] = {}
    scale_indices: dict[str, int] = {}
    development_losses: dict[str, float] = {}
    reversed_results: list[bool] = []
    loo_spans: dict[str, float] = {}
    for film, groups in development.items():
        scale, index, loss = _fit_scale(groups, scale_grid, model_ratios)
        reverse_scale, reverse_index, _ = _fit_scale(
            list(reversed(groups)), scale_grid, model_ratios
        )
        scales[film] = scale
        scale_indices[film] = index
        development_losses[film] = loss
        reversed_results.append(scale == reverse_scale and index == reverse_index)
        loo = [
            _fit_scale(groups[:omit] + groups[omit + 1 :], scale_grid, model_ratios)[0]
            for omit in range(len(groups))
        ]
        loo_spans[film] = (max(loo) - min(loo)) / float(np.median(loo))

    ordered_films = sorted(scales)
    wrong_scale = {
        film: scales[ordered_films[(index + 1) % len(ordered_films)]]
        for index, film in enumerate(ordered_films)
    }
    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    selwyn_errors: list[float] = []
    wrong_errors: list[float] = []
    oracle_errors: list[float] = []
    candidate_by_film: defaultdict[str, list[float]] = defaultdict(list)
    selwyn_by_film: defaultdict[str, list[float]] = defaultdict(list)
    for film, groups in confirmation.items():
        candidate = model_ratios[scale_indices[film]]
        wrong_index = int(np.argmin(np.abs(scale_grid - wrong_scale[film])))
        wrong = model_ratios[wrong_index]
        for group in groups:
            observed_apertures, observed = _group_ratios(group)
            if not np.array_equal(observed_apertures, apertures):
                raise KodakFiniteCorrelationScaleError("P4BI aperture order drift")
            oracle_scale, oracle_index, _ = _fit_scale(
                [group], scale_grid, model_ratios
            )
            candidate_error = _relative_errors(candidate, observed)
            selwyn_error = _relative_errors(selwyn_ratios, observed)
            wrong_error = _relative_errors(wrong, observed)
            oracle_error = _relative_errors(model_ratios[oracle_index], observed)
            candidate_errors.extend(float(value) for value in candidate_error)
            selwyn_errors.extend(float(value) for value in selwyn_error)
            wrong_errors.extend(float(value) for value in wrong_error)
            oracle_errors.extend(float(value) for value in oracle_error)
            candidate_by_film[film].extend(float(value) for value in candidate_error)
            selwyn_by_film[film].extend(float(value) for value in selwyn_error)
            rows.append(
                {
                    "film": film,
                    "density": group["density"],
                    "candidate_scale_micrometres": scales[film],
                    "wrong_film_scale_micrometres": wrong_scale[film],
                    "density_specific_oracle_scale_micrometres": oracle_scale,
                    "candidate_relative_errors": candidate_error.tolist(),
                    "selwyn_relative_errors": selwyn_error.tolist(),
                    "wrong_film_relative_errors": wrong_error.tolist(),
                    "density_specific_oracle_relative_errors": oracle_error.tolist(),
                }
            )

    candidate_median = float(np.median(candidate_errors))
    selwyn_median = float(np.median(selwyn_errors))
    wrong_median = float(np.median(wrong_errors))
    improvement_selwyn = 1.0 - candidate_median / selwyn_median
    improvement_wrong = 1.0 - candidate_median / wrong_median
    per_film_improvement = {
        film: 1.0
        - float(np.median(candidate_by_film[film]))
        / float(np.median(selwyn_by_film[film]))
        for film in ordered_films
    }
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "development_group_count": sum(map(len, development.values()))
        == int(gates["required_development_group_count"]),
        "confirmation_group_count": sum(map(len, confirmation.values()))
        == int(gates["required_confirmation_group_count"]),
        "confirmation_comparison_count": len(candidate_errors)
        == int(gates["required_confirmation_comparison_count"]),
        "confirmation_median_error": candidate_median
        <= float(gates["maximum_confirmation_median_relative_error"]),
        "confirmation_p90_error": float(np.quantile(candidate_errors, 0.9))
        <= float(gates["maximum_confirmation_p90_relative_error"]),
        "confirmation_worst_error": max(candidate_errors)
        <= float(gates["maximum_confirmation_worst_relative_error"]),
        "improvement_over_selwyn": improvement_selwyn
        >= float(gates["minimum_median_improvement_over_selwyn"]),
        "per_film_improvement_over_selwyn": min(per_film_improvement.values())
        >= float(gates["minimum_per_film_improvement_over_selwyn"]),
        "improvement_over_wrong_film": improvement_wrong
        >= float(gates["minimum_median_improvement_over_wrong_film"]),
        "leave_one_group_out_scale_stability": max(loo_spans.values())
        <= float(gates["maximum_leave_one_group_out_scale_relative_span"]),
        "grid_interior_scale": min(scale_indices.values()) > 0
        and max(scale_indices.values()) < len(scale_grid) - 1,
        "reversed_development_order_exact_scale": all(reversed_results),
        "confirmation_refit_forbidden": contract["candidate"][
            "confirmation_refit_allowed"
        ]
        is False,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "development_group_count": sum(map(len, development.values())),
        "confirmation_group_count": sum(map(len, confirmation.values())),
        "confirmation_comparison_count": len(candidate_errors),
        "fitted_correlation_scale_micrometres": scales,
        "development_loss": development_losses,
        "leave_one_group_out_scale_relative_span": loo_spans,
        "confirmation_median_relative_error": candidate_median,
        "confirmation_p90_relative_error": float(np.quantile(candidate_errors, 0.9)),
        "confirmation_worst_relative_error": max(candidate_errors),
        "selwyn_confirmation_median_relative_error": selwyn_median,
        "wrong_film_confirmation_median_relative_error": wrong_median,
        "density_specific_oracle_confirmation_median_relative_error": float(
            np.median(oracle_errors)
        ),
        "median_improvement_over_selwyn": improvement_selwyn,
        "per_film_improvement_over_selwyn": per_film_improvement,
        "median_improvement_over_wrong_film": improvement_wrong,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_historical_film_finite_correlation_scale_prior"
            if automatic_pass
            else "close_finite_gaussian_correlation_scale_identification"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "KodakFiniteCorrelationScaleError",
    "evaluate_scale",
    "load_contract",
    "write_report",
]
