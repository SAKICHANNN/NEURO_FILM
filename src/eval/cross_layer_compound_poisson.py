"""Evaluate the shared-component cross-layer Poisson reference primitive."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    cross_layer_counts_to_density,
    sample_cross_layer_poisson_region,
)
from src.film_physics.density_conditioned_structure import counter_poisson_rate_field

SCHEMA = "neuro_film.u6_p4cx_cross_layer_compound_poisson_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cx_cross_layer_compound_poisson_report.v1"


class CrossLayerPoissonError(RuntimeError):
    """Raised when the frozen P4CX experiment drifts."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "contract_frozen_implementation_ready"
    ):
        raise CrossLayerPoissonError("P4CX contract identity drift")
    for parent in payload["parents"].values():
        parent_path = root / parent["path"]
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        if (
            sha256_file(parent_path) != parent["sha256"]
            or parent_payload.get("decision") != parent["required_decision"]
        ):
            raise CrossLayerPoissonError("P4CX parent drift")
    field = payload["field"]
    if (
        field.get("shape") != [512, 768]
        or set(field["development_seeds"]) & set(field["confirmation_seeds"])
        or field.get("row_partitions") != [31, 127]
    ):
        raise CrossLayerPoissonError("P4CX field drift")
    return payload


def _profile(field: Mapping[str, Any], seed: int) -> CrossLayerPoissonProfile:
    return CrossLayerPoissonProfile(
        tuple(float(value) for value in field["marginal_count_rates_cmy"]),
        float(field["shared_all_rate"]),
        tuple(float(value) for value in field["shared_pair_rates_cm_cy_my"]),
        tuple(float(value) for value in field["mark_optical_density_cmy"]),
        seed,
        int(field["component_seed_stride"]),
    )


def _partitioned(
    profile: CrossLayerPoissonProfile, shape: tuple[int, int], rows: int
) -> np.ndarray:
    pieces = []
    for y0 in range(0, shape[0], rows):
        height = min(rows, shape[0] - y0)
        pieces.append(
            sample_cross_layer_poisson_region(
                profile, shape, origin_yx=(y0, 0), shape=(height, shape[1])
            )
        )
    return np.concatenate(pieces, axis=0)


def _correlation(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1, 3).astype(np.float64)
    return np.corrcoef(flattened, rowvar=False)


def _independent_control(
    profile: CrossLayerPoissonProfile, shape: tuple[int, int]
) -> np.ndarray:
    fields = []
    for index, rate in enumerate(profile.marginal_rates_cmy):
        fields.append(
            counter_poisson_rate_field(
                np.full(shape, rate, dtype=np.float64),
                shape,
                origin_yx=(0, 0),
                seed=(profile.seed + (index + 11) * profile.component_seed_stride)
                % (2**64),
                maximum_rate=rate,
            )
        )
    return np.stack(fields, axis=-1)


def _row(seed: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    field = contract["field"]
    shape = tuple(int(value) for value in field["shape"])
    profile = _profile(field, seed)
    counts = sample_cross_layer_poisson_region(
        profile, shape, origin_yx=(0, 0), shape=shape
    )
    repeat = sample_cross_layer_poisson_region(
        profile, shape, origin_yx=(0, 0), shape=shape
    )
    partition_exact = all(
        np.array_equal(counts, _partitioned(profile, shape, rows))
        for rows in field["row_partitions"]
    )
    density = cross_layer_counts_to_density(profile, counts)
    transmittance = np.power(np.float32(10.0), -density, dtype=np.float32)
    empirical = _correlation(counts)
    analytic = profile.analytic_correlation()
    means = np.mean(counts, axis=(0, 1), dtype=np.float64)
    variances = np.var(counts, axis=(0, 1), dtype=np.float64)
    expected = np.asarray(profile.marginal_rates_cmy, dtype=np.float64)
    independent = _correlation(_independent_control(profile, shape))
    offdiag = np.triu_indices(3, 1)
    return {
        "seed": seed,
        "analytic_correlation": analytic.tolist(),
        "empirical_correlation": empirical.tolist(),
        "maximum_absolute_correlation_error": float(
            np.max(np.abs(empirical[offdiag] - analytic[offdiag]))
        ),
        "maximum_marginal_mean_relative_error": float(
            np.max(np.abs(means - expected) / expected)
        ),
        "maximum_marginal_variance_relative_error": float(
            np.max(np.abs(variances - expected) / expected)
        ),
        "maximum_independent_control_absolute_correlation": float(
            np.max(np.abs(independent[offdiag]))
        ),
        "repeat_exact": bool(np.array_equal(counts, repeat)),
        "partition_exact": partition_exact,
        "density_minimum": float(np.min(density)),
        "transmittance_minimum": float(np.min(transmittance)),
        "transmittance_maximum": float(np.max(transmittance)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "maximum_absolute_correlation_error",
        "maximum_marginal_mean_relative_error",
        "maximum_marginal_variance_relative_error",
        "maximum_independent_control_absolute_correlation",
    )
    return {
        "rows": list(rows),
        "worst_metrics": {key: max(float(row[key]) for row in rows) for key in keys},
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    field = contract["field"]
    development = _aggregate(
        [_row(int(seed), contract) for seed in field["development_seeds"]]
    )
    confirmation = _aggregate(
        [_row(int(seed), contract) for seed in field["confirmation_seeds"]]
    )
    all_rows = (*development["rows"], *confirmation["rows"])
    worst = confirmation["worst_metrics"]
    metrics = contract["metrics"]
    gates = {
        "correlation": worst["maximum_absolute_correlation_error"]
        <= metrics["maximum_absolute_correlation_error"],
        "marginal_mean": worst["maximum_marginal_mean_relative_error"]
        <= metrics["maximum_marginal_mean_relative_error"],
        "marginal_variance": worst["maximum_marginal_variance_relative_error"]
        <= metrics["maximum_marginal_variance_relative_error"],
        "independent_control": worst["maximum_independent_control_absolute_correlation"]
        <= metrics["maximum_independent_control_absolute_correlation"],
        "repeat_exact": all(bool(row["repeat_exact"]) for row in all_rows),
        "partition_exact": all(bool(row["partition_exact"]) for row in all_rows),
        "physical_domain": all(
            row["density_minimum"] >= 0.0
            and row["transmittance_minimum"] > 0.0
            and row["transmittance_maximum"] <= 1.0
            for row in all_rows
        ),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "development_worst_metrics": development["worst_metrics"],
        "confirmation_worst_metrics": worst,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "development": development,
        "confirmation": confirmation,
    }


__all__ = ["CrossLayerPoissonError", "evaluate", "load_contract"]
