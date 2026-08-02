"""U6.P4BC deterministic aperture-cell reference-sampler evaluator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.aperture_cell_sampler import (
    counter_high_rate_poisson_region,
    sample_aperture_cell_density_region,
)
from src.film_physics.transmittance_granularity import (
    ApertureCellCompoundPoissonProfile,
)

SCHEMA = "neuro_film.u6_p4bc_aperture_cell_reference_sampler_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bc_aperture_cell_reference_sampler_report.v1"


class ApertureCellReferenceSamplerError(RuntimeError):
    """Raised when frozen P4BC evidence or semantics drift."""


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
        raise ApertureCellReferenceSamplerError("P4BC paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sampler = payload.get("sampler", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or sampler.get("coordinate_uniform") != "splitmix64-open-unit-interval"
        or sampler.get("poisson_inverse")
        != "ceil(scipy.special.pdtrik(uniform, rate))"
        or sampler.get("count_dtype") != "uint32"
        or sampler.get("field_shape_cells") != [192, 193]
        or sampler.get("measurement_cell_micrometres") != 48.0
        or sampler.get("independent_cells") is not True
        or sampler.get("independence_status")
        != "generic_hypothesis_not_measured_nps"
        or sampler.get("spatial_filtering_allowed") is not False
        or sampler.get("photographic_render_allowed") is not False
        or evaluation.get("minimum_probe_count") != 15
        or evaluation.get("maximum_density_mean_relative_error") != 0.0006
        or evaluation.get("maximum_density_rms_relative_error") != 0.02
        or evaluation.get("maximum_transmittance_mean_relative_error") != 0.0006
        or evaluation.get("maximum_transmittance_rms_relative_error") != 0.02
        or evaluation.get("maximum_absolute_lag_one_correlation") != 0.025
        or evaluation.get("require_repeat_exact") is not True
        or evaluation.get("require_odd_partition_exact") is not True
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise ApertureCellReferenceSamplerError("P4BC frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise ApertureCellReferenceSamplerError(
            f"P4BC parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual / expected - 1.0)


def _lag_one_correlations(values: np.ndarray) -> tuple[float, float]:
    horizontal = float(
        np.corrcoef(values[:, :-1].reshape(-1), values[:, 1:].reshape(-1))[0, 1]
    )
    vertical = float(
        np.corrcoef(values[:-1, :].reshape(-1), values[1:, :].reshape(-1))[0, 1]
    )
    return horizontal, vertical


def _partition_counts(
    full_shape: tuple[int, int], rate: float, seed: int
) -> np.ndarray:
    row_cuts = (0, 73, full_shape[0])
    column_cuts = (0, 61, full_shape[1])
    row_blocks: list[np.ndarray] = []
    for y0, y1 in pairwise(row_cuts):
        blocks = [
            counter_high_rate_poisson_region(
                full_shape,
                origin_yx=(y0, x0),
                shape=(y1 - y0, x1 - x0),
                rate=rate,
                seed=seed,
            )
            for x0, x1 in pairwise(column_cuts)
        ]
        row_blocks.append(np.concatenate(blocks, axis=1))
    return np.concatenate(row_blocks, axis=0)


def evaluate_sampler(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4bb_decision")
    parent_bundle = _load_parent(parents, root, "p4bb_bundle")
    parent_report = _load_parent(parents, root, "p4bb_report")
    if (
        decision.get("decision")
        != "retain_aperture_cell_compound_poisson_amplitude_compiler"
        or decision.get("automatic_pass") is not True
        or parent_report.get("stable_evidence_id") != parents["p4bb_stable_evidence_id"]
        or parent_report.get("probe_count") != 15
    ):
        raise ApertureCellReferenceSamplerError("P4BC parent decision mismatch")
    serialized = dict(parent_bundle)
    profile_id = serialized.pop("profile_id")
    profile = ApertureCellCompoundPoissonProfile.from_dict(serialized)
    if profile.identity() != profile_id:
        raise ApertureCellReferenceSamplerError("P4BC parent profile mismatch")

    sampler = contract["sampler"]
    shape = tuple(int(value) for value in sampler["field_shape_cells"])
    seed_base = int(sampler["seed_base"])
    seed_stride = int(sampler["seed_stride"])
    rows: list[dict[str, Any]] = []
    repeat_results: list[bool] = []
    partition_results: list[bool] = []
    density_mean_errors: list[float] = []
    density_rms_errors: list[float] = []
    transmittance_mean_errors: list[float] = []
    transmittance_rms_errors: list[float] = []
    correlations: list[float] = []
    for index, parent in enumerate(parent_report["rows"]):
        rate = float(parent["poisson_rate_per_48um_cell"])
        mark = float(parent["density_mark"])
        seed = seed_base + index * seed_stride
        counts, density = sample_aperture_cell_density_region(
            shape,
            origin_yx=(0, 0),
            shape=shape,
            rate=rate,
            density_mark=mark,
            seed=seed,
        )
        repeat = counter_high_rate_poisson_region(
            shape, origin_yx=(0, 0), shape=shape, rate=rate, seed=seed
        )
        partition = _partition_counts(shape, rate, seed)
        repeat_exact = np.array_equal(counts, repeat)
        partition_exact = np.array_equal(counts, partition)
        transmittance = np.power(10.0, -density)
        density_mean = float(np.mean(density, dtype=np.float64))
        density_rms = float(np.std(density, dtype=np.float64))
        transmittance_mean = float(np.mean(transmittance, dtype=np.float64))
        transmittance_rms = float(np.std(transmittance, dtype=np.float64))
        density_mean_error = _relative_error(density_mean, float(parent["density_mean"]))
        density_rms_error = _relative_error(density_rms, float(parent["density_rms"]))
        transmittance_mean_error = _relative_error(
            transmittance_mean, float(parent["transmittance_mean"])
        )
        transmittance_rms_error = _relative_error(
            transmittance_rms, float(parent["transmittance_rms"])
        )
        horizontal, vertical = _lag_one_correlations(density)
        repeat_results.append(repeat_exact)
        partition_results.append(partition_exact)
        density_mean_errors.append(density_mean_error)
        density_rms_errors.append(density_rms_error)
        transmittance_mean_errors.append(transmittance_mean_error)
        transmittance_rms_errors.append(transmittance_rms_error)
        correlations.extend((abs(horizontal), abs(vertical)))
        rows.append(
            {
                "channel": parent["channel"],
                "domain_fraction": parent["domain_fraction"],
                "seed": seed,
                "count_sha256": hashlib.sha256(
                    np.asarray(counts, dtype="<u4").tobytes(order="C")
                ).hexdigest(),
                "density_mean_relative_error": density_mean_error,
                "density_rms_relative_error": density_rms_error,
                "transmittance_mean_relative_error": transmittance_mean_error,
                "transmittance_rms_relative_error": transmittance_rms_error,
                "horizontal_lag_one_correlation": horizontal,
                "vertical_lag_one_correlation": vertical,
                "repeat_exact": repeat_exact,
                "odd_partition_exact": partition_exact,
            }
        )

    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "probe_count": len(rows) >= int(gates["minimum_probe_count"]),
        "density_mean": max(density_mean_errors)
        <= float(gates["maximum_density_mean_relative_error"]),
        "density_rms": max(density_rms_errors)
        <= float(gates["maximum_density_rms_relative_error"]),
        "transmittance_mean": max(transmittance_mean_errors)
        <= float(gates["maximum_transmittance_mean_relative_error"]),
        "transmittance_rms": max(transmittance_rms_errors)
        <= float(gates["maximum_transmittance_rms_relative_error"]),
        "lag_one_independence": max(correlations)
        <= float(gates["maximum_absolute_lag_one_correlation"]),
        "repeat_exact": all(repeat_results),
        "odd_partition_exact": all(partition_results),
        "generic_hypothesis_label": sampler["independence_status"]
        == "generic_hypothesis_not_measured_nps",
        "spatial_filtering_forbidden": sampler["spatial_filtering_allowed"] is False,
        "photographic_render_forbidden": sampler["photographic_render_allowed"]
        is False,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "parent_profile_id": profile.identity(),
        "field_shape_cells": list(shape),
        "probe_count": len(rows),
        "maximum_density_mean_relative_error": max(density_mean_errors),
        "maximum_density_rms_relative_error": max(density_rms_errors),
        "maximum_transmittance_mean_relative_error": max(
            transmittance_mean_errors
        ),
        "maximum_transmittance_rms_relative_error": max(transmittance_rms_errors),
        "maximum_absolute_lag_one_correlation": max(correlations),
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_generic_aperture_cell_reference_sampler"
            if automatic_pass
            else "close_aperture_cell_sampler_without_rescue"
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
    "ApertureCellReferenceSamplerError",
    "evaluate_sampler",
    "load_contract",
    "write_report",
]
