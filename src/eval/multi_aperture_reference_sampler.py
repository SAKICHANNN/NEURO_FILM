"""U6.P4BH deterministic multi-aperture sampler evaluator."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.aperture_cell_sampler import (
    counter_aperture_scaled_poisson_region,
    sample_aperture_scaled_density_region,
)
from src.film_physics.aperture_scaled_granularity import (
    ApertureScaledCompoundPoissonProfile,
)

SCHEMA = "neuro_film.u6_p4bh_multi_aperture_reference_sampler_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bh_multi_aperture_reference_sampler_report.v1"


class MultiApertureReferenceSamplerError(RuntimeError):
    """Raised when frozen P4BH evidence or semantics drift."""


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
        raise MultiApertureReferenceSamplerError("P4BH paths must be relative")
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
        or sampler.get("seed_base") != 2608021100
        or sampler.get("seed_stride") != 104729
        or sampler.get("apertures_micrometres")
        != [7.25, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
        or sampler.get("maximum_supported_rate") != 16_000_000.0
        or sampler.get("independent_cells") is not True
        or sampler.get("independence_status")
        != "generic_hypothesis_not_measured_nps"
        or sampler.get("realized_field_renormalization_allowed") is not False
        or sampler.get("photographic_render_allowed") is not False
        or evaluation.get("required_probe_count") != 15
        or evaluation.get("required_scaled_row_count") != 105
        or evaluation.get("maximum_density_mean_relative_error") != 0.005
        or evaluation.get("maximum_density_rms_relative_error") != 0.025
        or evaluation.get("maximum_transmittance_mean_relative_error") != 0.005
        or evaluation.get("maximum_transmittance_rms_relative_error") != 0.025
        or evaluation.get("maximum_absolute_lag_one_correlation") != 0.04
        or evaluation.get("maximum_per_probe_normalized_rms_span") != 0.06
        or evaluation.get("require_reference_aperture_exact_p4bc_counts") is not True
        or evaluation.get("require_repeat_exact") is not True
        or evaluation.get("require_odd_partition_exact") is not True
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise MultiApertureReferenceSamplerError("P4BH frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise MultiApertureReferenceSamplerError(
            f"P4BH parent integrity mismatch: {stem}"
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
    row_cuts = (0, min(73, full_shape[0] - 1), full_shape[0])
    column_cuts = (0, min(61, full_shape[1] - 1), full_shape[1])
    rows: list[np.ndarray] = []
    for y0, y1 in pairwise(row_cuts):
        blocks = [
            counter_aperture_scaled_poisson_region(
                full_shape,
                origin_yx=(y0, x0),
                shape=(y1 - y0, x1 - x0),
                rate=rate,
                seed=seed,
            )
            for x0, x1 in pairwise(column_cuts)
        ]
        rows.append(np.concatenate(blocks, axis=1))
    return np.concatenate(rows, axis=0)


def evaluate_sampler(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    p4bc_decision = _load_parent(parents, root, "p4bc_decision")
    p4bc_report = _load_parent(parents, root, "p4bc_report")
    p4bg_decision = _load_parent(parents, root, "p4bg_decision")
    p4bg_bundle = _load_parent(parents, root, "p4bg_bundle")
    p4bg_report = _load_parent(parents, root, "p4bg_report")
    if (
        p4bc_decision.get("automatic_pass") is not True
        or p4bc_report.get("stable_evidence_id")
        != parents["p4bc_stable_evidence_id"]
        or p4bg_decision.get("automatic_pass") is not True
        or p4bg_report.get("stable_evidence_id")
        != parents["p4bg_stable_evidence_id"]
        or p4bg_report.get("automatic_pass") is not True
    ):
        raise MultiApertureReferenceSamplerError("P4BH parent decision mismatch")
    serialized = dict(p4bg_bundle)
    profile_id = serialized.pop("profile_id")
    profile = ApertureScaledCompoundPoissonProfile.from_dict(serialized)
    if profile.identity() != profile_id:
        raise MultiApertureReferenceSamplerError("P4BH parent profile mismatch")

    sampler = contract["sampler"]
    shape = tuple(int(value) for value in sampler["field_shape_cells"])
    seed_base = int(sampler["seed_base"])
    seed_stride = int(sampler["seed_stride"])
    p4bc_by_probe = {
        (str(row["channel"]), float(row["domain_fraction"])): row
        for row in p4bc_report["rows"]
    }
    probe_order: dict[tuple[str, float], int] = {}
    normalized_rms_by_probe: defaultdict[tuple[str, float], list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    mean_errors: list[float] = []
    rms_errors: list[float] = []
    trans_mean_errors: list[float] = []
    trans_rms_errors: list[float] = []
    correlations: list[float] = []
    repeat_results: list[bool] = []
    partition_results: list[bool] = []
    reference_results: list[bool] = []
    for parent in p4bg_report["rows"]:
        key = (str(parent["channel"]), float(parent["domain_fraction"]))
        if key not in probe_order:
            probe_order[key] = len(probe_order)
        seed = seed_base + probe_order[key] * seed_stride
        rate = float(parent["poisson_rate"])
        mark = float(parent["density_mark"])
        aperture = float(parent["aperture_micrometres"])
        counts, density = sample_aperture_scaled_density_region(
            shape,
            origin_yx=(0, 0),
            shape=shape,
            rate=rate,
            density_mark=mark,
            seed=seed,
        )
        identity_shape = (17, 19)
        repeat = counter_aperture_scaled_poisson_region(
            identity_shape,
            origin_yx=(0, 0),
            shape=identity_shape,
            rate=rate,
            seed=seed,
        )
        repeat_again = counter_aperture_scaled_poisson_region(
            identity_shape,
            origin_yx=(0, 0),
            shape=identity_shape,
            rate=rate,
            seed=seed,
        )
        partition = _partition_counts(identity_shape, rate, seed)
        count_sha = hashlib.sha256(
            np.asarray(counts, dtype="<u4").tobytes(order="C")
        ).hexdigest()
        repeat_exact = np.array_equal(repeat, repeat_again)
        partition_exact = np.array_equal(repeat, partition)
        reference_exact = aperture != 48.0 or (
            count_sha == p4bc_by_probe[key]["count_sha256"]
        )
        transmittance = np.power(10.0, -density)
        density_mean = float(np.mean(density, dtype=np.float64))
        density_rms = float(np.std(density, dtype=np.float64))
        transmittance_mean = float(np.mean(transmittance, dtype=np.float64))
        transmittance_rms = float(np.std(transmittance, dtype=np.float64))
        mean_error = _relative_error(density_mean, float(parent["density_mean"]))
        rms_error = _relative_error(density_rms, float(parent["density_rms"]))
        trans_mean_error = _relative_error(
            transmittance_mean, float(parent["transmittance_mean"])
        )
        trans_rms_error = _relative_error(
            transmittance_rms, float(parent["transmittance_rms"])
        )
        horizontal, vertical = _lag_one_correlations(density)
        normalized_rms_by_probe[key].append(density_rms * aperture / 48.0)
        mean_errors.append(mean_error)
        rms_errors.append(rms_error)
        trans_mean_errors.append(trans_mean_error)
        trans_rms_errors.append(trans_rms_error)
        correlations.extend((abs(horizontal), abs(vertical)))
        repeat_results.append(repeat_exact)
        partition_results.append(partition_exact)
        reference_results.append(reference_exact)
        rows.append(
            {
                "channel": key[0],
                "domain_fraction": key[1],
                "aperture_micrometres": aperture,
                "seed": seed,
                "count_sha256": count_sha,
                "density_mean_relative_error": mean_error,
                "density_rms_relative_error": rms_error,
                "transmittance_mean_relative_error": trans_mean_error,
                "transmittance_rms_relative_error": trans_rms_error,
                "horizontal_lag_one_correlation": horizontal,
                "vertical_lag_one_correlation": vertical,
                "repeat_exact": repeat_exact,
                "odd_partition_exact": partition_exact,
                "reference_aperture_exact_p4bc_counts": reference_exact,
            }
        )

    normalized_spans = []
    for values in normalized_rms_by_probe.values():
        array = np.asarray(values, dtype=np.float64)
        normalized_spans.append(float((np.max(array) - np.min(array)) / np.median(array)))
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "probe_count": len(probe_order) == int(gates["required_probe_count"]),
        "scaled_row_count": len(rows) == int(gates["required_scaled_row_count"]),
        "density_mean": max(mean_errors)
        <= float(gates["maximum_density_mean_relative_error"]),
        "density_rms": max(rms_errors)
        <= float(gates["maximum_density_rms_relative_error"]),
        "transmittance_mean": max(trans_mean_errors)
        <= float(gates["maximum_transmittance_mean_relative_error"]),
        "transmittance_rms": max(trans_rms_errors)
        <= float(gates["maximum_transmittance_rms_relative_error"]),
        "lag_one_independence": max(correlations)
        <= float(gates["maximum_absolute_lag_one_correlation"]),
        "normalized_rms_span": max(normalized_spans)
        <= float(gates["maximum_per_probe_normalized_rms_span"]),
        "reference_aperture_exact_p4bc_counts": all(reference_results),
        "repeat_exact": all(repeat_results),
        "odd_partition_exact": all(partition_results),
        "generic_hypothesis_label": sampler["independence_status"]
        == "generic_hypothesis_not_measured_nps",
        "realized_field_renormalization_forbidden": sampler[
            "realized_field_renormalization_allowed"
        ]
        is False,
        "photographic_render_forbidden": sampler["photographic_render_allowed"]
        is False,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "field_shape_cells": list(shape),
        "probe_count": len(probe_order),
        "scaled_row_count": len(rows),
        "maximum_density_mean_relative_error": max(mean_errors),
        "maximum_density_rms_relative_error": max(rms_errors),
        "maximum_transmittance_mean_relative_error": max(trans_mean_errors),
        "maximum_transmittance_rms_relative_error": max(trans_rms_errors),
        "maximum_absolute_lag_one_correlation": max(correlations),
        "maximum_per_probe_normalized_rms_span": max(normalized_spans),
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_generic_multi_aperture_reference_sampler"
            if automatic_pass
            else "retain_analytic_scaling_and_48um_sampler_only"
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
    "MultiApertureReferenceSamplerError",
    "evaluate_sampler",
    "load_contract",
    "write_report",
]
