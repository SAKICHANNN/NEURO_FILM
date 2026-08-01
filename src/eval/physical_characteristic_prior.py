"""U6.P2Q compiler and evaluator for the Kodak 250D graph prior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.physical_characteristic_source import load_trace
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.manufacturer_characteristic import (
    CHANNELS,
    ManufacturerCharacteristicCurve,
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro_film.u6_p2q_kodak_250d_characteristic_prior_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p2q_kodak_250d_characteristic_prior_report.v1"


class CharacteristicPriorError(RuntimeError):
    """Raised when the frozen P2Q compiler contract or parent evidence drifts."""


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


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CharacteristicPriorError("P2Q paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parents = payload.get("parents", {})
    compiler = payload.get("compiler", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parents.get("p2p_decision", {}).get("sha256")
        != "3a176daa540a79e13a36a24192440325423dbeecf758d12d5ddebe5da2d2b1b6"
        or parents.get("trace", {}).get("sha256")
        != "0157c8235160a47c231f091aa8fd42dae8a50a8af3c949cb17424a55b45394a9"
        or parents.get("generic_u2_2", {}).get("sha256")
        != "8f43edcbc7e168af830081dce16ecb5f268a294d2f0cd777b942aec4b95c8af2"
        or compiler
        != {
            "channel_order": ["red", "green", "blue"],
            "input_domain": "relative_layer_log_exposure",
            "output_domain": "status_m_density",
            "monotonic_projection": "cumulative_max_at_source_knots",
            "interpolation": "piecewise_linear",
            "outside_observed_domain": "reject",
            "floating_point": "float64",
            "parameter_fit": False,
        }
        or evaluation.get("dense_samples_per_channel") != 4097
        or evaluation.get("normalized_shape_samples") != 1025
        or evaluation.get("partition_rows") != [1, 7, 31, 127]
        or gates.get("source_knot_density_max_abs_error") != 0.007
        or gates.get("generic_normalized_shape_rmse_each_channel_min") != 0.05
        or gates.get("generic_normalized_shape_rmse_median_min") != 0.05
        or not gates.get("two_byte_identical_audits")
    ):
        raise CharacteristicPriorError("P2Q frozen contract drift")
    for record in parents.values():
        _relative_path(str(record.get("path", "")))
    return payload


def _value_from_pixel(pixel: float, anchors: list[list[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    return float(value0 + ((pixel - pixel0) / (pixel1 - pixel0)) * (value1 - value0))


def _load_parents(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    loaded = []
    for name in ("p2p_decision", "trace", "generic_u2_2"):
        record = config["parents"][name]
        path = root / _relative_path(str(record["path"]))
        if not path.is_file() or _hash_file(path) != record["sha256"]:
            raise CharacteristicPriorError(f"P2Q parent integrity mismatch: {name}")
        loaded.append(json.loads(path.read_text(encoding="utf-8")))
    decision, trace, generic = loaded
    if (
        decision.get("status") != config["parents"]["p2p_decision"]["required_status"]
        or decision.get("stable_evidence_id")
        != config["parents"]["p2p_decision"]["required_stable_evidence_id"]
    ):
        raise CharacteristicPriorError("P2Q parent activation mismatch")
    load_trace(root / _relative_path(str(config["parents"]["trace"]["path"])))
    return decision, trace, generic


def compile_prior(
    trace: Mapping[str, Any], *, source_evidence_id: str
) -> tuple[ManufacturerCharacteristicPrior, dict[str, float]]:
    axes = trace["graph_axes"]
    errors: dict[str, float] = {}
    curves = []
    for channel in CHANNELS:
        coordinates = np.asarray(trace["curves"][channel], dtype=np.float64)
        exposure = np.asarray(
            [_value_from_pixel(value, axes["x_value_pixels"]) for value in coordinates[:, 0]],
            dtype=np.float64,
        )
        raw_density = np.asarray(
            [_value_from_pixel(value, axes["y_value_pixels"]) for value in coordinates[:, 1]],
            dtype=np.float64,
        )
        density = np.maximum.accumulate(raw_density)
        errors[channel] = float(np.max(np.abs(density - raw_density)))
        curves.append(ManufacturerCharacteristicCurve(channel, exposure, density))
    return (
        ManufacturerCharacteristicPrior(
            tuple(curves),  # type: ignore[arg-type]
            source_evidence_id,
            {
                "exposure": "5500 K daylight",
                "process": "ECN-2",
                "densitometry": "Status M",
            },
        ),
        errors,
    )


def _normalized_shape(values: np.ndarray) -> np.ndarray:
    span = float(values[-1] - values[0])
    if span <= 0.0:
        raise CharacteristicPriorError("cannot normalize a constant characteristic curve")
    return (values - values[0]) / span


def evaluate_prior(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision, trace, generic_config = _load_parents(config, root)
    prior, source_errors = compile_prior(
        trace, source_evidence_id=str(decision["stable_evidence_id"])
    )
    replay = ManufacturerCharacteristicPrior.from_dict(
        json.loads(json.dumps(prior.to_dict(), sort_keys=True))
    )
    dense_count = int(config["evaluation"]["dense_samples_per_channel"])
    dense_exposure: dict[str, np.ndarray] = {}
    dense_density: dict[str, np.ndarray] = {}
    dense_min_step: dict[str, float] = {}
    endpoint_error: dict[str, float] = {}
    bound_error: dict[str, float] = {}
    replay_error: dict[str, float] = {}
    for curve, replay_curve in zip(prior.curves, replay.curves):
        lower, upper = curve.domain
        exposure = np.linspace(lower, upper, dense_count, dtype=np.float64)
        density = curve.apply(exposure)
        replay_density = replay_curve.apply(exposure)
        density_lower, density_upper = curve.density_bounds
        dense_exposure[curve.layer] = exposure
        dense_density[curve.layer] = density
        dense_min_step[curve.layer] = float(np.min(np.diff(density)))
        endpoint_error[curve.layer] = float(
            max(abs(density[0] - density_lower), abs(density[-1] - density_upper))
        )
        bound_error[curve.layer] = float(
            max(0.0, density_lower - float(np.min(density)), float(np.max(density)) - density_upper)
        )
        replay_error[curve.layer] = float(np.max(np.abs(density - replay_density)))

    common_t = np.linspace(0.0, 1.0, dense_count, dtype=np.float64)
    matrix_exposure = np.column_stack(
        [lower + common_t * (upper - lower) for lower, upper in (curve.domain for curve in prior.curves)]
    )
    full = prior.apply(matrix_exposure)
    partition_errors: dict[str, float] = {}
    for rows in config["evaluation"]["partition_rows"]:
        pieces = [
            prior.apply(matrix_exposure[start : start + int(rows)])
            for start in range(0, len(matrix_exposure), int(rows))
        ]
        joined = np.concatenate(pieces, axis=0)
        partition_errors[str(rows)] = float(np.max(np.abs(joined - full)))

    domain_guards = {}
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        low = matrix_exposure[[0]].copy()
        low[0, index] = np.nextafter(lower, -np.inf)
        high = matrix_exposure[[-1]].copy()
        high[0, index] = np.nextafter(upper, np.inf)
        for label, value in (("below", low), ("above", high)):
            try:
                prior.apply(value)
                domain_guards[f"{curve.layer}_{label}"] = False
            except ValueError:
                domain_guards[f"{curve.layer}_{label}"] = True

    generic = build_operator(generic_config)
    shape_count = int(config["evaluation"]["normalized_shape_samples"])
    shape_t = np.linspace(0.0, 1.0, shape_count, dtype=np.float64)
    generic_by_layer = {curve.layer: curve for curve in generic.curves}
    shape_comparison: dict[str, dict[str, float]] = {}
    for curve in prior.curves:
        lower, upper = curve.domain
        manufacturer = _normalized_shape(curve.apply(lower + shape_t * (upper - lower)))
        generic_curve = generic_by_layer[curve.layer]
        generic_x = np.asarray(generic_config["curve_x_knots"], dtype=np.float64)
        generic_shape = _normalized_shape(
            generic_curve.apply(generic_x[0] + shape_t * (generic_x[-1] - generic_x[0]))
        )
        residual = manufacturer - generic_shape
        shape_comparison[curve.layer] = {
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "maximum_abs": float(np.max(np.abs(residual))),
            "mean_signed": float(np.mean(residual)),
        }

    gates = config["gates"]
    shape_rmse = [row["rmse"] for row in shape_comparison.values()]
    checks = {
        "source_knot_fidelity": max(source_errors.values())
        <= float(gates["source_knot_density_max_abs_error"]),
        "dense_monotonic": min(dense_min_step.values())
        >= float(gates["dense_monotonic_min_step"]),
        "endpoints_exact": max(endpoint_error.values())
        <= float(gates["endpoint_max_abs_error"]),
        "bounded": max(bound_error.values()) <= float(gates["output_bound_tolerance"]),
        "serialization_replay": max(replay_error.values())
        <= float(gates["replay_max_abs_error"]),
        "partition_exact": max(partition_errors.values())
        <= float(gates["partition_max_abs_error"]),
        "domain_guards": all(domain_guards.values()),
        "generic_shape_distinct_each_channel": min(shape_rmse)
        >= float(gates["generic_normalized_shape_rmse_each_channel_min"]),
        "generic_shape_distinct_median": float(np.median(shape_rmse))
        >= float(gates["generic_normalized_shape_rmse_median_min"]),
        "no_parameter_fit": config["compiler"]["parameter_fit"] is False,
    }
    passed = all(checks.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_stable_evidence_id": decision["stable_evidence_id"],
        "prior_identity": prior.identity(),
        "prior": prior.to_dict(),
        "source_knot_density_max_abs_error": source_errors,
        "dense_monotonic_min_step": dense_min_step,
        "endpoint_max_abs_error": endpoint_error,
        "output_bound_error": bound_error,
        "serialization_replay_max_abs_error": replay_error,
        "partition_max_abs_error": partition_errors,
        "domain_guards": domain_guards,
        "generic_u2_2_normalized_shape": shape_comparison,
        "gate_results": checks,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "passed": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "retain_research_manufacturer_characteristic_prior" if passed else "close_characteristic_prior",
        "claim_ceiling": config["claim_ceiling"],
    }
    arrays = {
        "matrix_exposure": matrix_exposure,
        "matrix_density": full,
        **{f"{name}_dense_exposure": value for name, value in dense_exposure.items()},
        **{f"{name}_dense_density": value for name, value in dense_density.items()},
    }
    fingerprints = {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name, value in sorted(arrays.items())
    }
    return report, {"prior": prior.to_dict(), "array_sha256": fingerprints}


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "CharacteristicPriorError",
    "compile_prior",
    "evaluate_prior",
    "load_contract",
]
