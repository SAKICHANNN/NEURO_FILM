"""U6.P2R evaluation of the anchored Kodak manufacturer-shape hypothesis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.physical_characteristic_prior import compile_prior, load_contract as load_p2q
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.manufacturer_shape_gauge import (
    compile_anchored_shape_gauge,
    mapped_source_target,
)
from src.roll2film.sensitometry import RGBSensitometryOperator

SCHEMA = "neuro_film.u6_p2r_kodak_250d_anchored_shape_gauge_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p2r_kodak_250d_anchored_shape_gauge_report.v1"


class AnchoredShapeGaugeError(RuntimeError):
    """Raised when the frozen P2R contract or evidence drifts."""


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
        raise AnchoredShapeGaugeError("P2R paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parents = payload.get("parents", {})
    gauge = payload.get("gauge", {})
    evaluation = payload.get("evaluation", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parents.get("p2q_decision", {}).get("sha256")
        != "80aef83b5868ed4dd86e9ea1c91e2a1ae9a6e5355ac450448c9c556f26a9da88"
        or parents.get("p2q_contract", {}).get("sha256")
        != "3edbc61b132385500f7bb4acb93f58bb38b666e341f503e193dec10c8a6661f3"
        or parents.get("generic_u2_2", {}).get("sha256")
        != "8f43edcbc7e168af830081dce16ecb5f268a294d2f0cd777b942aec4b95c8af2"
        or gauge.get("source_anchor_density_rule")
        != "match_generic_anchor_fraction_between_each_channels_generic_endpoint_densities"
        or gauge.get("source_anchor_exposure_rule")
        != "left_continuous_piecewise_linear_inverse"
        or gauge.get("x_mapping")
        != "piecewise_affine_source_min_anchor_max_to_generic_min_zero_max"
        or gauge.get("y_mapping")
        != "piecewise_affine_source_min_anchor_max_to_generic_min_one_max"
        or gauge.get("insert_exact_shared_anchor") != [0.0, 1.0]
        or gauge.get("strictness_repair")
        != "nextafter_up_only_for_equal_projected_density_knots"
        or gauge.get("spline") != "existing_rational_quadratic_from_knots"
        or gauge.get("encoder") != "unchanged_generic_u2_2_log_exposure_encoder"
        or gauge.get("input_space") != "relative_layer_exposure_rgb_order"
        or gauge.get("output_space") != "status_m_layer_density"
        or gauge.get("parameter_fit") is not False
        or evaluation != {
            "linear_layer_exposure_max": 16.0,
            "samples": 16387,
            "seed": 2026080102,
            "shape_samples": 4097,
            "partition_rows": [1, 7, 31, 127],
        }
        or gates.get("source_normalized_shape_rmse_each_channel_max") != 0.01
        or gates.get("generic_density_difference_max_each_channel_min") != 0.05
        or gates.get("generic_density_difference_median_all_samples_min") != 0.02
        or not gates.get("two_byte_identical_audits")
    ):
        raise AnchoredShapeGaugeError("P2R frozen contract drift")
    for record in parents.values():
        _relative_path(str(record.get("path", "")))
    return payload


def _load_json_exact(root: Path, record: Mapping[str, Any], name: str) -> dict[str, Any]:
    path = root / _relative_path(str(record["path"]))
    if not path.is_file() or _hash_file(path) != record["sha256"]:
        raise AnchoredShapeGaugeError(f"P2R parent integrity mismatch: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gauge(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = _load_json_exact(root, config["parents"]["p2q_decision"], "p2q_decision")
    if (
        decision.get("status")
        != config["parents"]["p2q_decision"]["required_status"]
        or decision.get("prior_identity")
        != config["parents"]["p2q_decision"]["required_prior_identity"]
    ):
        raise AnchoredShapeGaugeError("P2R activation mismatch")
    p2q = _load_json_exact(root, config["parents"]["p2q_contract"], "p2q_contract")
    generic_config = _load_json_exact(
        root, config["parents"]["generic_u2_2"], "generic_u2_2"
    )
    load_p2q(root / _relative_path(str(config["parents"]["p2q_contract"]["path"])))
    trace_record = p2q["parents"]["trace"]
    trace = _load_json_exact(root, trace_record, "p2q_trace")
    manufacturer, _ = compile_prior(
        trace,
        source_evidence_id=str(
            p2q["parents"]["p2p_decision"]["required_stable_evidence_id"]
        ),
    )
    if manufacturer.identity() != decision["prior_identity"]:
        raise AnchoredShapeGaugeError("P2R manufacturer prior identity mismatch")
    generic = build_operator(generic_config)
    operator, metadata = compile_anchored_shape_gauge(
        manufacturer, generic, generic_config
    )
    replay = RGBSensitometryOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )

    shape_x = np.linspace(
        float(generic_config["curve_x_knots"][0]),
        float(generic_config["curve_x_knots"][-1]),
        int(config["evaluation"]["shape_samples"]),
        dtype=np.float64,
    )
    source_shape: dict[str, dict[str, float]] = {}
    derivatives: dict[str, float] = {}
    anchors: dict[str, float] = {}
    repair: dict[str, float] = {}
    for source_curve, compiled_curve, row in zip(
        manufacturer.curves, operator.curves, metadata
    ):
        target = mapped_source_target(source_curve, row, shape_x)
        actual = compiled_curve.apply(shape_x)
        residual = actual - target
        source_shape[compiled_curve.layer] = {
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "maximum_abs": float(np.max(np.abs(residual))),
        }
        derivatives[compiled_curve.layer] = float(
            np.min(compiled_curve.derivative(shape_x))
        )
        anchors[compiled_curve.layer] = abs(
            float(compiled_curve.apply(np.asarray(0.0))) - 1.0
        )
        repair[compiled_curve.layer] = row.strictness_repair_max_density

    rng = np.random.default_rng(int(config["evaluation"]["seed"]))
    exposure = rng.uniform(
        0.0,
        float(config["evaluation"]["linear_layer_exposure_max"]),
        size=(int(config["evaluation"]["samples"]), 3),
    )
    exposure = np.concatenate(
        [
            np.asarray([[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [16.0, 16.0, 16.0]]),
            exposure,
        ],
        axis=0,
    )
    density = operator.apply(exposure)
    restored = operator.inverse(density)
    replay_density = replay.apply(exposure)
    generic_density = generic.apply(exposure)
    difference = np.abs(density - generic_density)
    per_channel_max = {
        layer: float(np.max(difference[:, index]))
        for index, layer in enumerate(("red", "green", "blue"))
    }
    endpoint_bounds = np.asarray(
        [row.generic_y_bounds for row in metadata], dtype=np.float64
    )
    bound_error = float(
        max(
            0.0,
            float(np.max(endpoint_bounds[:, 0] - np.min(density, axis=0))),
            float(np.max(np.max(density, axis=0) - endpoint_bounds[:, 1])),
        )
    )
    partitions: dict[str, float] = {}
    for rows in config["evaluation"]["partition_rows"]:
        joined = np.concatenate(
            [operator.apply(exposure[start : start + int(rows)]) for start in range(0, len(exposure), int(rows))],
            axis=0,
        )
        partitions[str(rows)] = float(np.max(np.abs(joined - density)))

    roundtrip = float(np.max(np.abs(restored - exposure)))
    replay_error = float(np.max(np.abs(replay_density - density)))
    jacobian = operator.jacobian_determinant(exposure)
    gates = config["gates"]
    checks = {
        "source_shape_fidelity": max(row["rmse"] for row in source_shape.values())
        <= float(gates["source_normalized_shape_rmse_each_channel_max"]),
        "source_anchor_mapping": max(anchors.values())
        <= float(gates["source_anchor_mapping_error_max"]),
        "strictness_repair": max(repair.values())
        <= float(gates["strictness_repair_max_density"]),
        "linear_roundtrip": roundtrip <= float(gates["linear_roundtrip_max_abs"]),
        "shared_anchor": max(anchors.values()) <= float(gates["anchor_density_error_max"]),
        "jacobian": float(np.min(jacobian))
        > float(gates["jacobian_determinant_min_exclusive"]),
        "curve_derivative": min(derivatives.values())
        > float(gates["curve_derivative_min_exclusive"]),
        "partition_exact": max(partitions.values())
        <= float(gates["partition_max_abs_error"]),
        "serialization_replay": replay_error
        <= float(gates["serialization_replay_max_abs_error"]),
        "generic_difference_each_channel": min(per_channel_max.values())
        >= float(gates["generic_density_difference_max_each_channel_min"]),
        "generic_difference_population": float(np.median(difference))
        >= float(gates["generic_density_difference_median_all_samples_min"]),
        "finite_and_bounded": bool(np.all(np.isfinite(density)) and bound_error == 0.0),
        "explicit_layer_exposure_semantics": operator.input_space
        == "relative_layer_exposure_rgb_order",
        "no_parameter_fit": config["gauge"]["parameter_fit"] is False,
    }
    passed = all(checks.values())
    stable = {
        "experiment_id": config["experiment_id"],
        "parent_prior_identity": manufacturer.identity(),
        "operator_identity": hashlib.sha256(
            json.dumps(
                operator.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
        "operator": operator.to_dict(),
        "gauge_channels": [row.to_dict() for row in metadata],
        "source_shape_error": source_shape,
        "source_anchor_mapping_error": anchors,
        "strictness_repair_max_density": repair,
        "curve_derivative_min": derivatives,
        "linear_roundtrip_max_abs": roundtrip,
        "jacobian_determinant_min": float(np.min(jacobian)),
        "serialization_replay_max_abs": replay_error,
        "partition_max_abs_error": partitions,
        "generic_density_difference_max": per_channel_max,
        "generic_density_difference_median_all_samples": float(np.median(difference)),
        "output_bound_error": bound_error,
        "gate_results": checks,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "passed": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "open_u6_p2s_photographic_challenger" if passed else "close_anchored_shape_gauge",
        "claim_ceiling": config["claim_ceiling"],
    }
    bundle = {
        "operator": operator.to_dict(),
        "operator_identity": stable["operator_identity"],
        "gauge_channels": stable["gauge_channels"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, bundle


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "AnchoredShapeGaugeError",
    "evaluate_gauge",
    "load_contract",
]
