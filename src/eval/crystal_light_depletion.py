"""U6.P4CC finite-crystal light-depletion mechanism evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.crystal_light_depletion import (
    CrystalLightDepletionProfile,
    render_crystal_light_depletion,
    render_independent_footprint_sum,
)

SCHEMA = "neuro_film.u6_p4cc_crystal_light_depletion_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cc_crystal_light_depletion_report.v1"


class CrystalLightDepletionEvaluationError(RuntimeError):
    """Raised when the frozen P4CC contract or parent evidence drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    mechanism = contract.get("mechanism", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or mechanism.get("field_shape") != [161, 167]
        or mechanism.get("layer_count_per_channel") != 12
        or mechanism.get("cell_pitch_pixels") != 8
        or mechanism.get("crystal_activation_probability") != 0.85
        or mechanism.get("capture_efficiency_per_layer") != 0.32
        or mechanism.get("global_mean_renormalization_allowed") is not False
        or mechanism.get("pointwise_output_clipping_allowed") is not False
        or evaluation.get("maximum_energy_conservation_absolute_error") != 2e-15
        or evaluation.get("minimum_constant_total_capture_fraction") != 0.7
        or evaluation.get("maximum_last_to_first_layer_capture_ratio") != 0.5
        or evaluation.get(
            "maximum_checker_first_layer_amplitude_ratio_vs_point_control"
        )
        != 0.5
        or evaluation.get("minimum_no_depletion_energy_excess_fraction") != 0.1
        or evaluation.get("minimum_forward_reverse_capture_rmse") != 0.0001
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise CrystalLightDepletionEvaluationError("P4CC frozen contract drift")


def _load_parent(contract: Mapping[str, Any], root: Path) -> None:
    binding = contract["parent"]
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise CrystalLightDepletionEvaluationError("P4CC parent identity mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != binding["required_decision"]:
        raise CrystalLightDepletionEvaluationError("P4CC parent decision mismatch")


def _profile(contract: Mapping[str, Any]) -> CrystalLightDepletionProfile:
    value = contract["mechanism"]
    return CrystalLightDepletionProfile(
        profile_id=str(value["profile_id"]),
        layer_count_per_channel=int(value["layer_count_per_channel"]),
        cell_pitch_pixels=int(value["cell_pitch_pixels"]),
        activation_probability=float(value["crystal_activation_probability"]),
        radius_min_pixels=float(value["crystal_radius_min_pixels"]),
        radius_max_pixels=float(value["crystal_radius_max_pixels"]),
        centre_jitter_fraction=float(value["centre_jitter_fraction_of_pitch"]),
        capture_efficiency=float(value["capture_efficiency_per_layer"]),
        channel_seed_bases=tuple(int(v) for v in value["channel_seed_bases"]),
        layer_seed_stride=int(value["layer_seed_stride"]),
    )


def _fixtures(contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    height, width = (int(v) for v in contract["mechanism"]["field_shape"])
    xs = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
    spec = contract["fixtures"]
    values: dict[str, np.ndarray] = {}
    for value in spec["constant_exposures"]:
        values[f"constant_{value:.2f}"] = np.full(
            (height, width), value, dtype=np.float64
        )
    lo, hi = (float(v) for v in spec["ramp_range"])
    values["ramp"] = lo + (hi - lo) * np.broadcast_to(xs, (height, width))
    checker_lo, checker_hi = (float(v) for v in spec["checker_values"])
    period = int(spec["checker_period_pixels"])
    checker = (
        (np.arange(height)[:, None] // period + np.arange(width)[None, :] // period) % 2
    ) == 0
    values["checker"] = np.where(checker, checker_hi, checker_lo).astype(np.float64)
    edge_lo, edge_hi = (float(v) for v in spec["edge_values"])
    values["edge"] = np.where(
        np.broadcast_to(xs, (height, width)) < 0.5, edge_lo, edge_hi
    )
    return values


def _typed(values: np.ndarray, pitch_um: float) -> PhysicalDomainArray:
    rgb = np.repeat(values[..., None], 3, axis=2)
    return PhysicalDomainArray(
        rgb,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(pitch_um),
    )


def _array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def _checker_amplitude(values: np.ndarray, period: int) -> float:
    height, width = values.shape[:2]
    sign = np.where(
        (
            (np.arange(height)[:, None] // period + np.arange(width)[None, :] // period)
            % 2
        )
        == 0,
        1.0,
        -1.0,
    )
    return float(abs(np.mean(values * sign[..., None], dtype=np.float64)))


def evaluate_crystal_light_depletion(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    _load_parent(contract, root)
    profile = _profile(contract)
    evaluation = contract["evaluation"]
    period = int(contract["fixtures"]["checker_period_pixels"])
    rows: list[dict[str, Any]] = []
    conservation_errors: list[float] = []
    constant_capture_fractions: list[float] = []
    last_first_ratios: list[float] = []
    order_rmses: list[float] = []
    no_depletion_excess_fractions: list[float] = []
    roundtrip_errors: list[float] = []
    exact_repeats: list[bool] = []
    canonical_geometry_ids: tuple[tuple[str, str, str], ...] | None = None
    geometry_repeat_exact = True
    checker_ratio = float("nan")
    domain_violation_count = 0
    finite = True
    for name, scalar in _fixtures(contract).items():
        exposure = _typed(scalar, float(contract["mechanism"]["pixel_pitch_um"]))
        candidate = render_crystal_light_depletion(exposure, profile)
        repeated = render_crystal_light_depletion(exposure, profile)
        reverse = render_crystal_light_depletion(exposure, profile, reverse_layers=True)
        point = render_crystal_light_depletion(exposure, profile, point_capture=True)
        independent = render_independent_footprint_sum(exposure, profile)
        captured = candidate.captured_exposure.values
        remaining = candidate.remaining_exposure.values
        conservation = float(
            np.max(np.abs(captured + remaining - exposure.values), initial=0.0)
        )
        conservation_errors.append(conservation)
        domain_violation_count += int(
            np.count_nonzero(captured < 0.0)
            + np.count_nonzero(remaining <= 0.0)
            + np.count_nonzero(captured > exposure.values)
        )
        exact_repeat = bool(
            np.array_equal(captured, repeated.captured_exposure.values)
            and np.array_equal(remaining, repeated.remaining_exposure.values)
            and candidate.geometry_ids == repeated.geometry_ids
        )
        exact_repeats.append(exact_repeat)
        order_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(captured - reverse.captured_exposure.values),
                    dtype=np.float64,
                )
            )
        )
        order_rmses.append(order_rmse)
        excess_fraction = float(
            np.mean(independent > exposure.values, dtype=np.float64)
        )
        no_depletion_excess_fractions.append(excess_fraction)
        relative_remaining = remaining / exposure.values
        density = -np.log10(relative_remaining)
        recovered = np.power(10.0, -density)
        roundtrip = float(np.max(np.abs(recovered - relative_remaining), initial=0.0))
        roundtrip_errors.append(roundtrip)
        finite &= bool(
            np.all(np.isfinite(captured))
            and np.all(np.isfinite(remaining))
            and np.all(np.isfinite(independent))
            and np.all(np.isfinite(density))
        )
        if name.startswith("constant_"):
            constant_capture_fractions.append(
                float(np.mean(captured / exposure.values, dtype=np.float64))
            )
            first = float(np.mean(candidate.layer_captures[0], dtype=np.float64))
            last = float(np.mean(candidate.layer_captures[-1], dtype=np.float64))
            last_first_ratios.append(last / first)
        if name == "checker":
            checker_ratio = _checker_amplitude(
                candidate.layer_captures[0], period
            ) / _checker_amplitude(point.layer_captures[0], period)
        if canonical_geometry_ids is None:
            canonical_geometry_ids = candidate.geometry_ids
        else:
            geometry_repeat_exact &= candidate.geometry_ids == canonical_geometry_ids
        rows.append(
            {
                "fixture": name,
                "captured_sha256": _array_sha(captured),
                "remaining_sha256": _array_sha(remaining),
                "energy_conservation_absolute_error": conservation,
                "forward_reverse_capture_rmse": order_rmse,
                "no_depletion_energy_excess_fraction": excess_fraction,
                "density_transmittance_roundtrip_error": roundtrip,
                "exact_repeat": exact_repeat,
                "minimum_layer_coverage_fraction": min(
                    value for layer in candidate.coverage_fractions for value in layer
                ),
                "maximum_layer_coverage_fraction": max(
                    value for layer in candidate.coverage_fractions for value in layer
                ),
            }
        )
    checks = {
        "energy_conservation": max(conservation_errors)
        <= float(evaluation["maximum_energy_conservation_absolute_error"]),
        "physical_domain": domain_violation_count
        <= int(evaluation["maximum_domain_violation_count"]),
        "constant_capture": min(constant_capture_fractions)
        >= float(evaluation["minimum_constant_total_capture_fraction"]),
        "layer_depletion": max(last_first_ratios)
        <= float(evaluation["maximum_last_to_first_layer_capture_ratio"]),
        "footprint_detail_loss": checker_ratio
        <= float(
            evaluation["maximum_checker_first_layer_amplitude_ratio_vs_point_control"]
        ),
        "no_depletion_control_violates_energy": min(no_depletion_excess_fractions)
        >= float(evaluation["minimum_no_depletion_energy_excess_fraction"]),
        "layer_order_is_material": min(order_rmses)
        >= float(evaluation["minimum_forward_reverse_capture_rmse"]),
        "density_transmittance_roundtrip": max(roundtrip_errors)
        <= float(evaluation["maximum_density_transmittance_roundtrip_error"]),
        "distinct_channel_crystal_maps": canonical_geometry_ids is not None
        and geometry_repeat_exact
        and len(
            {
                geometry_id
                for layer_ids in canonical_geometry_ids
                for geometry_id in layer_ids
            }
        )
        == 3 * profile.layer_count_per_channel,
        "exact_repeat": all(exact_repeats),
        "finite": finite,
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4cc_crystal_light_depletion_v1.json", "sha256"
        ),
        "parent_sha256": contract["parent"]["sha256"],
        "profile_id": profile.identity(),
        "maximum_energy_conservation_absolute_error": max(conservation_errors),
        "domain_violation_count": domain_violation_count,
        "minimum_constant_total_capture_fraction": min(constant_capture_fractions),
        "maximum_last_to_first_layer_capture_ratio": max(last_first_ratios),
        "checker_first_layer_amplitude_ratio_vs_point_control": checker_ratio,
        "minimum_no_depletion_energy_excess_fraction": min(
            no_depletion_excess_fractions
        ),
        "minimum_forward_reverse_capture_rmse": min(order_rmses),
        "maximum_density_transmittance_roundtrip_error": max(roundtrip_errors),
        "rows": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_crystal_light_depletion_mechanism"
            if automatic_pass
            else "close_crystal_light_depletion_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CrystalLightDepletionEvaluationError",
    "evaluate_crystal_light_depletion",
    "write_report",
]
