"""U6.P3Q1 synthetic evaluation of analytical film-base return topology."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.film_physics.base_return_geometry import (
    BaseReturnGeometryProfile,
    apply_positive_base_return_geometry,
    base_return_geometry_kernel,
    equal_second_moment_gaussian,
    radial_quantile,
)

SCHEMA = "neuro_film.u6_p3q1_analytical_base_return_geometry_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3q1_analytical_base_return_geometry_report.v1"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("P3Q1 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    parent = config.get("parent", {})
    profile = config.get("generic_hypothesis_profile", {})
    gates = config.get("automatic_gates", {})
    if (
        config.get("schema") != SCHEMA
        or config.get("experiment_id") != "U6.P3Q1"
        or parent.get("decision_sha256") != "7f01192f2e34b5e0504712dad65e71214bdadb40e0b5d80785a41fdbd5d285a0"
        or parent.get("stable_evidence_id") != "e1401c890382f14c53375ac96551ca20ef6a67bf1543a497fd02921c65d8b0b7"
        or profile != {
            "support_thickness_um": 120.0,
            "refractive_index": 1.5,
            "attenuation_length_um": 1200.0,
            "maximum_radius_um": 1200.0,
            "pixel_pitch_um": 8.0,
            "return_fraction_rgb": [0.065, 0.018, 0.006],
        }
        or gates.get("minimum_l1_distance_from_equal_moment_gaussian") != 0.5
        or gates.get("minimum_thickness_q50_ratio") != 1.85
        or gates.get("maximum_thickness_q50_ratio") != 2.15
        or gates.get("minimum_thickness_q95_ratio") != 1.85
        or gates.get("maximum_thickness_q95_ratio") != 2.15
        or not gates.get("require_shorter_attenuation_q95_reduction")
        or not gates.get("two_byte_identical_audits")
        or config["candidate"].get("fitting_allowed")
        or config["candidate"].get("photograph_selection_allowed")
        or config["candidate"].get("hard_clipping_allowed")
    ):
        raise ValueError("invalid frozen U6.P3Q1 contract")
    _relative_path(str(parent.get("decision_path", "")))
    return config


def _profile(row: dict[str, Any]) -> BaseReturnGeometryProfile:
    return BaseReturnGeometryProfile(
        support_thickness_um=float(row["support_thickness_um"]),
        refractive_index=float(row["refractive_index"]),
        attenuation_length_um=float(row["attenuation_length_um"]),
        maximum_radius_um=float(row["maximum_radius_um"]),
        pixel_pitch_um=float(row["pixel_pitch_um"]),
    )


def _mode_radius(kernel: np.ndarray, pitch: float) -> float:
    centre = kernel.shape[0] // 2
    y, x = np.unravel_index(int(np.argmax(kernel)), kernel.shape)
    return float(math.hypot(x - centre, y - centre) * pitch)


def evaluate_geometry(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    parent = config["parent"]
    decision_path = root / _relative_path(str(parent["decision_path"]))
    if not decision_path.is_file() or _hash_file(decision_path) != parent["decision_sha256"]:
        raise ValueError("P3Q1 parent source decision drift")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "open_u6_p3q1_analytical_base_return_geometry" or decision["formal_evidence"].get("stable_evidence_id") != parent["stable_evidence_id"]:
        raise ValueError("P3Q1 parent source decision mismatch")

    base_row = dict(config["generic_hypothesis_profile"])
    fractions = tuple(float(value) for value in base_row.pop("return_fraction_rgb"))
    profile = _profile(base_row)
    kernel = base_return_geometry_kernel(profile)
    gaussian = equal_second_moment_gaussian(kernel, profile.pixel_pitch_um)
    radius = np.indices(kernel.shape, dtype=np.float64)
    centre = kernel.shape[0] // 2
    radial_um = np.hypot(radius[1] - centre, radius[0] - centre) * profile.pixel_pitch_um
    inner_energy = float(np.sum(kernel[radial_um < profile.critical_radius_um], dtype=np.float64))
    moment = float(np.sum(kernel * np.square(radial_um), dtype=np.float64))
    gaussian_moment = float(np.sum(gaussian * np.square(radial_um), dtype=np.float64))

    scale_profiles = []
    for override in config["controls"]["thickness_scale_profiles"]:
        row = {**base_row, **override}
        scaled_profile = _profile(row)
        scaled_kernel = base_return_geometry_kernel(scaled_profile)
        scale_profiles.append(
            {
                "support_thickness_um": scaled_profile.support_thickness_um,
                "q50_um": radial_quantile(scaled_kernel, scaled_profile.pixel_pitch_um, 0.50),
                "q95_um": radial_quantile(scaled_kernel, scaled_profile.pixel_pitch_um, 0.95),
            }
        )
    q50_ratio = scale_profiles[1]["q50_um"] / scale_profiles[0]["q50_um"]
    q95_ratio = scale_profiles[1]["q95_um"] / scale_profiles[0]["q95_um"]
    high_index = _profile({**base_row, "refractive_index": float(config["controls"]["higher_index"])})
    short_attenuation = _profile({**base_row, "attenuation_length_um": float(config["controls"]["shorter_attenuation_length_um"])})
    short_kernel = base_return_geometry_kernel(short_attenuation)
    base_q95 = radial_quantile(kernel, profile.pixel_pitch_um, 0.95)
    short_q95 = radial_quantile(short_kernel, short_attenuation.pixel_pitch_um, 0.95)

    shape = tuple(int(value) for value in config["witnesses"]["shape"])
    constant = np.full(shape, float(config["witnesses"]["constant_exposure"]), dtype=np.float64)
    background = float(config["witnesses"]["background_exposure"])
    highlight = float(config["witnesses"]["highlight_exposure"])
    impulse = np.full(shape, background, dtype=np.float64)
    witness_center = (shape[0] // 2, shape[1] // 2)
    impulse[witness_center] = highlight
    edge = np.full(shape, background, dtype=np.float64)
    edge[:, int(config["witnesses"]["edge_column"]) :] = highlight
    constant_result = apply_positive_base_return_geometry(constant, kernel, fractions)
    impulse_result = apply_positive_base_return_geometry(impulse, kernel, fractions)
    edge_result = apply_positive_base_return_geometry(edge, kernel, fractions)
    mode_offset = int(round(_mode_radius(kernel, profile.pixel_pitch_um) / profile.pixel_pitch_um))
    far = impulse_result.residual[witness_center[0], witness_center[1] + mode_offset]

    metrics = {
        "critical_angle_degrees": math.degrees(profile.critical_angle_rad),
        "critical_radius_um": profile.critical_radius_um,
        "kernel_shape": list(kernel.shape),
        "kernel_sum_absolute_error": abs(float(np.sum(kernel, dtype=np.float64)) - 1.0),
        "minimum_kernel_weight": float(np.min(kernel)),
        "center_weight": float(kernel[centre, centre]),
        "inner_support_energy": inner_energy,
        "mode_radius_um": _mode_radius(kernel, profile.pixel_pitch_um),
        "radial_second_moment_um2": moment,
        "gaussian_radial_second_moment_um2": gaussian_moment,
        "l1_distance_from_equal_moment_gaussian": float(np.sum(np.abs(kernel - gaussian), dtype=np.float64)),
        "thickness_scale_profiles": scale_profiles,
        "thickness_q50_ratio": q50_ratio,
        "thickness_q95_ratio": q95_ratio,
        "higher_index_critical_radius_ratio": high_index.critical_radius_um / profile.critical_radius_um,
        "base_q95_um": base_q95,
        "shorter_attenuation_q95_um": short_q95,
        "constant_maximum_residual": float(np.max(np.abs(constant_result.residual))),
        "impulse_exterior_energy": float(np.sum(impulse_result.residual, dtype=np.float64)),
        "far_red_over_blue_ratio": float(far[0] / max(far[2], 1e-30)),
        "edge_dark_side_maximum_increment": float(np.max(edge_result.residual[:, : int(config["witnesses"]["edge_column"])])),
        "minimum_output": float(min(np.min(impulse_result.output), np.min(edge_result.output))),
        "out_of_domain_fraction": float(np.mean(~np.isfinite(impulse_result.output) | (impulse_result.output < 0.0))),
    }
    gates = config["automatic_gates"]
    checks = {
        "kernel_normalization": metrics["kernel_sum_absolute_error"] <= float(gates["maximum_kernel_sum_absolute_error"]),
        "kernel_nonnegative": metrics["minimum_kernel_weight"] >= float(gates["minimum_kernel_weight"]),
        "annular_center": metrics["center_weight"] <= float(gates["maximum_center_weight"]),
        "critical_inner_support": metrics["inner_support_energy"] <= float(gates["maximum_inner_support_energy"]),
        "mode_radius": metrics["mode_radius_um"] >= float(gates["minimum_mode_radius_fraction_of_critical_radius"]) * metrics["critical_radius_um"],
        "gaussian_topology_distinct": metrics["l1_distance_from_equal_moment_gaussian"] >= float(gates["minimum_l1_distance_from_equal_moment_gaussian"]),
        "thickness_q50_order": float(gates["minimum_thickness_q50_ratio"]) <= q50_ratio <= float(gates["maximum_thickness_q50_ratio"]),
        "thickness_q95_order": float(gates["minimum_thickness_q95_ratio"]) <= q95_ratio <= float(gates["maximum_thickness_q95_ratio"]),
        "refractive_index_order": metrics["higher_index_critical_radius_ratio"] <= float(gates["maximum_higher_index_critical_radius_ratio"]),
        "attenuation_tail_order": short_q95 < base_q95,
        "constant_identity": metrics["constant_maximum_residual"] <= float(gates["maximum_constant_residual"]),
        "impulse_exterior_energy": metrics["impulse_exterior_energy"] >= float(gates["minimum_impulse_exterior_energy"]),
        "spectral_tail": metrics["far_red_over_blue_ratio"] >= float(gates["minimum_far_red_over_blue_ratio"]),
        "domain": metrics["out_of_domain_fraction"] <= float(gates["maximum_out_of_domain_fraction"]),
        "no_parameter_fit": True,
    }
    passed = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "parent_decision_sha256": parent["decision_sha256"],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": config["branches"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {**stable, "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest()}
    return report, {"candidate": kernel, "gaussian": gaussian, "impulse_red": impulse_result.residual[..., 0]}


def render_diagnostic(arrays: dict[str, np.ndarray], output: Path) -> str:
    raw_panels = []
    for key in ("candidate", "gaussian", "impulse_red"):
        values = np.asarray(arrays[key], dtype=np.float64)
        display = np.log1p(values / max(float(np.max(values)), 1e-30) * 1000.0) / math.log1p(1000.0)
        raw_panels.append(np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8))
    height = max(panel.shape[0] for panel in raw_panels)
    width = max(panel.shape[1] for panel in raw_panels)
    panels = []
    for panel in raw_panels:
        top = (height - panel.shape[0]) // 2
        left = (width - panel.shape[1]) // 2
        padded = np.zeros((height, width), dtype=np.uint8)
        padded[top : top + panel.shape[0], left : left + panel.shape[1]] = panel
        panels.append(padded)
    canvas = np.concatenate(panels, axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(output)
    return _hash_file(output)


__all__ = ["REPORT_SCHEMA", "SCHEMA", "evaluate_geometry", "load_contract", "render_diagnostic"]
