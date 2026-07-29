"""U6.P4X one-parameter generic material-amplitude development/confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_physical import (
    render_anisotropic_structure,
)
from src.eval.real_uniform_grain_source import hash_file
from src.eval.scan_amplitude_feasibility import robust_relative_amplitude


SCHEMA = "neuro_film.u6_p4x_global_material_amplitude_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4x_global_material_amplitude_report.v1"


class GlobalMaterialAmplitudeError(RuntimeError):
    """Raised when the P4X proxy, split or evidence drifts."""


def _load_exact_json(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise GlobalMaterialAmplitudeError(f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GlobalMaterialAmplitudeError("parent must be a JSON object")
    return payload


def validate_contract(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = config["proxy_model"]
    if (
        config.get("schema") != SCHEMA
        or config["observed_target"].get("stock_labels_available_to_fit")
        or model.get("per_channel_scale_allowed")
        or model.get("stock_specific_fit_allowed")
        or model.get("per_scan_fit_allowed")
        or model.get("photographic_fit_allowed")
        or config["execution"].get("photographic_render_allowed")
        or config["execution"].get("post_confirmation_retuning_allowed")
        or config["execution"].get("production_integration_allowed")
    ):
        raise GlobalMaterialAmplitudeError("unsupported P4X contract")
    parents = config["parents"]
    decision = _load_exact_json(root, parents["p4w_decision"])
    observed = _load_exact_json(root, parents["p4w_report"])
    p4t = _load_exact_json(root, parents["p4t_contract"])
    p4t_decision = _load_exact_json(root, parents["p4t_decision"])
    if (
        decision["decision"]
        != parents["p4w_decision"]["required_decision"]
        or not observed["automatic_pass"]
        or observed["stock_labels_used"]
        or p4t_decision["result"]["status"]
        != "automatic_and_synthetic_severe_pass"
        or [float(value) for value in model["fixed_sigma_yx_pixels"]]
        != [
            float(value)
            for value in p4t["candidate"]["sigma_yx_pixels"]
        ]
        or [
            float(value)
            for value in model[
                "base_grain_optical_density_by_rgb_layer"
            ]
        ]
        != [
            float(value)
            for value in p4t["candidate"][
                "grain_optical_density_by_rgb_layer"
            ]
        ]
        or model["density_to_transmittance"] != "10^-D"
        or model["assumption_is_calibration"]
    ):
        raise GlobalMaterialAmplitudeError("P4X parent or proxy drift")
    scales = [float(value) for value in model["shared_amplitude_scale_grid"]]
    if (
        scales != sorted(set(scales))
        or any(value <= 0.0 or value > 1.0 for value in scales)
        or scales[-1] != 1.0
    ):
        raise GlobalMaterialAmplitudeError("P4X scale grid drift")
    return observed, p4t


def _render_proxy(
    *, model: dict[str, Any], scale: float, seed: int
) -> tuple[float, dict[str, Any], np.ndarray]:
    height, width = (int(value) for value in model["field_shape"])
    target_density = float(model["target_density"])
    target = np.full((height, width, 3), target_density, dtype=np.float64)
    grain_od = [
        float(value) * float(scale)
        for value in model["base_grain_optical_density_by_rgb_layer"]
    ]
    density, _ = render_anisotropic_structure(
        target,
        grain_optical_density_by_channel=grain_od,
        sigma_yx=tuple(
            float(value) for value in model["fixed_sigma_yx_pixels"]
        ),
        seeds=[
            int(seed + 104729 * channel) for channel in range(3)
        ],
        maximum_target_density=float(model["maximum_target_density"]),
        truncate=float(model["truncate"]),
    )
    y0, x0, crop_h, crop_w = (
        int(value) for value in model["central_analysis_crop"]
    )
    transmittance = np.power(
        10.0, -density.astype(np.float64)
    )
    amplitudes = [
        robust_relative_amplitude(
            transmittance[
                y0 : y0 + crop_h,
                x0 : x0 + crop_w,
                channel,
            ]
        )
        for channel in range(3)
    ]
    border = 8
    interior = density[border:-border, border:-border]
    diagnostics = {
        "channel_amplitudes": amplitudes,
        "proxy_amplitude": float(np.median(amplitudes)),
        "maximum_channel_mean_density_error": float(
            max(
                abs(
                    np.mean(interior[..., channel], dtype=np.float64)
                    - target_density
                )
                for channel in range(3)
            )
        ),
        "minimum_density": float(np.min(density)),
        "maximum_transmittance": float(np.max(transmittance)),
        "density_sha256": hashlib.sha256(
            np.ascontiguousarray(density).tobytes()
        ).hexdigest(),
    }
    return diagnostics["proxy_amplitude"], diagnostics, density


def _candidate_summary(
    *, model: dict[str, Any], scale: float, seeds: list[int]
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        amplitude, diagnostics, _ = _render_proxy(
            model=model, scale=scale, seed=int(seed)
        )
        rows.append(
            {
                "seed": int(seed),
                "proxy_amplitude": amplitude,
                **diagnostics,
            }
        )
    return {
        "scale": float(scale),
        "seed_rows": rows,
        "median_proxy_amplitude": float(
            np.median([row["proxy_amplitude"] for row in rows])
        ),
        "maximum_channel_mean_density_error": float(
            max(row["maximum_channel_mean_density_error"] for row in rows)
        ),
        "minimum_density": float(min(row["minimum_density"] for row in rows)),
        "maximum_transmittance": float(
            max(row["maximum_transmittance"] for row in rows)
        ),
    }


def _partition_exact(
    *, model: dict[str, Any], scale: float, seed: int
) -> bool:
    _, _, full = _render_proxy(model=model, scale=scale, seed=seed)
    height, width = full.shape[:2]
    target = np.full(
        (height, width, 3),
        float(model["target_density"]),
        dtype=np.float64,
    )
    grain_od = [
        float(value) * float(scale)
        for value in model["base_grain_optical_density_by_rgb_layer"]
    ]
    assembled = np.empty_like(full)
    for y0 in range(0, height, 31):
        rows = min(31, height - y0)
        density, _ = render_anisotropic_structure(
            target,
            grain_optical_density_by_channel=grain_od,
            sigma_yx=tuple(
                float(value) for value in model["fixed_sigma_yx_pixels"]
            ),
            seeds=[
                int(seed + 104729 * channel) for channel in range(3)
            ],
            maximum_target_density=float(model["maximum_target_density"]),
            truncate=float(model["truncate"]),
            origin_yx=(y0, 0),
            shape=(rows, width),
        )
        assembled[y0 : y0 + rows] = density
    return bool(np.array_equal(full, assembled))


def evaluate_global_material_amplitude(
    *, root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    observed, _ = validate_contract(root, config)
    model = config["proxy_model"]
    development_target = float(
        observed["summary"]["development_median_amplitude"]
    )
    confirmation_target = float(
        observed["summary"]["confirmation_median_amplitude"]
    )
    development = [
        _candidate_summary(
            model=model,
            scale=float(scale),
            seeds=[int(value) for value in model["development_seeds"]],
        )
        for scale in model["shared_amplitude_scale_grid"]
    ]
    selected = min(
        development,
        key=lambda row: (
            abs(row["median_proxy_amplitude"] - development_target)
            / development_target,
            row["scale"],
        ),
    )
    selected_scale = float(selected["scale"])
    confirmation = _candidate_summary(
        model=model,
        scale=selected_scale,
        seeds=[int(value) for value in model["confirmation_seeds"]],
    )
    repeated = _candidate_summary(
        model=model,
        scale=selected_scale,
        seeds=[int(value) for value in model["confirmation_seeds"]],
    )
    baseline = _candidate_summary(
        model=model,
        scale=1.0,
        seeds=[int(value) for value in model["confirmation_seeds"]],
    )
    candidate_error = abs(
        confirmation["median_proxy_amplitude"] - confirmation_target
    ) / confirmation_target
    baseline_error = abs(
        baseline["median_proxy_amplitude"] - confirmation_target
    ) / confirmation_target
    error_improvement = 1.0 - candidate_error / max(
        baseline_error, np.finfo(np.float64).eps
    )
    gates = config["automatic_gates"]
    checks = {
        "confirmation_amplitude": candidate_error
        <= float(gates["maximum_confirmation_relative_amplitude_error"]),
        "baseline_improvement": error_improvement
        >= float(
            gates[
                "minimum_confirmation_error_improvement_over_original_p4t"
            ]
        ),
        "observed_range": float(
            observed["summary"]["minimum_scan_amplitude"]
        )
        <= confirmation["median_proxy_amplitude"]
        <= float(observed["summary"]["maximum_scan_amplitude"]),
        "density_mean": confirmation[
            "maximum_channel_mean_density_error"
        ]
        <= float(gates["maximum_channel_mean_density_error"]),
        "physical_domain": confirmation["minimum_density"]
        >= float(gates["minimum_density"])
        and confirmation["maximum_transmittance"]
        <= float(gates["maximum_transmittance"]),
        "repeat_exact": confirmation == repeated,
        "row_partition_exact": _partition_exact(
            model=model,
            scale=selected_scale,
            seed=int(model["confirmation_seeds"][0]),
        ),
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "stock_labels_used_for_fit": False,
        "development_target_amplitude": development_target,
        "confirmation_target_amplitude": confirmation_target,
        "development_grid": development,
        "selected_shared_amplitude_scale": selected_scale,
        "confirmation_candidate": confirmation,
        "confirmation_original_p4t": baseline,
        "confirmation_relative_error": float(candidate_error),
        "original_p4t_relative_error": float(baseline_error),
        "confirmation_error_improvement_fraction": float(error_improvement),
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_global_transmittance_proxy_amplitude_candidate"
            if automatic_pass
            else "close_global_scan_amplitude_fit"
        ),
        "branch": config["branch_rule"][
            "pass" if automatic_pass else "fail"
        ],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "evaluate_global_material_amplitude",
    "validate_contract",
    "write_report",
]
