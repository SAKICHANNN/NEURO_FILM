"""U6.P5N positive finite residual compiler for a measured total-film MTF."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
from scipy.optimize import minimize

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.eval.physical_measured_mtf_budget import _legacy_response
from src.film_physics.measured_mtf import (
    CompiledPsfComponent,
    compiled_channel_response,
)
from src.film_physics.measured_mtf_budget import (
    RGB_CHANNELS,
    MeasuredTotalFilmSpatialBudget,
)

SCHEMA = "neuro_film.u6_p5n_positive_residual_compiler_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5n_positive_residual_compiler_report.v1"
BUNDLE_SCHEMA = "neuro_film.positive_measured_mtf_residual_bundle.v1"


class PositiveMtfResidualError(RuntimeError):
    """Raised when the P5N contract, inputs or constrained fit fail."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PositiveMtfResidualError("P5N paths must be repository-relative")
    return path


def _load_exact(root: Path, path: str, expected: str) -> dict[str, Any]:
    relative = _relative(path)
    resolved = root / relative
    if _hash(resolved) != expected:
        raise PositiveMtfResidualError(f"P5N parent hash mismatch: {relative}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(config: Mapping[str, Any]) -> None:
    family = config.get("candidate_family", {})
    gates = config.get("gates", {})
    if (
        config.get("schema") != SCHEMA
        or family.get("type") != "symmetric_nonnegative_fir_pair_mass"
        or family.get("candidate_radii_px") != [1, 2, 5, 7]
        or not family.get("negative_taps_forbidden")
        or family.get("post_fit_renormalization_forbidden") is not True
        or gates.get("development_max_absolute_response_error") != 0.02
        or gates.get("confirmation_max_absolute_response_error") != 0.01
        or config.get("charts", {}).get("row_partitions") != [7, 17, 31]
    ):
        raise PositiveMtfResidualError("P5N frozen contract drift")
    split = config.get("frequency_split", {})
    development = set(split.get("development_frequencies_cycles_per_mm", []))
    confirmation = set(split.get("confirmation_frequencies_cycles_per_mm", []))
    if development & confirmation or development | confirmation != set(range(25, 66)):
        raise PositiveMtfResidualError("P5N frequency split drift")


def _design_matrix(frequencies: np.ndarray, radius: int, pitch_um: float) -> np.ndarray:
    columns = [np.ones_like(frequencies)]
    for offset in range(1, radius + 1):
        columns.append(np.cos(2.0 * math.pi * frequencies * (pitch_um * 1e-3) * offset))
    return np.column_stack(columns)


def _fit_pair_masses(
    frequencies: np.ndarray,
    target: np.ndarray,
    *,
    radius: int,
    pitch_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = _design_matrix(frequencies, radius, pitch_um)
    initial = np.full(radius + 1, 1.0 / (radius + 1), dtype=np.float64)
    result = minimize(
        lambda masses: float(np.mean((design @ masses - target) ** 2)),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (radius + 1),
        constraints={"type": "eq", "fun": lambda masses: np.sum(masses) - 1.0},
        options={"ftol": 1e-15, "maxiter": 5000},
    )
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or np.any(result.x < -1e-15)
        or abs(float(np.sum(result.x)) - 1.0) > 1e-12
    ):
        raise PositiveMtfResidualError("P5N constrained optimizer failed")
    return result.x, design @ result.x


def _kernel_from_pair_masses(masses: np.ndarray) -> np.ndarray:
    radius = len(masses) - 1
    kernel = np.zeros(2 * radius + 1, dtype=np.float64)
    kernel[radius] = masses[0]
    for offset in range(1, radius + 1):
        kernel[radius - offset] = masses[offset] * 0.5
        kernel[radius + offset] = masses[offset] * 0.5
    return kernel


def _blur(
    values: np.ndarray, sigmas_um: Sequence[float], *, pixel_pitch_um: float
) -> np.ndarray:
    output = np.empty_like(values)
    for index, sigma_um in enumerate(sigmas_um):
        output[..., index] = gaussian_filter(
            values[..., index],
            sigma=float(sigma_um) / pixel_pitch_um,
            mode="nearest",
            truncate=4.0,
        )
    return output


def _apply_legacy_residual(
    values: np.ndarray,
    profile: Mapping[str, Any],
    residual_rgb: Sequence[CompiledPsfComponent],
    *,
    pixel_pitch_um: float,
) -> np.ndarray:
    forward = _blur(
        values,
        profile["forward_scatter_sigma_um_rgb"],
        pixel_pitch_um=pixel_pitch_um,
    )
    adjacency_blur = _blur(
        forward,
        profile["development_adjacency_sigma_um_rgb"],
        pixel_pitch_um=pixel_pitch_um,
    )
    gain = np.asarray(profile["development_adjacency_gain_rgb"], dtype=np.float64)
    adjacent = forward + gain * (forward - adjacency_blur)
    diffused = _blur(
        adjacent,
        profile["dye_diffusion_sigma_um_rgb"],
        pixel_pitch_um=pixel_pitch_um,
    )
    output = np.empty_like(diffused)
    for index, component in enumerate(residual_rgb):
        plane = convolve1d(
            diffused[..., index], component.kernel_1d, axis=0, mode="nearest"
        )
        output[..., index] = convolve1d(
            plane, component.kernel_1d, axis=1, mode="nearest"
        )
    return output


def _legacy_halo(profile: Mapping[str, Any], pitch_um: float) -> int:
    return sum(
        int(4.0 * max(float(value) for value in profile[name]) / pitch_um + 0.5)
        for name in (
            "forward_scatter_sigma_um_rgb",
            "development_adjacency_sigma_um_rgb",
            "dye_diffusion_sigma_um_rgb",
        )
    )


def _apply_row_tiled(
    values: np.ndarray,
    profile: Mapping[str, Any],
    residual_rgb: Sequence[CompiledPsfComponent],
    *,
    pixel_pitch_um: float,
    tile_rows: int,
) -> np.ndarray:
    halo = _legacy_halo(profile, pixel_pitch_um) + max(
        component.radius for component in residual_rgb
    )
    output = np.empty_like(values)
    for y0 in range(0, values.shape[0], tile_rows):
        y1 = min(values.shape[0], y0 + tile_rows)
        sy0 = max(0, y0 - halo)
        sy1 = min(values.shape[0], y1 + halo)
        rendered = _apply_legacy_residual(
            values[sy0:sy1],
            profile,
            residual_rgb,
            pixel_pitch_um=pixel_pitch_um,
        )
        output[y0:y1] = rendered[y0 - sy0 : y1 - sy0]
    return output


def _charts(config: Mapping[str, Any]) -> list[tuple[str, np.ndarray, float]]:
    baseline = float(config["charts"]["small_signal_baseline"])
    amplitude = float(config["charts"]["small_signal_step_amplitude"])
    rows = []
    for height, width in config["charts"]["shapes"]:
        impulse = np.full((height, width, 3), baseline, dtype=np.float64)
        impulse[height // 2, width // 2] += amplitude
        rows.append((f"impulse-{height}x{width}", impulse, amplitude))
        step = np.full_like(impulse, baseline - amplitude * 0.5)
        step[:, width // 2 :] = baseline + amplitude * 0.5
        rows.append((f"step-{height}x{width}", step, amplitude))
        y, x = np.mgrid[:height, :width]
        checker = (((x // 3) + (y // 5)) % 2).astype(np.float64)
        rows.append(
            (
                f"checker-{height}x{width}",
                baseline + amplitude * (np.repeat(checker[..., None], 3, axis=2) - 0.5),
                amplitude,
            )
        )
        for seed in config["charts"]["random_seeds"]:
            random = np.random.default_rng(int(seed) + height * 1000 + width).random(
                (height, width, 3)
            )
            rows.append(
                (
                    f"random-{seed}-{height}x{width}",
                    baseline + amplitude * (random - 0.5),
                    amplitude,
                )
            )
    return rows


def _save_diagnostic(
    output: Path,
    frequencies: np.ndarray,
    targets: Mapping[str, np.ndarray],
    fitted: Mapping[str, np.ndarray],
    chart_target: np.ndarray,
    chart_candidate: np.ndarray,
    chart_amplitude: float,
) -> str:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis, name in zip(axes[0], RGB_CHANNELS, strict=True):
        axis.plot(frequencies, targets[name], label="required residual", lw=2.2)
        axis.plot(frequencies, fitted[name], label="positive FIR", lw=1.8)
        axis.set_title(name)
        axis.set_xlabel("cycles/mm")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.35, 1.0)
    axes[0, 0].set_ylabel("response")
    axes[0, -1].legend(loc="lower left", fontsize=8)
    normalized_difference = np.clip(
        np.abs(chart_candidate - chart_target) / chart_amplitude, 0.0, 1.0
    )
    for axis, image, title in zip(
        axes[1],
        (chart_target, chart_candidate, normalized_difference),
        ("measured 2D target", "legacy + residual", "normalized |difference|"),
        strict=True,
    ):
        axis.imshow(np.clip(image, 0.0, 1.0), interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, metadata={"Software": "neuro-film"})
    plt.close(figure)
    return _hash(output)


def compile_and_evaluate_residual(
    *, root: Path, config: Mapping[str, Any], diagnostic_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_contract(config)
    parents = config["parents"]
    p5m = _load_exact(root, parents["p5m_contract"], parents["p5m_contract_sha256"])
    decision = _load_exact(
        root, parents["p5m_decision"], parents["p5m_decision_sha256"]
    )
    if decision.get(
        "decision"
    ) != "open_positive_residual_compiler" or not decision.get("automatic_pass"):
        raise PositiveMtfResidualError("P5N parent decision drift")
    p5l_payload = _load_exact(
        root,
        p5m["parents"]["p5l_bundle"],
        p5m["parents"]["p5l_bundle_sha256"],
    )
    generic = _load_exact(
        root,
        p5m["parents"]["generic_spatial_contract"],
        p5m["parents"]["generic_spatial_contract_sha256"],
    )
    measured_budget = MeasuredTotalFilmSpatialBudget.from_lod_bundle(p5l_payload)
    profile = generic["profile"]
    pitch_um = float(p5m["domain"]["pixel_pitch_um"])
    development = np.asarray(
        config["frequency_split"]["development_frequencies_cycles_per_mm"],
        dtype=np.float64,
    )
    confirmation = np.asarray(
        config["frequency_split"]["confirmation_frequencies_cycles_per_mm"],
        dtype=np.float64,
    )
    all_frequencies = np.arange(25.0, 66.0, dtype=np.float64)
    targets = {}
    fitted = {}
    residual_components = []
    channel_rows = {}
    for rgb_index, name in enumerate(RGB_CHANNELS):
        measured_all = compiled_channel_response(
            measured_budget.compiled_rgb[rgb_index],
            all_frequencies,
            pixel_pitch_um=pitch_um,
        )
        legacy_all = _legacy_response(profile, rgb_index, all_frequencies)
        target_all = measured_all / legacy_all
        development_target = np.interp(development, all_frequencies, target_all)
        selected = None
        trials = []
        for radius in config["candidate_family"]["candidate_radii_px"]:
            masses, development_fit = _fit_pair_masses(
                development,
                development_target,
                radius=int(radius),
                pitch_um=pitch_um,
            )
            development_max = float(
                np.max(np.abs(development_fit - development_target))
            )
            trials.append({"radius": int(radius), "development_max": development_max})
            if selected is None and development_max <= float(
                config["gates"]["development_max_absolute_response_error"]
            ):
                selected = (int(radius), masses, development_max)
        if selected is None:
            raise PositiveMtfResidualError(f"no development family passed for {name}")
        radius, masses, development_max = selected
        kernel = _kernel_from_pair_masses(masses)
        component = CompiledPsfComponent(1.0, kernel)
        residual_components.append(component)
        all_fit = compiled_channel_response(
            (component,), all_frequencies, pixel_pitch_um=pitch_um
        )
        confirmation_target = np.interp(confirmation, all_frequencies, target_all)
        confirmation_fit = compiled_channel_response(
            (component,), confirmation, pixel_pitch_um=pitch_um
        )
        confirmation_max = float(np.max(np.abs(confirmation_fit - confirmation_target)))
        targets[name] = target_all
        fitted[name] = all_fit
        channel_rows[name] = {
            "selected_radius_px": radius,
            "pair_masses": masses.tolist(),
            "development_max_absolute_error": development_max,
            "confirmation_max_absolute_error": confirmation_max,
            "candidate_trials": trials,
        }

    residual_rgb = tuple(residual_components)
    chart_rows = []
    normalized_errors = []
    diagnostic_arrays = None
    for name, values, amplitude in _charts(config):
        target = measured_budget.apply(values)
        candidate = _apply_legacy_residual(
            values, profile, residual_rgb, pixel_pitch_um=pitch_um
        )
        normalized_error = np.abs(candidate - target) / amplitude
        normalized_errors.extend(normalized_error.ravel().tolist())
        chart_rows.append(
            {
                "name": name,
                "normalized_max_absolute_error": float(np.max(normalized_error)),
                "normalized_mean_absolute_error": float(np.mean(normalized_error)),
            }
        )
        if diagnostic_arrays is None and name.startswith("checker-64x97"):
            diagnostic_arrays = (target, candidate)
    if diagnostic_arrays is None:
        raise PositiveMtfResidualError("P5N diagnostic chart missing")

    constant_max = 0.0
    for value in config["charts"]["constant_values"]:
        field = np.full((37, 53, 3), float(value), dtype=np.float64)
        constant_max = max(
            constant_max,
            float(
                np.max(
                    np.abs(
                        _apply_legacy_residual(
                            field, profile, residual_rgb, pixel_pitch_um=pitch_um
                        )
                        - field
                    )
                )
            ),
        )
    baseline = float(config["charts"]["small_signal_baseline"])
    amplitude = float(config["charts"]["small_signal_step_amplitude"])
    step = np.full((129, 191, 3), baseline - amplitude * 0.5)
    step[:, 95:] = baseline + amplitude * 0.5
    step_output = _apply_legacy_residual(
        step, profile, residual_rgb, pixel_pitch_um=pitch_um
    )
    undershoot = max(0.0, float((baseline - amplitude * 0.5) - np.min(step_output)))
    overshoot = max(0.0, float(np.max(step_output) - (baseline + amplitude * 0.5)))
    partitions = {
        str(rows): bool(
            np.array_equal(
                step_output,
                _apply_row_tiled(
                    step,
                    profile,
                    residual_rgb,
                    pixel_pitch_um=pitch_um,
                    tile_rows=int(rows),
                ),
            )
        )
        for rows in config["charts"]["row_partitions"]
    }
    repeat = bool(
        np.array_equal(
            step_output,
            _apply_legacy_residual(
                step, profile, residual_rgb, pixel_pitch_um=pitch_um
            ),
        )
    )
    impulse = np.zeros((257, 257, 3), dtype=np.float64)
    impulse[128, 128] = 1.0
    effective = _apply_legacy_residual(
        impulse, profile, residual_rgb, pixel_pitch_um=pitch_um
    )
    impulse_min = float(np.min(effective))
    impulse_sum_error = float(np.max(np.abs(np.sum(effective, axis=(0, 1)) - 1.0)))

    diagnostic_sha = _save_diagnostic(
        diagnostic_path,
        all_frequencies,
        targets,
        fitted,
        diagnostic_arrays[0],
        diagnostic_arrays[1],
        float(config["charts"]["small_signal_step_amplitude"]),
    )
    gates = config["gates"]
    chart_max = float(max(normalized_errors))
    chart_mean = float(np.mean(normalized_errors))
    gate_results = {
        "development_fit": all(
            row["development_max_absolute_error"]
            <= float(gates["development_max_absolute_response_error"])
            for row in channel_rows.values()
        ),
        "confirmation_fit": all(
            row["confirmation_max_absolute_error"]
            <= float(gates["confirmation_max_absolute_response_error"])
            for row in channel_rows.values()
        ),
        "kernel_positive_normalized": all(
            float(np.min(component.kernel_1d)) >= float(gates["kernel_minimum"])
            and abs(float(np.sum(component.kernel_1d)) - 1.0)
            <= float(gates["kernel_sum_absolute_error"])
            for component in residual_rgb
        ),
        "combined_effective_impulse": impulse_min
        >= float(gates["combined_effective_impulse_minimum"])
        and impulse_sum_error <= float(gates["combined_effective_impulse_sum_error"]),
        "two_dimensional_chart_max": chart_max
        <= float(gates["two_dimensional_chart_max_absolute_error"]),
        "two_dimensional_chart_mean": chart_mean
        <= float(gates["two_dimensional_chart_mean_absolute_error"]),
        "constant_preservation": constant_max
        <= float(gates["constant_max_absolute_error"]),
        "step_no_undershoot": undershoot <= float(gates["step_undershoot_max"]),
        "step_no_overshoot": overshoot <= float(gates["step_overshoot_max"]),
        "row_partition_exact": all(partitions.values()),
        "repeat_exact": repeat,
    }
    bundle_core = {
        "schema": BUNDLE_SCHEMA,
        "parent_spatial_budget_id": decision["spatial_budget_id"],
        "pixel_pitch_um": pitch_um,
        "channels": {
            name: {
                "selected_radius_px": channel_rows[name]["selected_radius_px"],
                "pair_masses": channel_rows[name]["pair_masses"],
                "kernel_1d": residual_rgb[index].kernel_1d.tolist(),
            }
            for index, name in enumerate(RGB_CHANNELS)
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    bundle = {
        **bundle_core,
        "residual_bundle_id": hashlib.sha256(_canonical_json(bundle_core)).hexdigest(),
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash(
            root / "configs/u6_p5n_positive_residual_compiler_v1.json"
        ),
        "parent_spatial_budget_id": decision["spatial_budget_id"],
        "residual_bundle_id": bundle["residual_bundle_id"],
        "channel_rows": channel_rows,
        "chart_rows": chart_rows,
        "two_dimensional_chart_max_absolute_error": chart_max,
        "two_dimensional_chart_mean_absolute_error": chart_mean,
        "combined_effective_impulse_minimum": impulse_min,
        "combined_effective_impulse_sum_error": impulse_sum_error,
        "constant_max_absolute_error": constant_max,
        "step_undershoot": undershoot,
        "step_overshoot": overshoot,
        "row_partition_exact": partitions,
        "repeat_exact": repeat,
        "diagnostic_sha256": diagnostic_sha,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "open_synthetic_density_nonlinearity_stress"
        if passed
        else "close_residual_use_measured_total_replacement",
        "claim_ceiling": config["claim_ceiling"],
    }
    return bundle, report


__all__ = [
    "BUNDLE_SCHEMA",
    "REPORT_SCHEMA",
    "SCHEMA",
    "PositiveMtfResidualError",
    "compile_and_evaluate_residual",
    "validate_contract",
]
