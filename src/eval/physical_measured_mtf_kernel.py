"""U6.P5K discrete positive-kernel conformance for the measured MTF bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.film_physics.measured_mtf import (
    apply_compiled_positive_psf,
    apply_compiled_positive_psf_row_tiled,
    channel_psf_from_json,
    compile_channel_psf,
    compiled_channel_response,
    required_compiled_psf_halo,
)

SCHEMA = "neuro_film.u6_p5k_measured_mtf_kernel_conformance_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5k_measured_mtf_kernel_conformance_report.v1"
CHANNELS = ("blue", "green", "red")


class MeasuredMtfKernelError(RuntimeError):
    """Raised when the frozen P5K contract or parent bundle drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MeasuredMtfKernelError("P5K paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    compiler = payload.get("compiler", {})
    charts = payload.get("charts", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("bundle_sha256")
        != "b2d0ff39743ba3ec21a8d033fe38f3ebf6e7360e5a75ea6b838dc67b4b856270"
        or parent.get("bundle_id")
        != "80bff073e5da5a4da1dcdd46b11238e1b2b3ca96596b3f082c419d791b60d1b8"
        or compiler.get("reference_dpi") != 16000.0
        or compiler.get("reference_pixel_pitch_um") != 1.5875
        or compiler.get("direct_product_dpi_control") != 4000.0
        or compiler.get("gaussian_cell_integration")
        != "normal CDF over each pixel cell"
        or compiler.get("truncate_sigma") != 5.0
        or compiler.get("boundary_mode") != "nearest"
        or compiler.get("frequency_interval_cycles_per_mm") != [25.0, 65.0]
        or compiler.get("frequency_grid_count") != 161
        or charts.get("impulse_shape") != [129, 129]
        or charts.get("step_shape") != [257, 509]
        or charts.get("row_partitions") != [31, 47, 127]
        or gates.get("reference_discrete_vs_analytic_max_error") != 0.01
        or gates.get("direct_4000dpi_warning_error") != 0.02
        or not gates.get("row_partition_byte_exact")
        or not gates.get("repeat_byte_exact")
    ):
        raise MeasuredMtfKernelError("P5K frozen contract drift")
    _relative_path(parent.get("bundle", ""))
    return payload


def _compiled(bundle: Mapping[str, Any], pitch: float, truncate: float) -> tuple[Any, ...]:
    return tuple(
        compile_channel_psf(
            channel_psf_from_json(bundle["channels"][channel]),
            pixel_pitch_um=pitch,
            truncate_sigma=truncate,
        )
        for channel in CHANNELS
    )


def _diagnostic(impulse: np.ndarray, step: np.ndarray, output: Path) -> str:
    impulse_crop = impulse[48:81, 48:81]
    impulse_view = np.log1p(1000.0 * impulse_crop)
    impulse_view /= max(float(np.max(impulse_view)), 1e-12)
    step_view = step[95:161]
    center = step.shape[1] // 2
    canvas = np.concatenate(
        (np.repeat(impulse_view, 2, axis=0), step_view[:, center - 33 : center + 33]),
        axis=1,
    )
    encoded = np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="RGB").save(output)
    return _hash_file(output)


def evaluate_kernel(
    config: Mapping[str, Any], root: Path, *, diagnostic_path: Path
) -> dict[str, Any]:
    bundle_path = root / _relative_path(str(config["parent"]["bundle"]))
    if _hash_file(bundle_path) != config["parent"]["bundle_sha256"]:
        raise MeasuredMtfKernelError("P5K parent bundle hash mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("bundle_id") != config["parent"]["bundle_id"]:
        raise MeasuredMtfKernelError("P5K parent bundle identity mismatch")
    compiler = config["compiler"]
    reference_pitch = float(compiler["reference_pixel_pitch_um"])
    control_pitch = float(compiler["direct_product_pixel_pitch_um"])
    truncate = float(compiler["truncate_sigma"])
    reference = _compiled(bundle, reference_pitch, truncate)
    control = _compiled(bundle, control_pitch, truncate)
    frequencies = np.linspace(
        *compiler["frequency_interval_cycles_per_mm"],
        int(compiler["frequency_grid_count"]),
        dtype=np.float64,
    )
    response_rows: dict[str, Any] = {}
    reference_matrix = []
    reference_errors = []
    control_errors = []
    for index, channel in enumerate(CHANNELS):
        model = channel_psf_from_json(bundle["channels"][channel])
        analytic = model.response(frequencies)
        reference_response = compiled_channel_response(
            reference[index], frequencies, pixel_pitch_um=reference_pitch
        )
        control_response = compiled_channel_response(
            control[index], frequencies, pixel_pitch_um=control_pitch
        )
        reference_error = np.abs(reference_response - analytic)
        control_error = np.abs(control_response - analytic)
        reference_matrix.append(reference_response)
        reference_errors.extend(reference_error.tolist())
        control_errors.extend(control_error.tolist())
        response_rows[channel] = {
            "component_radii_reference_px": [row.radius for row in reference[index]],
            "component_radii_direct_4000dpi_px": [row.radius for row in control[index]],
            "reference_max_error": float(np.max(reference_error)),
            "reference_rmse": float(np.sqrt(np.mean(np.square(reference_error)))),
            "direct_4000dpi_max_error": float(np.max(control_error)),
            "direct_4000dpi_rmse": float(np.sqrt(np.mean(np.square(control_error)))),
        }
    reference_matrix = np.asarray(reference_matrix)

    impulse_shape = tuple(int(value) for value in config["charts"]["impulse_shape"])
    impulse = np.zeros((*impulse_shape, 3), dtype=np.float64)
    impulse[impulse_shape[0] // 2, impulse_shape[1] // 2, :] = 1.0
    impulse_output = apply_compiled_positive_psf(impulse, reference)
    impulse_sums = np.sum(impulse_output, axis=(0, 1))
    step_shape = tuple(int(value) for value in config["charts"]["step_shape"])
    step = np.zeros((*step_shape, 3), dtype=np.float64)
    step[:, step_shape[1] // 2 :, :] = 1.0
    step_output = apply_compiled_positive_psf(step, reference)
    constant_error = 0.0
    for value in config["charts"]["constant_values"]:
        field = np.full((41, 53, 3), float(value), dtype=np.float64)
        constant_error = max(
            constant_error,
            float(np.max(np.abs(apply_compiled_positive_psf(field, reference) - field))),
        )
    partition_exact = {}
    for rows in config["charts"]["row_partitions"]:
        tiled = apply_compiled_positive_psf_row_tiled(step, reference, tile_rows=int(rows))
        partition_exact[str(rows)] = bool(np.array_equal(tiled, step_output))
    repeat_exact = bool(
        np.array_equal(step_output, apply_compiled_positive_psf(step, reference))
    )
    diagnostic_sha = _diagnostic(impulse_output, step_output, diagnostic_path)
    gates = config["gates"]
    max_reference_error = float(max(reference_errors))
    max_control_error = float(max(control_errors))
    gate_results = {
        "reference_discrete_mtf": max_reference_error
        <= float(gates["reference_discrete_vs_analytic_max_error"]),
        "kernel_normalization": all(
            abs(float(np.sum(component.kernel_1d)) - 1.0)
            <= float(gates["kernel_sum_absolute_error"])
            for channel in reference
            for component in channel
        ),
        "constant_preservation": constant_error <= float(gates["constant_max_error"]),
        "impulse_nonnegative": float(np.min(impulse_output))
        >= float(gates["impulse_negative_min"]),
        "impulse_mass": bool(
            np.all(np.abs(impulse_sums - 1.0) <= float(gates["kernel_sum_absolute_error"]))
        ),
        "step_no_undershoot": float(max(0.0, -np.min(step_output)))
        <= float(gates["step_undershoot_max"]),
        "step_no_overshoot": float(max(0.0, np.max(step_output) - 1.0))
        <= float(gates["step_overshoot_max"]),
        "measured_frequency_channel_order": bool(
            np.all(
                reference_matrix[0] + float(gates["measured_frequency_channel_order_tolerance"])
                >= reference_matrix[1]
            )
            and np.all(
                reference_matrix[1] + float(gates["measured_frequency_channel_order_tolerance"])
                >= reference_matrix[2]
            )
        ),
        "row_partition_byte_exact": all(partition_exact.values()),
        "repeat_byte_exact": repeat_exact,
        "direct_4000dpi_warning_detected": max_control_error
        > float(gates["direct_4000dpi_warning_error"]),
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash_file(root / "configs/u6_p5k_measured_mtf_kernel_conformance_v1.json"),
        "parent_bundle_sha256": _hash_file(bundle_path),
        "parent_bundle_id": bundle["bundle_id"],
        "reference_dpi": compiler["reference_dpi"],
        "reference_pixel_pitch_um": reference_pitch,
        "direct_control_dpi": compiler["direct_product_dpi_control"],
        "direct_control_pixel_pitch_um": control_pitch,
        "required_reference_halo_px": required_compiled_psf_halo(reference),
        "response": response_rows,
        "reference_max_error": max_reference_error,
        "direct_4000dpi_max_error": max_control_error,
        "constant_max_error": constant_error,
        "impulse_channel_sums": impulse_sums.tolist(),
        "impulse_min": float(np.min(impulse_output)),
        "step_min": float(np.min(step_output)),
        "step_max": float(np.max(step_output)),
        "row_partition_exact": partition_exact,
        "repeat_exact": repeat_exact,
        "diagnostic_sha256": diagnostic_sha,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "retain_reference_open_lod_compiler" if passed else "close_discrete_kernel_compiler",
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredMtfKernelError",
    "evaluate_kernel",
    "load_contract",
]
