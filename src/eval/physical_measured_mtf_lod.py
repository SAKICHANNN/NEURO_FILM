"""U6.P5L exact target-grid LOD compiler for the measured-MTF reference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.film_physics.measured_mtf import (
    CompiledPsfComponent,
    apply_compiled_positive_psf,
    apply_compiled_positive_psf_row_tiled,
    apply_zero_order_hold_reference,
    channel_psf_from_json,
    compile_channel_psf,
    compile_zero_order_hold_lod,
    required_compiled_psf_halo,
)

SCHEMA = "neuro_film.u6_p5l_measured_mtf_lod_compiler_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p5l_measured_mtf_lod_compiler_report.v1"
LOD_SCHEMA = "neuro_film.measured_positive_psf_lod_bundle.v1"
CHANNELS = ("blue", "green", "red")


class MeasuredMtfLodError(RuntimeError):
    """Raised when the frozen P5L contract or parent evidence drifts."""


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
        raise MeasuredMtfLodError("P5L paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    reference = payload.get("reference", {})
    compiler = payload.get("compiler", {})
    charts = payload.get("charts", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("bundle_sha256")
        != "b2d0ff39743ba3ec21a8d033fe38f3ebf6e7360e5a75ea6b838dc67b4b856270"
        or parent.get("p5k_report_sha256")
        != "283bcf9ccb550f040a1e6b6f331fd45ddf290ff4dd4ecb765605bfb286980b3c"
        or parent.get("p5k_stable_evidence_id")
        != "0e8c31e16ab8165cb59a21eaa884619167449fec8e744a9863de1d3357cfb8c7"
        or reference.get("target_dpi") != 4000.0
        or reference.get("reference_dpi") != 16000.0
        or reference.get("integer_scale") != 4
        or reference.get("input_reconstruction") != "zero_order_hold_repeat"
        or reference.get("downsample") != "nonoverlapping 4x4 area mean"
        or compiler.get("target_halo_max_px") != 8
        or not compiler.get("negative_taps_forbidden")
        or compiler.get("renormalization_after_derivation")
        or charts.get("row_partitions") != [7, 17, 31]
        or gates.get("reference_vs_lod_max_error") != 1e-12
        or gates.get("lod_vs_naive_error_reduction_min") != 0.99
        or not gates.get("row_partition_byte_exact")
        or not gates.get("repeat_byte_exact")
    ):
        raise MeasuredMtfLodError("P5L frozen contract drift")
    for value in (parent.get("bundle", ""), parent.get("p5k_report", "")):
        _relative_path(value)
    return payload


def _compile_rgb(
    bundle: Mapping[str, Any], *, pixel_pitch_um: float, truncate_sigma: float
) -> tuple[Any, ...]:
    return tuple(
        compile_channel_psf(
            channel_psf_from_json(bundle["channels"][channel]),
            pixel_pitch_um=pixel_pitch_um,
            truncate_sigma=truncate_sigma,
        )
        for channel in CHANNELS
    )


def _serialize_lod(
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    lod: Sequence[Sequence[CompiledPsfComponent]],
) -> dict[str, Any]:
    core = {
        "schema": LOD_SCHEMA,
        "parent_bundle_id": bundle["bundle_id"],
        "target_dpi": config["reference"]["target_dpi"],
        "reference_dpi": config["reference"]["reference_dpi"],
        "input_reconstruction": config["reference"]["input_reconstruction"],
        "downsample": config["reference"]["downsample"],
        "boundary_mode": config["reference"]["boundary_mode"],
        "channels": {
            channel: [
                {"weight": component.weight, "kernel_1d": component.kernel_1d.tolist()}
                for component in lod[index]
            ]
            for index, channel in enumerate(CHANNELS)
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "lod_bundle_id": hashlib.sha256(_canonical_json(core)).hexdigest()}


def _charts(config: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    for height, width in config["charts"]["shapes"]:
        impulse = np.zeros((height, width, 3), dtype=np.float64)
        impulse[height // 2, width // 2, :] = 1.0
        rows.append((f"impulse-{height}x{width}", impulse))
        step = np.zeros_like(impulse)
        step[:, width // 2 :, :] = 1.0
        rows.append((f"step-{height}x{width}", step))
        y, x = np.mgrid[:height, :width]
        checker = (((x // 3) + (y // 5)) % 2).astype(np.float64)
        rows.append(
            (f"checker-{height}x{width}", np.repeat(checker[..., None], 3, axis=2))
        )
        for seed in config["charts"]["random_seeds"]:
            random = np.random.default_rng(int(seed) + height * 1000 + width).random(
                (height, width, 3)
            )
            rows.append((f"random-{seed}-{height}x{width}", random))
    return rows


def _diagnostic(
    reference: np.ndarray, naive: np.ndarray, lod: np.ndarray, output: Path
) -> str:
    difference = np.clip(np.abs(reference - naive) * 8.0, 0.0, 1.0)
    canvas = np.concatenate((reference, naive, difference, lod), axis=1)
    encoded = np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="RGB").save(output)
    return _hash_file(output)


def compile_and_evaluate_lod(
    config: Mapping[str, Any], root: Path, *, diagnostic_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path = root / _relative_path(str(config["parent"]["bundle"]))
    p5k_path = root / _relative_path(str(config["parent"]["p5k_report"]))
    if _hash_file(bundle_path) != config["parent"]["bundle_sha256"]:
        raise MeasuredMtfLodError("P5L parent bundle hash mismatch")
    if _hash_file(p5k_path) != config["parent"]["p5k_report_sha256"]:
        raise MeasuredMtfLodError("P5L parent report hash mismatch")
    p5k = json.loads(p5k_path.read_text(encoding="utf-8"))
    if p5k.get("stable_evidence_id") != config["parent"]["p5k_stable_evidence_id"]:
        raise MeasuredMtfLodError("P5L parent evidence identity mismatch")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    scale = int(config["reference"]["integer_scale"])
    reference_pitch = 25_400.0 / float(config["reference"]["reference_dpi"])
    target_pitch = 25_400.0 / float(config["reference"]["target_dpi"])
    reference = _compile_rgb(bundle, pixel_pitch_um=reference_pitch, truncate_sigma=5.0)
    naive = _compile_rgb(bundle, pixel_pitch_um=target_pitch, truncate_sigma=5.0)
    lod = compile_zero_order_hold_lod(reference, scale=scale)
    lod_bundle = _serialize_lod(config, bundle, lod)

    chart_rows = []
    lod_errors = []
    naive_errors = []
    diagnostic_values: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for name, values in _charts(config):
        reference_output = apply_zero_order_hold_reference(
            values, reference, scale=scale
        )
        lod_output = apply_compiled_positive_psf(values, lod)
        naive_output = apply_compiled_positive_psf(values, naive)
        lod_error = np.abs(reference_output - lod_output)
        naive_error = np.abs(reference_output - naive_output)
        lod_errors.extend(lod_error.ravel().tolist())
        naive_errors.extend(naive_error.ravel().tolist())
        chart_rows.append(
            {
                "name": name,
                "lod_max_error": float(np.max(lod_error)),
                "lod_mean_error": float(np.mean(lod_error)),
                "naive_max_error": float(np.max(naive_error)),
                "naive_mean_error": float(np.mean(naive_error)),
            }
        )
        if diagnostic_values is None and name.startswith("checker-64x97"):
            diagnostic_values = (reference_output, naive_output, lod_output)
    if diagnostic_values is None:
        raise MeasuredMtfLodError("P5L diagnostic chart missing")

    constants_error = 0.0
    for value in config["charts"]["constant_values"]:
        field = np.full((37, 53, 3), float(value), dtype=np.float64)
        constants_error = max(
            constants_error,
            float(np.max(np.abs(apply_compiled_positive_psf(field, lod) - field))),
        )
    step = np.zeros((129, 191, 3), dtype=np.float64)
    step[:, 95:, :] = 1.0
    step_output = apply_compiled_positive_psf(step, lod)
    partition_exact = {
        str(rows): bool(
            np.array_equal(
                step_output,
                apply_compiled_positive_psf_row_tiled(step, lod, tile_rows=int(rows)),
            )
        )
        for rows in config["charts"]["row_partitions"]
    }
    repeat_exact = bool(
        np.array_equal(step_output, apply_compiled_positive_psf(step, lod))
    )
    diagnostic_sha = _diagnostic(*diagnostic_values, diagnostic_path)
    lod_max = float(max(lod_errors))
    lod_mean = float(np.mean(lod_errors))
    naive_mean = float(np.mean(naive_errors))
    reduction = 1.0 - lod_mean / naive_mean
    gates = config["gates"]
    gate_results = {
        "reference_vs_lod_max": lod_max <= float(gates["reference_vs_lod_max_error"]),
        "reference_vs_lod_mean": lod_mean
        <= float(gates["reference_vs_lod_mean_error"]),
        "beats_naive_direct": reduction
        >= float(gates["lod_vs_naive_error_reduction_min"]),
        "kernel_normalization": all(
            abs(float(np.sum(component.kernel_1d)) - 1.0)
            <= float(gates["kernel_sum_absolute_error"])
            for channel in lod
            for component in channel
        ),
        "kernel_symmetry": all(
            float(np.max(np.abs(component.kernel_1d - component.kernel_1d[::-1])))
            <= float(gates["kernel_symmetry_max_error"])
            for channel in lod
            for component in channel
        ),
        "kernel_nonnegative": all(
            float(np.min(component.kernel_1d)) >= float(gates["kernel_minimum"])
            for channel in lod
            for component in channel
        ),
        "target_halo": required_compiled_psf_halo(lod)
        <= int(config["compiler"]["target_halo_max_px"]),
        "constant_preservation": constants_error <= float(gates["constant_max_error"]),
        "step_no_undershoot": float(max(0.0, -np.min(step_output)))
        <= float(gates["step_undershoot_max"]),
        "step_no_overshoot": float(max(0.0, np.max(step_output) - 1.0))
        <= float(gates["step_overshoot_max"]),
        "row_partition_byte_exact": all(partition_exact.values()),
        "repeat_byte_exact": repeat_exact,
    }
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash_file(
            root / "configs/u6_p5l_measured_mtf_lod_compiler_v1.json"
        ),
        "parent_bundle_id": bundle["bundle_id"],
        "parent_p5k_evidence_id": p5k["stable_evidence_id"],
        "lod_bundle_id": lod_bundle["lod_bundle_id"],
        "reference_scale": scale,
        "target_halo_px": required_compiled_psf_halo(lod),
        "component_radii_px": {
            channel: [component.radius for component in lod[index]]
            for index, channel in enumerate(CHANNELS)
        },
        "chart_rows": chart_rows,
        "reference_vs_lod_max_error": lod_max,
        "reference_vs_lod_mean_error": lod_mean,
        "reference_vs_naive_mean_error": naive_mean,
        "lod_vs_naive_error_reduction": reduction,
        "constant_max_error": constants_error,
        "step_min": float(np.min(step_output)),
        "step_max": float(np.max(step_output)),
        "row_partition_exact": partition_exact,
        "repeat_exact": repeat_exact,
        "diagnostic_sha256": diagnostic_sha,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "open_joint_spatial_budget_audit"
        if passed
        else "close_target_grid_lod",
        "claim_ceiling": config["claim_ceiling"],
    }
    return lod_bundle, report


__all__ = [
    "LOD_SCHEMA",
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredMtfLodError",
    "compile_and_evaluate_lod",
    "load_contract",
]
