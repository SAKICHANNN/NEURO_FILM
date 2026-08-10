"""Frozen U6.P9J emulsion-grain and gate-weave ordering audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.gate_weave import GateWeaveProfile, generate_gate_weave
from src.film_physics.gate_weave_sampling import sample_padded_translation
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.temporal_grain import (
    baseline_scan_from_log_exposure,
    build_temporal_reference_exposure,
    render_temporal_typed_thomas_frame,
)


class GrainGateWeaveOrderError(RuntimeError):
    """Raised when the P9J contract or bound parents drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GrainGateWeaveOrderError("contract must be an object")
    return payload


def _bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise GrainGateWeaveOrderError(f"bound artifact drift: {path}")
    return load_contract(path)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9j_grain_gate_weave_order_contract.v1":
        raise GrainGateWeaveOrderError("unsupported P9J contract")
    parents = contract["parents"]
    bindings = (
        ("p9h_contract", "p9h_contract_sha256"),
        ("p9h_evidence", "p9h_evidence_sha256"),
        ("p9a_contract", "p9a_contract_sha256"),
        ("p9d_contract", "p9d_contract_sha256"),
    )
    parent_hashes_exact = all(
        _sha(root / parents[path_key]) == parents[sha_key]
        for path_key, sha_key in bindings
    )
    if not parent_hashes_exact:
        raise GrainGateWeaveOrderError("parent hash mismatch")
    if load_contract(root / parents["p9h_evidence"])["automatic_pass"] is not True:
        raise GrainGateWeaveOrderError("P9H does not admit P9J")
    p9a = load_contract(root / parents["p9a_contract"])
    p9d = load_contract(root / parents["p9d_contract"])
    p9d_parents = p9d["parents"]
    grain_profile = DensityConditionedThomasProfile.from_dict(
        _bound(root, p9d_parents["p4bw_bundle"])
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        _bound(root, p9d_parents["p2q_bundle"])["prior"]
    )
    p9d_experiment = p9d["experiment"]
    shape = tuple(int(value) for value in p9d_experiment["field_shape"])
    exposure, relative_log = build_temporal_reference_exposure(
        shape,
        prior,
        minimum_fraction=float(p9d_experiment["normalized_exposure_fraction_minimum"]),
        maximum_fraction=float(p9d_experiment["normalized_exposure_fraction_maximum"]),
        pixel_pitch_um=grain_profile.sample_pitch_micrometres,
    )
    baseline = baseline_scan_from_log_exposure(prior, relative_log)
    weave_profile = GateWeaveProfile(
        **{key: value for key, value in p9a["profile"].items() if key != "schema"}
    )
    weave = generate_gate_weave(weave_profile, int(p9a["experiment"]["frame_count"]))
    experiment = contract["experiment"]
    frame_count = int(experiment["frame_count"])
    frames = list(range(frame_count))
    start = int(experiment["trajectory_frame_start"])
    padding_yx = (weave_profile.padding_y_pixels, weave_profile.padding_x_pixels)
    output_shape = (
        shape[0] - 2 * padding_yx[0],
        shape[1] - 2 * padding_yx[1],
    )
    center = baseline[
        padding_yx[0] : padding_yx[0] + output_shape[0],
        padding_yx[1] : padding_yx[1] + output_shape[1],
    ]

    def render(
        frame: int, offset: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grain_scan = render_temporal_typed_thomas_frame(
            exposure,
            grain_profile,
            prior,
            profile_sha256=p9d_experiment["profile_sha256"],
            seed=int(p9d_experiment["seed"]),
            frame=frame,
            scanner_profile_id=p9d_experiment["scanner_profile_id"],
            canonical_receipt_row_block_height=int(
                p9d_experiment["canonical_receipt_row_block_height"]
            ),
        ).result.scan_linear.values
        residual_center = (
            grain_scan[
                padding_yx[0] : padding_yx[0] + output_shape[0],
                padding_yx[1] : padding_yx[1] + output_shape[1],
            ]
            - center
        )
        correct = sample_padded_translation(
            grain_scan,
            output_shape=output_shape,
            padding_yx=padding_yx,
            offset_yx=offset,
            interpolation="bilinear",
        )
        smooth_shifted = sample_padded_translation(
            baseline,
            output_shape=output_shape,
            padding_yx=padding_yx,
            offset_yx=offset,
            interpolation="bilinear",
        )
        screen_control = smooth_shifted + residual_center
        return correct, screen_control, smooth_shifted

    offsets = {
        frame: (
            float(weave.y_pixels[start + frame]),
            float(weave.x_pixels[start + frame]),
        )
        for frame in frames
    }

    def render_order(
        order: list[int],
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return {frame: render(frame, offsets[frame]) for frame in order}

    full = render_order(frames)
    replay = render_order(frames)
    reverse = render_order(list(reversed(frames)))
    replay_exact = all(
        all(
            np.array_equal(left, right)
            for left, right in zip(full[frame], replay[frame], strict=True)
        )
        for frame in frames
    )
    reverse_exact = all(
        all(
            np.array_equal(left, right)
            for left, right in zip(full[frame], reverse[frame], strict=True)
        )
        for frame in frames
    )
    partition_rendered: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cursor = 0
    for count in experiment["frame_partition_counts"]:
        selected = frames[cursor : cursor + int(count)]
        partition_rendered.update(render_order(selected))
        cursor += int(count)
    if cursor != frame_count:
        raise GrainGateWeaveOrderError("frame partitions do not cover sequence")
    partition_exact = all(
        all(
            np.array_equal(left, right)
            for left, right in zip(full[frame], partition_rendered[frame], strict=True)
        )
        for frame in frames
    )
    zero_errors = []
    rmse = []
    absolute_differences = []
    different = []
    correct_frames = []
    smooth_frames = []
    for frame in frames:
        zero_correct, zero_control, _ = render(frame, (0.0, 0.0))
        zero_errors.append(float(np.max(np.abs(zero_correct - zero_control))))
        correct, control, smooth = full[frame]
        delta = correct - control
        rmse.append(float(np.sqrt(np.mean(np.square(delta)))))
        absolute_differences.append(np.abs(delta).ravel())
        different.append(not np.array_equal(correct, control))
        correct_frames.append(correct)
        smooth_frames.append(smooth)
    correct_stack = np.stack(correct_frames)
    smooth_stack = np.stack(smooth_frames)
    absolute = np.concatenate(absolute_differences)
    epsilon = 1.0 / 65535.0
    new_boundary = ((correct_stack <= epsilon) | (correct_stack >= 1.0 - epsilon)) & ~(
        (smooth_stack <= epsilon) | (smooth_stack >= 1.0 - epsilon)
    )
    finite = bool(np.all(np.isfinite(correct_stack)))
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "replay_byte_exact": replay_exact,
        "reverse_frame_order_byte_exact": reverse_exact,
        "frame_partition_byte_exact": partition_exact,
        "zero_offset_order_maximum_error": max(zero_errors),
        "nonzero_offset_different_frame_fraction": float(np.mean(different)),
        "order_difference_median_rmse": float(np.median(rmse)),
        "order_difference_p95_absolute": float(np.quantile(absolute, 0.95)),
        "correct_order_all_values_finite": finite,
        "correct_order_new_boundary_fraction": float(np.mean(new_boundary)),
        "display_rgb_effect_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "replay_byte_exact": replay_exact is gates["replay_byte_exact"],
        "reverse_frame_order_byte_exact": reverse_exact
        is gates["reverse_frame_order_byte_exact"],
        "frame_partition_byte_exact": partition_exact
        is gates["frame_partition_byte_exact"],
        "zero_offset_order_error_maximum": measurements[
            "zero_offset_order_maximum_error"
        ]
        <= gates["zero_offset_order_error_maximum"],
        "minimum_nonzero_offset_different_frame_fraction": measurements[
            "nonzero_offset_different_frame_fraction"
        ]
        >= gates["minimum_nonzero_offset_different_frame_fraction"],
        "minimum_order_difference_median_rmse": measurements[
            "order_difference_median_rmse"
        ]
        >= gates["minimum_order_difference_median_rmse"],
        "minimum_order_difference_p95_absolute": measurements[
            "order_difference_p95_absolute"
        ]
        >= gates["minimum_order_difference_p95_absolute"],
        "correct_order_all_values_finite": finite
        is gates["correct_order_all_values_finite"],
        "correct_order_new_boundary_fraction_maximum": measurements[
            "correct_order_new_boundary_fraction"
        ]
        <= gates["correct_order_new_boundary_fraction_maximum"],
        "display_rgb_effect_count_zero": (measurements["display_rgb_effect_count"] == 0)
        is gates["display_rgb_effect_count_zero"],
    }
    if set(gate_results) != set(gates):
        raise GrainGateWeaveOrderError("P9J gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9j_grain_gate_weave_order_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9j_grain_gate_weave_order_v1.json"),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
