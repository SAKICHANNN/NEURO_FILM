"""Frozen U6.P9G gate-weave physical-domain ordering audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.film_physics.gate_weave import GateWeaveProfile, generate_gate_weave
from src.film_physics.gate_weave_sampling import sample_padded_translation


class GateWeavePhysicalOrderError(RuntimeError):
    """Raised when P9G inputs or physical ordering drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateWeavePhysicalOrderError("contract must be an object")
    return payload


def _chart(
    *,
    output_shape: tuple[int, int],
    padding_yx: tuple[int, int],
    minimum: float,
    maximum: float,
    period_x: int,
    period_y: int,
) -> np.ndarray:
    height = output_shape[0] + 2 * padding_yx[0]
    width = output_shape[1] + 2 * padding_yx[1]
    yy, xx = np.mgrid[:height, :width]
    exponent = (
        0.35 * xx / (width - 1)
        + 0.25 * yy / (height - 1)
        + 0.40 * ((xx // period_x + yy // period_y) % 2)
    )
    base = minimum * np.power(maximum / minimum, exponent)
    return np.stack(
        (
            base,
            np.clip(base * (0.65 + 0.35 * np.square(np.sin(xx * 0.11))), minimum, maximum),
            np.clip(base * (0.70 + 0.30 * np.square(np.cos(yy * 0.09))), minimum, maximum),
        ),
        axis=-1,
    )


def _render_frames(
    *,
    exposure: np.ndarray,
    developed_scan: np.ndarray,
    offsets_yx: np.ndarray,
    output_shape: tuple[int, int],
    padding_yx: tuple[int, int],
    operator: Any,
) -> tuple[np.ndarray, np.ndarray]:
    correct: list[np.ndarray] = []
    wrong: list[np.ndarray] = []
    for offset_y, offset_x in offsets_yx:
        correct.append(
            sample_padded_translation(
                developed_scan,
                output_shape=output_shape,
                padding_yx=padding_yx,
                offset_yx=(float(offset_y), float(offset_x)),
                interpolation="bilinear",
            )
        )
        sampled_exposure = sample_padded_translation(
            exposure,
            output_shape=output_shape,
            padding_yx=padding_yx,
            offset_yx=(float(offset_y), float(offset_x)),
            interpolation="bilinear",
        )
        wrong.append(np.power(10.0, -operator.apply(sampled_exposure)))
    return np.stack(correct), np.stack(wrong)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9g_gate_weave_physical_order_contract.v1":
        raise GateWeavePhysicalOrderError("unsupported P9G contract")
    parents = contract["parents"]
    bindings = (
        ("p9b_evidence", "p9b_evidence_sha256"),
        ("p9f_evidence", "p9f_evidence_sha256"),
        ("gate_weave_contract", "gate_weave_contract_sha256"),
        ("sensitometry_config", "sensitometry_config_sha256"),
    )
    parent_hashes_exact = all(
        _sha(root / parents[path_key]) == parents[hash_key]
        for path_key, hash_key in bindings
    )
    if not parent_hashes_exact:
        raise GateWeavePhysicalOrderError("parent artifact hash mismatch")
    p9b = json.loads((root / parents["p9b_evidence"]).read_text(encoding="utf-8"))
    p9f = json.loads((root / parents["p9f_evidence"]).read_text(encoding="utf-8"))
    if p9b.get("results", {}).get("automatic_pass") is not True:
        raise GateWeavePhysicalOrderError("P9B does not admit P9G")
    if p9f.get("automatic_pass") is not True:
        raise GateWeavePhysicalOrderError("P9F does not admit P9G")

    gate_payload = json.loads(
        (root / parents["gate_weave_contract"]).read_text(encoding="utf-8")
    )
    profile_payload = {
        key: value for key, value in gate_payload["profile"].items() if key != "schema"
    }
    profile = GateWeaveProfile(**profile_payload)
    full_trajectory = generate_gate_weave(
        profile, int(gate_payload["experiment"]["frame_count"])
    )
    experiment = contract["experiment"]
    frame_start = int(experiment["frame_start"])
    frame_stop = frame_start + int(experiment["frame_count"])
    offsets = np.stack(
        (
            full_trajectory.y_pixels[frame_start:frame_stop],
            full_trajectory.x_pixels[frame_start:frame_stop],
        ),
        axis=1,
    )
    output_shape = (int(experiment["output_height"]), int(experiment["output_width"]))
    padding_yx = (profile.padding_y_pixels, profile.padding_x_pixels)
    exposure = _chart(
        output_shape=output_shape,
        padding_yx=padding_yx,
        minimum=float(experiment["minimum_layer_exposure"]),
        maximum=float(experiment["maximum_layer_exposure"]),
        period_x=int(experiment["checker_period_x"]),
        period_y=int(experiment["checker_period_y"]),
    )
    operator = build_operator(
        json.loads((root / parents["sensitometry_config"]).read_text(encoding="utf-8"))
    )
    developed_scan = np.power(10.0, -operator.apply(exposure))
    correct, wrong = _render_frames(
        exposure=exposure,
        developed_scan=developed_scan,
        offsets_yx=offsets,
        output_shape=output_shape,
        padding_yx=padding_yx,
        operator=operator,
    )
    replay, _ = _render_frames(
        exposure=exposure,
        developed_scan=developed_scan,
        offsets_yx=offsets,
        output_shape=output_shape,
        padding_yx=padding_yx,
        operator=operator,
    )
    replay_exact = np.array_equal(correct, replay)

    parts: list[np.ndarray] = []
    start = 0
    for count in experiment["frame_partition_counts"]:
        stop = start + int(count)
        part, _ = _render_frames(
            exposure=exposure,
            developed_scan=developed_scan,
            offsets_yx=offsets[start:stop],
            output_shape=output_shape,
            padding_yx=padding_yx,
            operator=operator,
        )
        parts.append(part)
        start = stop
    if start != offsets.shape[0]:
        raise GateWeavePhysicalOrderError("frame partitions do not cover window")
    partition_exact = np.array_equal(np.concatenate(parts), correct)

    zero_correct, zero_wrong = _render_frames(
        exposure=exposure,
        developed_scan=developed_scan,
        offsets_yx=np.zeros((1, 2), dtype=np.float64),
        output_shape=output_shape,
        padding_yx=padding_yx,
        operator=operator,
    )
    difference = np.abs(correct - wrong)
    frame_rmse = np.sqrt(np.mean(np.square(difference), axis=(1, 2, 3)))
    frame_p95 = np.quantile(difference, 0.95, axis=(1, 2, 3))
    frame_maximum = np.max(difference, axis=(1, 2, 3))
    finite = bool(np.all(np.isfinite(correct)) and np.all(np.isfinite(wrong)))
    output_minimum = float(min(np.min(correct), np.min(wrong)))
    output_maximum = float(max(np.max(correct), np.max(wrong)))
    hard_clip_count = int(
        np.count_nonzero((correct <= 0.0) | (correct >= 1.0))
        + np.count_nonzero((wrong <= 0.0) | (wrong >= 1.0))
    )
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "correct_order_replay_byte_exact": replay_exact,
        "correct_order_partition_byte_exact": partition_exact,
        "zero_offset_orders_byte_exact": bool(np.array_equal(zero_correct, zero_wrong)),
        "minimum_frame_rmse": float(np.min(frame_rmse)),
        "median_frame_rmse": float(np.median(frame_rmse)),
        "p95_frame_rmse": float(np.quantile(frame_rmse, 0.95)),
        "median_frame_p95_absolute_error": float(np.median(frame_p95)),
        "p95_frame_absolute_error": float(np.quantile(frame_p95, 0.95)),
        "p95_frame_maximum_error": float(np.quantile(frame_maximum, 0.95)),
        "all_values_finite": finite,
        "output_minimum": output_minimum,
        "output_maximum": output_maximum,
        "hard_clip_count": hard_clip_count,
        "display_rgb_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "correct_order_replay_byte_exact": replay_exact
        is gates["correct_order_replay_byte_exact"],
        "correct_order_partition_byte_exact": partition_exact
        is gates["correct_order_partition_byte_exact"],
        "zero_offset_orders_byte_exact": measurements["zero_offset_orders_byte_exact"]
        is gates["zero_offset_orders_byte_exact"],
        "minimum_frame_rmse": measurements["minimum_frame_rmse"]
        >= gates["minimum_frame_rmse"],
        "minimum_median_frame_rmse": measurements["median_frame_rmse"]
        >= gates["minimum_median_frame_rmse"],
        "minimum_p95_frame_absolute_error": measurements["p95_frame_absolute_error"]
        >= gates["minimum_p95_frame_absolute_error"],
        "minimum_p95_frame_maximum_error": measurements["p95_frame_maximum_error"]
        >= gates["minimum_p95_frame_maximum_error"],
        "all_values_finite": finite is gates["all_values_finite"],
        "output_range_minimum": output_minimum >= gates["output_range_minimum"],
        "output_range_maximum": output_maximum <= gates["output_range_maximum"],
        "hard_clip_count_maximum": hard_clip_count <= gates["hard_clip_count_maximum"],
        "display_rgb_transform_count_zero": gates["display_rgb_transform_count_zero"]
        is True,
    }
    if set(gate_results) != set(gates):
        raise GateWeavePhysicalOrderError("P9G gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9g_gate_weave_physical_order_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9g_gate_weave_physical_order_v1.json"),
        "frame_count": int(offsets.shape[0]),
        "output_shape": list(output_shape),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
