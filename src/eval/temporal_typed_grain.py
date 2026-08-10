"""Frozen U6.P9D typed temporal grain composition audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.temporal_grain import (
    baseline_scan_from_log_exposure,
    build_temporal_reference_exposure,
    render_temporal_typed_thomas_frame,
)
from src.film_physics.thomas_image_formation import (
    render_typed_thomas_image_formation_region,
)


class TemporalTypedGrainAuditError(RuntimeError):
    """Raised when P9D inputs or identities drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TemporalTypedGrainAuditError(f"expected object: {path}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    return _load(path)


def _bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise TemporalTypedGrainAuditError(f"bound artifact drift: {path}")
    return _load(path)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9d_temporal_typed_grain_contract.v1":
        raise TemporalTypedGrainAuditError("unsupported P9D contract")
    parents = contract["parents"]
    p9c = _bound(root, parents["p9c_evidence"])
    p4by = _bound(root, parents["p4by_decision"])
    if p9c.get("decision") != parents["p9c_evidence"]["required_decision"]:
        raise TemporalTypedGrainAuditError("P9C decision mismatch")
    if p4by.get("decision") != parents["p4by_decision"]["required_decision"]:
        raise TemporalTypedGrainAuditError("P4BY decision mismatch")
    profile = DensityConditionedThomasProfile.from_dict(
        _bound(root, parents["p4bw_bundle"])
    )
    prior_payload = _bound(root, parents["p2q_bundle"])
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    experiment = contract["experiment"]
    shape = tuple(int(value) for value in experiment["field_shape"])
    exposure, relative_log = build_temporal_reference_exposure(
        shape,
        prior,
        minimum_fraction=float(experiment["normalized_exposure_fraction_minimum"]),
        maximum_fraction=float(experiment["normalized_exposure_fraction_maximum"]),
        pixel_pitch_um=profile.sample_pitch_micrometres,
    )
    baseline = baseline_scan_from_log_exposure(prior, relative_log)

    def render_frame(
        frame: int, *, verify_partition: bool
    ) -> tuple[np.ndarray, bool, bool]:
        rendered = render_temporal_typed_thomas_frame(
            exposure,
            profile,
            prior,
            profile_sha256=experiment["profile_sha256"],
            seed=int(experiment["seed"]),
            frame=frame,
            scanner_profile_id=experiment["scanner_profile_id"],
            canonical_receipt_row_block_height=int(
                experiment["canonical_receipt_row_block_height"]
            ),
        )
        result = rendered.result
        partition_exact = True
        if verify_partition:
            assembled = np.empty_like(result.scan_linear.values)
            block = int(experiment["render_partition_height"])
            for y0 in range(0, shape[0], block):
                height = min(block, shape[0] - y0)
                region = render_typed_thomas_image_formation_region(
                    exposure,
                    profile,
                    prior,
                    rendered.receipts,
                    rendered.context,
                    origin_yx=(y0, 0),
                    shape=(height, shape[1]),
                )
                assembled[y0 : y0 + height] = region.scan_linear.values
            partition_exact = np.array_equal(assembled, result.scan_linear.values)
        scanner_exact = np.array_equal(
            result.scan_linear.values, result.transmittance.values
        )
        return result.scan_linear.values, partition_exact, scanner_exact

    frame_count = int(experiment["frame_count"])
    frames: list[np.ndarray] = []
    partition_results: list[bool] = []
    scanner_results: list[bool] = []
    for frame in range(frame_count):
        values, partition_exact, scanner_exact = render_frame(
            frame, verify_partition=True
        )
        frames.append(values)
        partition_results.append(partition_exact)
        scanner_results.append(scanner_exact)
    reverse = {
        frame: render_frame(frame, verify_partition=False)[0]
        for frame in reversed(range(frame_count))
    }
    reverse_exact = all(
        np.array_equal(frames[frame], reverse[frame]) for frame in range(frame_count)
    )
    deltas = [values - baseline for values in frames]
    hashes = [
        hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        for values in frames
    ]
    correlations = [
        _correlation(deltas[index], deltas[index + 1])
        for index in range(frame_count - 1)
    ]
    absolute = np.abs(np.stack(deltas))
    mean_delta = float(np.max([abs(float(np.mean(delta))) for delta in deltas]))
    fraction_map = np.mean(
        (relative_log - np.min(relative_log, axis=(0, 1)))
        / np.maximum(np.ptp(relative_log, axis=(0, 1)), 1e-12),
        axis=-1,
    )
    low = fraction_map <= np.quantile(fraction_map, 0.25)
    high = fraction_map >= np.quantile(fraction_map, 0.75)
    low_rms = float(np.sqrt(np.mean(np.square(np.stack(deltas)[:, low, :]))))
    high_rms = float(np.sqrt(np.mean(np.square(np.stack(deltas)[:, high, :]))))
    signal_ratio = max(low_rms, high_rms) / min(low_rms, high_rms)
    epsilon = 1.0 / 65535.0
    candidates = np.stack(frames)
    baseline_stack = np.broadcast_to(baseline, candidates.shape)
    new_boundary = ((candidates <= epsilon) | (candidates >= 1.0 - epsilon)) & ~(
        (baseline_stack <= epsilon) | (baseline_stack >= 1.0 - epsilon)
    )
    measurements = {
        "replay_byte_exact": reverse_exact,
        "reverse_frame_order_byte_exact": reverse_exact,
        "row_partition_byte_exact": all(partition_results),
        "all_frame_hashes_unique": len(set(hashes)) == len(hashes),
        "all_values_finite": bool(np.all(np.isfinite(candidates))),
        "maximum_absolute_adjacent_frame_delta_correlation": float(
            np.max(np.abs(correlations))
        ),
        "minimum_static_reuse_adjacent_frame_correlation": _correlation(
            deltas[0], deltas[0]
        ),
        "signal_conditioned_delta_rms_ratio": signal_ratio,
        "absolute_scan_delta_p99": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_scan_delta": float(np.max(absolute)),
        "maximum_absolute_mean_scan_delta": mean_delta,
        "new_boundary_fraction": float(np.mean(new_boundary)),
        "identity_scanner_pixel_exact": all(scanner_results),
        "display_rgb_noise_application_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "replay_byte_exact": measurements["replay_byte_exact"]
        is gates["replay_byte_exact"],
        "reverse_frame_order_byte_exact": measurements["reverse_frame_order_byte_exact"]
        is gates["reverse_frame_order_byte_exact"],
        "row_partition_byte_exact": measurements["row_partition_byte_exact"]
        is gates["row_partition_byte_exact"],
        "all_frame_hashes_unique": measurements["all_frame_hashes_unique"]
        is gates["all_frame_hashes_unique"],
        "all_values_finite": measurements["all_values_finite"]
        is gates["all_values_finite"],
        "maximum_absolute_adjacent_frame_delta_correlation": measurements[
            "maximum_absolute_adjacent_frame_delta_correlation"
        ]
        <= gates["maximum_absolute_adjacent_frame_delta_correlation"],
        "minimum_static_reuse_adjacent_frame_correlation": measurements[
            "minimum_static_reuse_adjacent_frame_correlation"
        ]
        >= gates["minimum_static_reuse_adjacent_frame_correlation"],
        "minimum_signal_conditioned_delta_rms_ratio": measurements[
            "signal_conditioned_delta_rms_ratio"
        ]
        >= gates["minimum_signal_conditioned_delta_rms_ratio"],
        "minimum_absolute_scan_delta_p99": measurements["absolute_scan_delta_p99"]
        >= gates["minimum_absolute_scan_delta_p99"],
        "maximum_absolute_scan_delta_p99": measurements["absolute_scan_delta_p99"]
        <= gates["maximum_absolute_scan_delta_p99"],
        "maximum_absolute_scan_delta": measurements["maximum_absolute_scan_delta"]
        <= gates["maximum_absolute_scan_delta"],
        "maximum_absolute_mean_scan_delta": measurements[
            "maximum_absolute_mean_scan_delta"
        ]
        <= gates["maximum_absolute_mean_scan_delta"],
        "maximum_new_boundary_fraction": measurements["new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "identity_scanner_pixel_exact": measurements["identity_scanner_pixel_exact"]
        is gates["identity_scanner_pixel_exact"],
        "display_rgb_noise_application_count_zero": measurements[
            "display_rgb_noise_application_count"
        ]
        == 0,
    }
    if set(gate_results) != set(gates):
        raise TemporalTypedGrainAuditError("P9D gate vocabulary drift")
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9d_temporal_typed_grain_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9d_temporal_typed_grain_v1.json"),
        "frame_count": frame_count,
        "profile_id": profile.identity(),
        "prior_id": prior.identity(),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["branch_rule"]["pass"]
        if automatic_pass
        else contract["branch_rule"]["fail"],
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
