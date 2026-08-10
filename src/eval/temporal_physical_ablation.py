"""Frozen U6.P9H joint temporal physical-stage ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.contracts import PhysicalDomainArray
from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.gate_weave import GateWeaveProfile, generate_gate_weave
from src.film_physics.gate_weave_sampling import sample_padded_translation
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.temporal_exposure import (
    TemporalExposureProfile,
    generate_temporal_exposure,
)
from src.film_physics.temporal_grain import (
    baseline_scan_from_log_exposure,
    build_temporal_reference_exposure,
    render_temporal_typed_thomas_frame,
)


class TemporalPhysicalAblationError(RuntimeError):
    """Raised when the P9H contract or parent identities drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TemporalPhysicalAblationError(f"expected object: {path}")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    return _load(path)


def _bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise TemporalPhysicalAblationError(f"bound artifact drift: {path}")
    return _load(path)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != "neuro_film.u6_p9h_temporal_physical_ablation_contract.v1":
        raise TemporalPhysicalAblationError("unsupported P9H contract")
    parents = contract["parents"]
    path_hash_keys = (
        ("p9d_contract", "p9d_contract_sha256"),
        ("p9d_evidence", "p9d_evidence_sha256"),
        ("p9e_evidence", "p9e_evidence_sha256"),
        ("p9e_contract", "p9e_contract_sha256"),
        ("p9g_evidence", "p9g_evidence_sha256"),
        ("p9a_contract", "p9a_contract_sha256"),
    )
    parent_hashes_exact = all(
        _sha(root / parents[path_key]) == parents[hash_key]
        for path_key, hash_key in path_hash_keys
    )
    if not parent_hashes_exact:
        raise TemporalPhysicalAblationError("parent hash mismatch")
    for key in ("p9d_evidence", "p9e_evidence", "p9g_evidence"):
        evidence = _load(root / parents[key])
        passed = evidence.get("automatic_pass")
        if passed is None:
            passed = evidence.get("results", {}).get("automatic_pass")
        if passed is not True:
            raise TemporalPhysicalAblationError(f"{key} does not admit P9H")

    p9d = _load(root / parents["p9d_contract"])
    p9d_parents = p9d["parents"]
    profile = DensityConditionedThomasProfile.from_dict(
        _bound(root, p9d_parents["p4bw_bundle"])
    )
    prior_payload = _bound(root, p9d_parents["p2q_bundle"])
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    p9d_experiment = p9d["experiment"]
    shape = tuple(int(value) for value in p9d_experiment["field_shape"])
    base_exposure, _ = build_temporal_reference_exposure(
        shape,
        prior,
        minimum_fraction=float(p9d_experiment["normalized_exposure_fraction_minimum"]),
        maximum_fraction=float(p9d_experiment["normalized_exposure_fraction_maximum"]),
        pixel_pitch_um=profile.sample_pitch_micrometres,
    )

    p9e = _load(root / parents["p9e_contract"])
    p9e_profile = TemporalExposureProfile(
        **{key: value for key, value in p9e["profile"].items() if key != "schema"}
    )
    exposure_trajectory = generate_temporal_exposure(
        p9e_profile, int(p9e["experiment"]["frame_count"])
    )
    p9a = _load(root / parents["p9a_contract"])
    p9a_profile = GateWeaveProfile(
        **{key: value for key, value in p9a["profile"].items() if key != "schema"}
    )
    weave_trajectory = generate_gate_weave(
        p9a_profile, int(p9a["experiment"]["frame_count"])
    )
    experiment = contract["experiment"]
    start = int(experiment["trajectory_frame_start"])
    frames = list(range(int(experiment["frame_count"])))
    output_shape = (
        shape[0] - 2 * p9a_profile.padding_y_pixels,
        shape[1] - 2 * p9a_profile.padding_x_pixels,
    )
    padding_yx = (p9a_profile.padding_y_pixels, p9a_profile.padding_x_pixels)

    def render(frame: int, *, flicker: bool, grain: bool, weave: bool) -> np.ndarray:
        trajectory_index = start + frame
        multiplier = (
            float(exposure_trajectory.multiplier[trajectory_index]) if flicker else 1.0
        )
        exposure_values = np.ascontiguousarray(base_exposure.values * multiplier)
        exposure = PhysicalDomainArray(
            exposure_values,
            base_exposure.domain,
            base_exposure.unit,
            base_exposure.channels,
            base_exposure.scale,
        )
        if grain:
            scan = render_temporal_typed_thomas_frame(
                exposure,
                profile,
                prior,
                profile_sha256=p9d_experiment["profile_sha256"],
                seed=int(p9d_experiment["seed"]),
                frame=frame,
                scanner_profile_id=p9d_experiment["scanner_profile_id"],
                canonical_receipt_row_block_height=int(
                    p9d_experiment["canonical_receipt_row_block_height"]
                ),
            ).result.scan_linear.values
        else:
            scan = baseline_scan_from_log_exposure(prior, np.log10(exposure_values))
        offset = (
            (
                float(weave_trajectory.y_pixels[trajectory_index]),
                float(weave_trajectory.x_pixels[trajectory_index]),
            )
            if weave
            else (0.0, 0.0)
        )
        return sample_padded_translation(
            scan,
            output_shape=output_shape,
            padding_yx=padding_yx,
            offset_yx=offset,
            interpolation="bilinear",
        )

    def render_sequence(order: list[int], switches: tuple[bool, bool, bool]) -> dict[int, np.ndarray]:
        return {
            frame: render(
                frame,
                flicker=switches[0],
                grain=switches[1],
                weave=switches[2],
            )
            for frame in order
        }

    full = render_sequence(frames, (True, True, True))
    replay = render_sequence(frames, (True, True, True))
    reverse = render_sequence(list(reversed(frames)), (True, True, True))
    replay_exact = all(np.array_equal(full[index], replay[index]) for index in frames)
    reverse_exact = all(np.array_equal(full[index], reverse[index]) for index in frames)
    parts: list[np.ndarray] = []
    part_start = 0
    for count in experiment["frame_partition_counts"]:
        selected = frames[part_start : part_start + int(count)]
        rendered = render_sequence(selected, (True, True, True))
        parts.extend(rendered[index] for index in selected)
        part_start += int(count)
    if part_start != len(frames):
        raise TemporalPhysicalAblationError("frame partitions do not cover sequence")
    partition_exact = all(
        np.array_equal(parts[index], full[index]) for index in frames
    )

    no_flicker = render_sequence(frames, (False, True, True))
    no_grain = render_sequence(frames, (True, False, True))
    no_weave = render_sequence(frames, (True, True, False))
    all_off = render_sequence(frames, (False, False, False))
    full_stack = np.stack([full[index] for index in frames])
    all_off_stack = np.stack([all_off[index] for index in frames])
    deltas = {
        "exposure_flicker": full_stack
        - np.stack([no_flicker[index] for index in frames]),
        "density_grain": full_stack - np.stack([no_grain[index] for index in frames]),
        "gate_weave": full_stack - np.stack([no_weave[index] for index in frames]),
    }
    rms = {
        key: float(np.sqrt(np.mean(np.square(value)))) for key, value in deltas.items()
    }
    correlations = {
        "exposure_flicker__density_grain": _correlation(
            deltas["exposure_flicker"], deltas["density_grain"]
        ),
        "exposure_flicker__gate_weave": _correlation(
            deltas["exposure_flicker"], deltas["gate_weave"]
        ),
        "density_grain__gate_weave": _correlation(
            deltas["density_grain"], deltas["gate_weave"]
        ),
    }
    max_correlation = float(max(abs(value) for value in correlations.values()))
    finite = bool(
        np.all(np.isfinite(full_stack))
        and np.all(np.isfinite(all_off_stack))
        and all(np.all(np.isfinite(value)) for value in deltas.values())
    )
    epsilon = 1.0 / 65535.0
    new_boundary = ((full_stack <= epsilon) | (full_stack >= 1.0 - epsilon)) & ~(
        (all_off_stack <= epsilon) | (all_off_stack >= 1.0 - epsilon)
    )
    new_boundary_fraction = float(np.mean(new_boundary))
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "full_chain_replay_byte_exact": replay_exact,
        "reverse_frame_order_byte_exact": reverse_exact,
        "frame_partition_byte_exact": partition_exact,
        "effect_delta_rms": rms,
        "pairwise_effect_correlations": correlations,
        "maximum_absolute_pairwise_effect_correlation": max_correlation,
        "all_values_finite": finite,
        "new_boundary_fraction": new_boundary_fraction,
        "effect_switch_count": len(deltas),
        "display_rgb_effect_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "full_chain_replay_byte_exact": replay_exact
        is gates["full_chain_replay_byte_exact"],
        "reverse_frame_order_byte_exact": reverse_exact
        is gates["reverse_frame_order_byte_exact"],
        "frame_partition_byte_exact": partition_exact
        is gates["frame_partition_byte_exact"],
        "minimum_exposure_flicker_delta_rms": rms["exposure_flicker"]
        >= gates["minimum_exposure_flicker_delta_rms"],
        "minimum_density_grain_delta_rms": rms["density_grain"]
        >= gates["minimum_density_grain_delta_rms"],
        "minimum_gate_weave_delta_rms": rms["gate_weave"]
        >= gates["minimum_gate_weave_delta_rms"],
        "maximum_absolute_pairwise_effect_correlation": max_correlation
        <= gates["maximum_absolute_pairwise_effect_correlation"],
        "all_values_finite": finite is gates["all_values_finite"],
        "new_boundary_fraction_maximum": new_boundary_fraction
        <= gates["new_boundary_fraction_maximum"],
        "effect_switch_count_exact": len(deltas) == gates["effect_switch_count_exact"],
        "display_rgb_effect_count_zero": gates["display_rgb_effect_count_zero"]
        is True,
    }
    if set(gate_results) != set(gates):
        raise TemporalPhysicalAblationError("P9H gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9h_temporal_physical_ablation_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p9h_temporal_physical_ablation_v1.json"),
        "frame_count": len(frames),
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
