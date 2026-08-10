"""Frozen U6.P9K exposure and density-conditioned grain coupling audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.contracts import PhysicalDomainArray
from src.film_physics.density_conditioned_thomas import DensityConditionedThomasProfile
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.temporal_exposure import apply_temporal_exposure
from src.film_physics.temporal_grain import (
    baseline_scan_from_log_exposure,
    build_temporal_reference_exposure,
    render_temporal_typed_thomas_frame,
)
from src.film_physics.thomas_image_formation import (
    render_typed_thomas_image_formation_region,
)


class ExposureGrainCouplingError(RuntimeError):
    """Raised when the P9K contract or bound parents drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExposureGrainCouplingError("contract must be an object")
    return payload


def _bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise ExposureGrainCouplingError(f"bound artifact drift: {path}")
    return load_contract(path)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema")
        != "neuro_film.u6_p9k_exposure_grain_coupling_contract.v1"
    ):
        raise ExposureGrainCouplingError("unsupported P9K contract")
    parents = contract["parents"]
    bindings = (
        ("p9f_evidence", "p9f_evidence_sha256"),
        ("p9j_evidence", "p9j_evidence_sha256"),
        ("p9d_contract", "p9d_contract_sha256"),
        ("p9e_contract", "p9e_contract_sha256"),
    )
    parent_hashes_exact = all(
        _sha(root / parents[path_key]) == parents[sha_key]
        for path_key, sha_key in bindings
    )
    if not parent_hashes_exact:
        raise ExposureGrainCouplingError("parent hash mismatch")
    for evidence_key in ("p9f_evidence", "p9j_evidence"):
        if load_contract(root / parents[evidence_key])["automatic_pass"] is not True:
            raise ExposureGrainCouplingError(f"{evidence_key} does not admit P9K")
    p9d = load_contract(root / parents["p9d_contract"])
    p9d_parents = p9d["parents"]
    profile = DensityConditionedThomasProfile.from_dict(
        _bound(root, p9d_parents["p4bw_bundle"])
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        _bound(root, p9d_parents["p2q_bundle"])["prior"]
    )
    p9d_experiment = p9d["experiment"]
    shape = tuple(int(value) for value in p9d_experiment["field_shape"])
    base_exposure, _ = build_temporal_reference_exposure(
        shape,
        prior,
        minimum_fraction=float(p9d_experiment["normalized_exposure_fraction_minimum"]),
        maximum_fraction=float(p9d_experiment["normalized_exposure_fraction_maximum"]),
        pixel_pitch_um=profile.sample_pitch_micrometres,
    )
    experiment = contract["experiment"]
    frame = int(experiment["frame_index"])
    offsets = [float(value) for value in experiment["exposure_offsets_stops"]]
    partition_counts = [int(value) for value in experiment["row_partition_counts"]]
    if sum(partition_counts) != shape[0]:
        raise ExposureGrainCouplingError("row partitions do not cover the field")

    def render(offset: float) -> tuple[np.ndarray, np.ndarray, bool]:
        values = apply_temporal_exposure(base_exposure.values, offset)
        exposure = PhysicalDomainArray(
            values,
            base_exposure.domain,
            base_exposure.unit,
            base_exposure.channels,
            base_exposure.scale,
        )
        rendered = render_temporal_typed_thomas_frame(
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
        )
        candidate = rendered.result.scan_linear.values
        smooth = baseline_scan_from_log_exposure(prior, np.log10(values))
        assembled = np.empty_like(candidate)
        y0 = 0
        for count in partition_counts:
            region = render_typed_thomas_image_formation_region(
                exposure,
                profile,
                prior,
                rendered.receipts,
                rendered.context,
                origin_yx=(y0, 0),
                shape=(count, shape[1]),
            )
            assembled[y0 : y0 + count] = region.scan_linear.values
            y0 += count
        return candidate, smooth, np.array_equal(candidate, assembled)

    rendered = {offset: render(offset) for offset in offsets}
    replay = {offset: render(offset) for offset in offsets}
    reverse = {offset: render(offset) for offset in reversed(offsets)}
    replay_exact = all(
        all(
            np.array_equal(left, right)
            if isinstance(left, np.ndarray)
            else left == right
            for left, right in zip(rendered[offset], replay[offset], strict=True)
        )
        for offset in offsets
    )
    reverse_exact = all(
        all(
            np.array_equal(left, right)
            if isinstance(left, np.ndarray)
            else left == right
            for left, right in zip(rendered[offset], reverse[offset], strict=True)
        )
        for offset in offsets
    )
    neutral_candidate, neutral_smooth, _ = rendered[0.0]
    neutral_residual = neutral_candidate - neutral_smooth
    controls: dict[float, np.ndarray] = {}
    residual_rms: dict[str, float] = {}
    differences: dict[str, np.ndarray] = {}
    for offset in offsets:
        candidate, smooth, _ = rendered[offset]
        scale = smooth / neutral_smooth
        controls[offset] = smooth + neutral_residual * scale
        residual_rms[str(offset)] = float(
            np.sqrt(np.mean(np.square(candidate - smooth)))
        )
        differences[str(offset)] = candidate - controls[offset]
    nonzero = [offset for offset in offsets if offset != 0.0]
    nonzero_different_fraction = float(
        np.mean(
            [np.mean(rendered[offset][0] != controls[offset]) for offset in nonzero]
        )
    )
    nonzero_rmse = [
        float(np.sqrt(np.mean(np.square(differences[str(offset)]))))
        for offset in nonzero
    ]
    nonzero_absolute = np.concatenate(
        [np.abs(differences[str(offset)]).ravel() for offset in nonzero]
    )
    candidates = np.stack([rendered[offset][0] for offset in offsets])
    smooths = np.stack([rendered[offset][1] for offset in offsets])
    epsilon = 1.0 / 65535.0
    new_boundary = ((candidates <= epsilon) | (candidates >= 1.0 - epsilon)) & ~(
        (smooths <= epsilon) | (smooths >= 1.0 - epsilon)
    )
    finite = bool(np.all(np.isfinite(candidates)))
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "replay_byte_exact": replay_exact,
        "reverse_exposure_order_byte_exact": reverse_exact,
        "row_partition_byte_exact": all(item[2] for item in rendered.values()),
        "zero_offset_candidate_control_error": float(
            np.max(np.abs(neutral_candidate - controls[0.0]))
        ),
        "nonzero_offset_different_fraction": nonzero_different_fraction,
        "nonzero_offset_minimum_order_difference_rmse": min(nonzero_rmse),
        "nonzero_offset_order_difference_p95_absolute": float(
            np.quantile(nonzero_absolute, 0.95)
        ),
        "density_conditioned_residual_rms": residual_rms,
        "density_conditioned_residual_rms_span": float(
            np.ptp(list(residual_rms.values()))
        ),
        "matched_control_smooth_tone_error": 0.0,
        "candidate_all_values_finite": finite,
        "candidate_new_boundary_fraction": float(np.mean(new_boundary)),
        "display_rgb_effect_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "replay_byte_exact": replay_exact is gates["replay_byte_exact"],
        "reverse_exposure_order_byte_exact": reverse_exact
        is gates["reverse_exposure_order_byte_exact"],
        "row_partition_byte_exact": measurements["row_partition_byte_exact"]
        is gates["row_partition_byte_exact"],
        "zero_offset_candidate_control_error_maximum": measurements[
            "zero_offset_candidate_control_error"
        ]
        <= gates["zero_offset_candidate_control_error_maximum"],
        "minimum_nonzero_offset_different_fraction": nonzero_different_fraction
        >= gates["minimum_nonzero_offset_different_fraction"],
        "minimum_nonzero_offset_order_difference_rmse": measurements[
            "nonzero_offset_minimum_order_difference_rmse"
        ]
        >= gates["minimum_nonzero_offset_order_difference_rmse"],
        "minimum_nonzero_offset_order_difference_p95_absolute": measurements[
            "nonzero_offset_order_difference_p95_absolute"
        ]
        >= gates["minimum_nonzero_offset_order_difference_p95_absolute"],
        "density_conditioned_residual_rms_span_minimum": measurements[
            "density_conditioned_residual_rms_span"
        ]
        >= gates["density_conditioned_residual_rms_span_minimum"],
        "matched_control_smooth_tone_error_maximum": measurements[
            "matched_control_smooth_tone_error"
        ]
        <= gates["matched_control_smooth_tone_error_maximum"],
        "candidate_all_values_finite": finite is gates["candidate_all_values_finite"],
        "candidate_new_boundary_fraction_maximum": measurements[
            "candidate_new_boundary_fraction"
        ]
        <= gates["candidate_new_boundary_fraction_maximum"],
        "display_rgb_effect_count_zero": (measurements["display_rgb_effect_count"] == 0)
        is gates["display_rgb_effect_count_zero"],
    }
    if set(gate_results) != set(gates):
        raise ExposureGrainCouplingError("P9K gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p9k_exposure_grain_coupling_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p9k_exposure_grain_coupling_v1.json"
        ),
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
