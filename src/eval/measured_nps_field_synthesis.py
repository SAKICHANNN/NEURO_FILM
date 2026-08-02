"""U6.P4BM evaluator for historical measured-NPS field synthesis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.measured_nps_field import (
    physical_periodogram,
    synthesize_measured_nps_field_pair,
    target_nps_grids,
)

SCHEMA = "neuro_film.u6_p4bm_measured_nps_field_synthesis_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bm_measured_nps_field_synthesis_report.v1"


class MeasuredNPSFieldSynthesisError(RuntimeError):
    """Raised when frozen P4BM evidence or semantics drift."""


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


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MeasuredNPSFieldSynthesisError("P4BM paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    synthesis = payload.get("synthesis", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or synthesis.get("field_shape") != [512, 512]
        or synthesis.get("sample_pitch_millimetres") != 0.001
        or synthesis.get("nyquist_lines_per_mm") != 500.0
        or synthesis.get("radial_bandlimit_lines_per_mm") != 500.0
        or synthesis.get("seeds") != [2608022301, 2608022302, 2608022303]
        or synthesis.get("phase_source")
        != "counter_normal_region_then_unit_magnitude_fft_phase"
        or synthesis.get("realized_amplitude_renormalization_allowed") is not False
        or synthesis.get("tile_or_crop_parity_claimed") is not False
        or synthesis.get("gaussian_marginal_claimed") is not False
        or synthesis.get("microscopic_geometry_claimed") is not False
        or synthesis.get("render_allowed") is not False
        or evaluation.get("required_row_count") != 15
        or evaluation.get("maximum_intrinsic_periodogram_relative_error") != 1e-10
        or evaluation.get("maximum_aperture_periodogram_relative_error") != 1e-10
        or evaluation.get("maximum_parseval_relative_error") != 1e-12
        or evaluation.get("maximum_ifft_imaginary_residual") != 1e-12
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise MeasuredNPSFieldSynthesisError("P4BM frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise MeasuredNPSFieldSynthesisError(f"P4BM parent mismatch: {stem}")
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_spectrum_error(
    actual: np.ndarray, expected: np.ndarray, supported: np.ndarray
) -> float:
    selected = supported & (expected > 0.0)
    return float(np.max(np.abs(actual[selected] / expected[selected] - 1.0)))


def _parseval_error(
    field: np.ndarray,
    target: np.ndarray,
    sample_pitch_millimetres: float,
) -> float:
    spatial = float(np.mean(np.square(field), dtype=np.float64))
    spectral = float(
        np.sum(target, dtype=np.float64)
        / (target.size * sample_pitch_millimetres**2)
    )
    return abs(spatial / spectral - 1.0)


def evaluate_measured_nps_field_synthesis(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = contract["parents"]
    _load_parent(parents, root, "p4bl_contract")
    decision = _load_parent(parents, root, "p4bl_decision")
    bundle = _load_parent(parents, root, "p4bl_bundle")
    report = _load_parent(parents, root, "p4bl_report")
    if (
        decision.get("automatic_pass") is not True
        or decision.get("decision") != "retain_historical_bw_measured_nps_profiles"
        or bundle.get("bundle_id") != parents["p4bl_bundle_id"]
        or report.get("bundle_id") != parents["p4bl_bundle_id"]
        or report.get("stable_evidence_id") != parents["p4bl_stable_evidence_id"]
        or bundle.get("profiles") is None
        or bundle.get("render_allowed") is not False
    ):
        raise MeasuredNPSFieldSynthesisError("P4BM parent decision mismatch")

    profiles = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in bundle["profiles"]
    ]
    shape = tuple(int(value) for value in contract["synthesis"]["field_shape"])
    pitch = float(contract["synthesis"]["sample_pitch_millimetres"])
    rows: list[dict[str, Any]] = []
    intrinsic_errors: list[float] = []
    observed_errors: list[float] = []
    parseval_errors: list[float] = []
    means: list[float] = []
    imaginary_residuals: list[float] = []
    construction_zeros: list[bool] = []
    aperture_rms_orders: list[bool] = []
    repeat_exact: list[bool] = []
    all_hashes: list[str] = []
    profile_identity_preserved = True
    for profile, serialized in zip(profiles, bundle["profiles"], strict=True):
        profile_identity_preserved &= profile.to_dict() == serialized
        for seed in contract["synthesis"]["seeds"]:
            fields = synthesize_measured_nps_field_pair(
                profile, shape=shape, sample_pitch_millimetres=pitch, seed=int(seed)
            )
            repeated = synthesize_measured_nps_field_pair(
                profile, shape=shape, sample_pitch_millimetres=pitch, seed=int(seed)
            )
            intrinsic_target, observed_target, supported = target_nps_grids(
                profile, shape, pitch
            )
            intrinsic_periodogram = physical_periodogram(fields.intrinsic, pitch)
            observed_periodogram = physical_periodogram(
                fields.aperture_observed, pitch
            )
            intrinsic_error = _relative_spectrum_error(
                intrinsic_periodogram, intrinsic_target, supported
            )
            observed_error = _relative_spectrum_error(
                observed_periodogram, observed_target, supported
            )
            intrinsic_parseval = _parseval_error(
                fields.intrinsic, intrinsic_target, pitch
            )
            observed_parseval = _parseval_error(
                fields.aperture_observed, observed_target, pitch
            )
            hashes = fields.field_hashes()
            repeat_hashes = repeated.field_hashes()
            intrinsic_rms = float(
                np.sqrt(np.mean(np.square(fields.intrinsic), dtype=np.float64))
            )
            observed_rms = float(
                np.sqrt(
                    np.mean(np.square(fields.aperture_observed), dtype=np.float64)
                )
            )
            field_repeat_exact = hashes == repeat_hashes
            intrinsic_errors.append(intrinsic_error)
            observed_errors.append(observed_error)
            parseval_errors.extend((intrinsic_parseval, observed_parseval))
            means.extend(
                (
                    abs(float(np.mean(fields.intrinsic, dtype=np.float64))),
                    abs(float(np.mean(fields.aperture_observed, dtype=np.float64))),
                )
            )
            imaginary_residuals.extend(
                (
                    fields.intrinsic_ifft_imaginary_residual,
                    fields.aperture_ifft_imaginary_residual,
                )
            )
            construction_zeros.append(
                fields.construction_fourier_outside_band_exact_zero
            )
            aperture_rms_orders.append(observed_rms <= intrinsic_rms)
            repeat_exact.append(field_repeat_exact)
            all_hashes.extend(hashes)
            rows.append(
                {
                    "profile_id": profile.identity(),
                    "material_id": profile.material_id,
                    "diffuse_density": profile.diffuse_density,
                    "seed": int(seed),
                    "intrinsic_field_sha256": hashes[0],
                    "aperture_observed_field_sha256": hashes[1],
                    "intrinsic_rms_source_units": intrinsic_rms,
                    "aperture_observed_rms_source_units": observed_rms,
                    "intrinsic_periodogram_relative_error": intrinsic_error,
                    "aperture_periodogram_relative_error": observed_error,
                    "maximum_parseval_relative_error": max(
                        intrinsic_parseval, observed_parseval
                    ),
                    "maximum_absolute_sample_mean": max(means[-2:]),
                    "maximum_ifft_imaginary_residual": max(
                        imaginary_residuals[-2:]
                    ),
                    "construction_fourier_outside_band_exact_zero": (
                        fields.construction_fourier_outside_band_exact_zero
                    ),
                    "repeat_exact": field_repeat_exact,
                }
            )

    evaluation = contract["evaluation"]
    gates = {
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "seed_count": len(contract["synthesis"]["seeds"])
        == evaluation["required_seed_count"],
        "row_count": len(rows) == evaluation["required_row_count"],
        "intrinsic_periodogram": max(intrinsic_errors)
        <= evaluation["maximum_intrinsic_periodogram_relative_error"],
        "aperture_periodogram": max(observed_errors)
        <= evaluation["maximum_aperture_periodogram_relative_error"],
        "parseval": max(parseval_errors)
        <= evaluation["maximum_parseval_relative_error"],
        "sample_mean": max(means) <= evaluation["maximum_absolute_sample_mean"],
        "ifft_imaginary": max(imaginary_residuals)
        <= evaluation["maximum_ifft_imaginary_residual"],
        "repeat_exact": all(repeat_exact),
        "seed_distinct_fields": len(set(all_hashes)) == len(all_hashes),
        "construction_zero_outside_band": all(construction_zeros),
        "aperture_rms_not_above_intrinsic": all(aperture_rms_orders),
        "profile_identity_preserved": profile_identity_preserved,
    }
    automatic_pass = all(gates.values())
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "parent_bundle_id": parents["p4bl_bundle_id"],
        "field_shape": list(shape),
        "sample_pitch_millimetres": pitch,
        "physical_extent_millimetres": contract["synthesis"][
            "physical_extent_millimetres"
        ],
        "profile_count": len(profiles),
        "seed_count": len(contract["synthesis"]["seeds"]),
        "row_count": len(rows),
        "maximum_intrinsic_periodogram_relative_error": max(intrinsic_errors),
        "maximum_aperture_periodogram_relative_error": max(observed_errors),
        "maximum_parseval_relative_error": max(parseval_errors),
        "maximum_absolute_sample_mean": max(means),
        "maximum_ifft_imaginary_residual": max(imaginary_residuals),
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_offline_measured_nps_field_synthesizer"
            if automatic_pass
            else "close_measured_nps_field_synthesis"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(result)
    stable_payload.pop("stable_evidence_id", None)
    result["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(stable_payload)
    ).hexdigest()
    return result


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

