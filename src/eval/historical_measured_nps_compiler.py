"""U6.P4BL evaluator for the historical measured-NPS compiler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
    profile_from_source_row,
)

SCHEMA = "neuro_film.u6_p4bl_historical_measured_nps_compiler_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bl_historical_measured_nps_compiler_report.v1"
BUNDLE_SCHEMA = "neuro_film.u6_p4bl_historical_measured_nps_bundle.v1"


class HistoricalMeasuredNPSCompilerError(RuntimeError):
    """Raised when frozen P4BL evidence or semantics drift."""


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
        raise HistoricalMeasuredNPSCompilerError("P4BL paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compiler = payload.get("compiler", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or compiler.get("profile_schema")
        != "neuro_film.historical_bw_noise_spectrum_profile.v1"
        or compiler.get("scanning_aperture_diameter_millimetres") != 0.001
        or compiler.get("frequency_minimum_lines_per_mm") != 0.0
        or compiler.get("frequency_maximum_lines_per_mm") != 500.0
        or compiler.get("frequency_grid_lines_per_mm")
        != [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        or compiler.get("projection_check_frequencies_lines_per_mm")
        != [0.0, 25.0, 100.0, 250.0, 500.0]
        or compiler.get("density_interpolation_allowed") is not False
        or compiler.get("density_extrapolation_allowed") is not False
        or compiler.get("coefficient_refit_allowed") is not False
        or compiler.get("render_allowed") is not False
        or evaluation.get("required_profile_count") != 5
        or evaluation.get("required_component_count") != 11
        or evaluation.get("maximum_projection_relative_error") != 1e-10
        or evaluation.get("minimum_aperture_mtf_on_grid") != 0.7
        or evaluation.get("maximum_intrinsic_to_aperture_convolved_ratio") != 2.0
        or evaluation.get("require_two_byte_identical_bundles_and_reports") is not True
    ):
        raise HistoricalMeasuredNPSCompilerError("P4BL frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise HistoricalMeasuredNPSCompilerError(
            f"P4BL parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _projection_relative_error(
    profile: HistoricalBWNoiseSpectrumProfile, frequency: float
) -> float:
    projected, _ = quad(
        lambda v: float(profile.aperture_convolved_2d(frequency, v)),
        -np.inf,
        np.inf,
        epsabs=1e-10,
        epsrel=1e-12,
        limit=200,
    )
    expected = float(profile.one_dimensional(frequency))
    return abs(projected / expected - 1.0)


def _half_power_frequency(profile: HistoricalBWNoiseSpectrumProfile) -> float:
    zero = float(profile.one_dimensional(0.0))
    return float(
        brentq(
            lambda frequency: float(profile.one_dimensional(frequency)) - 0.5 * zero,
            0.0,
            profile.maximum_intrinsic_frequency_lines_per_mm,
        )
    )


def evaluate_historical_measured_nps_compiler(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    source_contract = _load_parent(parents, root, "p4bk_contract")
    source_decision = _load_parent(parents, root, "p4bk_decision")
    source_report = _load_parent(parents, root, "p4bk_report")
    if (
        source_decision.get("automatic_pass") is not True
        or source_decision.get("decision")
        != "open_historical_bw_measured_nps_compiler"
        or source_report.get("stable_evidence_id")
        != parents["p4bk_stable_evidence_id"]
        or source_report.get("transcription_sha256")
        != source_decision.get("transcription_sha256")
        or source_report.get("table_1") != source_contract.get("table_1")
    ):
        raise HistoricalMeasuredNPSCompilerError("P4BL parent decision mismatch")

    profiles = [
        profile_from_source_row(row, str(parents["p4bk_stable_evidence_id"]))
        for row in source_contract["table_1"]
    ]
    serialized_profiles = [profile.to_dict() for profile in profiles]
    roundtrips = [
        HistoricalBWNoiseSpectrumProfile.from_dict(payload)
        for payload in serialized_profiles
    ]
    exact_roundtrip = all(
        restored.to_dict() == original.to_dict()
        and restored.identity() == original.identity()
        for original, restored in zip(profiles, roundtrips, strict=True)
    )
    profile_ids = [profile.identity() for profile in profiles]
    bundle_without_id: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "source_evidence_id": parents["p4bk_stable_evidence_id"],
        "source_pdf_sha256": source_report["source"]["pdf_sha256"],
        "profile_count": len(profiles),
        "profiles": serialized_profiles,
        "density_interpolation_allowed": False,
        "density_extrapolation_allowed": False,
        "render_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    bundle_id = hashlib.sha256(_canonical_json(bundle_without_id)).hexdigest()
    bundle = dict(bundle_without_id)
    bundle["bundle_id"] = bundle_id

    frequency_grid = np.asarray(
        contract["compiler"]["frequency_grid_lines_per_mm"], dtype=np.float64
    )
    projection_grid = tuple(
        float(value)
        for value in contract["compiler"][
            "projection_check_frequencies_lines_per_mm"
        ]
    )
    rows: list[dict[str, Any]] = []
    projection_errors: list[float] = []
    all_positive_finite = True
    all_one_dimensional_monotone = True
    all_convolved_monotone = True
    all_intrinsic_not_below = True
    minimum_mtf = float("inf")
    maximum_correction = 0.0
    for profile in profiles:
        one_dimensional = profile.one_dimensional(frequency_grid)
        convolved = profile.aperture_convolved_2d(frequency_grid, 0.0)
        mtf = profile.circular_aperture_mtf(frequency_grid)
        intrinsic = profile.intrinsic_2d(frequency_grid, 0.0)
        correction = intrinsic / convolved
        errors = [
            _projection_relative_error(profile, frequency)
            for frequency in projection_grid
        ]
        projection_errors.extend(errors)
        all_positive_finite &= bool(
            np.all(np.isfinite(one_dimensional))
            and np.all(np.isfinite(convolved))
            and np.all(np.isfinite(intrinsic))
            and np.all(one_dimensional > 0.0)
            and np.all(convolved > 0.0)
            and np.all(intrinsic > 0.0)
        )
        all_one_dimensional_monotone &= bool(np.all(np.diff(one_dimensional) <= 0.0))
        all_convolved_monotone &= bool(np.all(np.diff(convolved) <= 0.0))
        all_intrinsic_not_below &= bool(np.all(intrinsic >= convolved))
        minimum_mtf = min(minimum_mtf, float(np.min(mtf)))
        maximum_correction = max(maximum_correction, float(np.max(correction)))
        rows.append(
            {
                "profile_id": profile.identity(),
                "material_id": profile.material_id,
                "diffuse_density": profile.diffuse_density,
                "component_count": len(profile.k),
                "one_dimensional_at_zero": float(one_dimensional[0]),
                "one_dimensional_at_500": float(one_dimensional[-1]),
                "half_power_frequency_lines_per_mm": _half_power_frequency(profile),
                "minimum_aperture_mtf": float(np.min(mtf)),
                "maximum_aperture_correction_ratio": float(np.max(correction)),
                "maximum_projection_relative_error": max(errors),
            }
        )

    evaluation = contract["evaluation"]
    gates = {
        "profile_count": len(profiles) == evaluation["required_profile_count"],
        "component_count": sum(len(profile.k) for profile in profiles)
        == evaluation["required_component_count"],
        "projection_identity": max(projection_errors)
        <= evaluation["maximum_projection_relative_error"],
        "positive_finite_spectra": all_positive_finite,
        "one_dimensional_monotone": all_one_dimensional_monotone,
        "aperture_convolved_monotone": all_convolved_monotone,
        "aperture_mtf_zero_free": minimum_mtf
        >= evaluation["minimum_aperture_mtf_on_grid"],
        "bounded_aperture_correction": maximum_correction
        <= evaluation["maximum_intrinsic_to_aperture_convolved_ratio"],
        "intrinsic_not_below_aperture_convolved": all_intrinsic_not_below,
        "unique_profile_identities": len(set(profile_ids)) == len(profile_ids),
        "exact_bundle_roundtrip": exact_roundtrip,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "bundle_id": bundle_id,
        "source_evidence_id": parents["p4bk_stable_evidence_id"],
        "profile_count": len(profiles),
        "component_count": sum(len(profile.k) for profile in profiles),
        "frequency_interval_lines_per_mm": [
            contract["compiler"]["frequency_minimum_lines_per_mm"],
            contract["compiler"]["frequency_maximum_lines_per_mm"],
        ],
        "maximum_projection_relative_error": max(projection_errors),
        "minimum_aperture_mtf": minimum_mtf,
        "maximum_aperture_correction_ratio": maximum_correction,
        "rows": rows,
        "gate_results": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_historical_bw_measured_nps_profiles"
            if automatic_pass
            else "close_historical_bw_measured_nps_compiler"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(report)
    stable_payload.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(stable_payload)
    ).hexdigest()
    return bundle, report


def write_payload(payload: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()

