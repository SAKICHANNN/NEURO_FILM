from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.aperture_scaled_compound_poisson import (
    ApertureScaledCompoundPoissonError,
    compile_and_evaluate,
    load_contract,
)
from src.film_physics.aperture_scaled_granularity import (
    ApertureScaledCompoundPoissonProfile,
    scale_density_rms_between_apertures,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bg_aperture_scaled_compound_poisson_v1.json"


def test_contract_rejects_exponent_or_render_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["density_rms_scaling"] = "fit exponent"
    payload["candidate"]["render_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ApertureScaledCompoundPoissonError, match="contract drift"):
        load_contract(path)


def test_scaling_roundtrip_composition_and_validation() -> None:
    values = np.asarray([0.004, 0.01, 0.015])
    scaled = scale_density_rms_between_apertures(values, 48.0, 192.0)
    restored = scale_density_rms_between_apertures(scaled, 192.0, 48.0)
    composed = scale_density_rms_between_apertures(
        scale_density_rms_between_apertures(values, 48.0, 24.0), 24.0, 192.0
    )
    assert np.allclose(restored, values, rtol=0.0, atol=1e-18)
    assert np.allclose(composed, scaled, rtol=0.0, atol=1e-18)
    with pytest.raises(ValueError, match="invalid density-RMS"):
        scale_density_rms_between_apertures(values, 48.0, 0.0)


def test_profile_and_evaluator_repeat() -> None:
    contract = load_contract(CONTRACT)
    first_bundle, first_report = compile_and_evaluate(contract, ROOT)
    second_bundle, second_report = compile_and_evaluate(contract, ROOT)
    assert first_bundle == second_bundle
    assert first_report == second_report
    serialized = dict(first_bundle)
    profile_id = serialized.pop("profile_id")
    profile = ApertureScaledCompoundPoissonProfile.from_dict(serialized)
    assert profile.identity() == profile_id
    assert first_report["automatic_pass"] is True
    assert (
        first_report["stable_evidence_id"]
        == "38fbae4e66dbd742e080dcd4cbe86deb3ad17be32de94e044f79bbdc7a42dd53"
    )
    assert first_report["scaled_row_count"] == 105
    assert all(first_report["gate_results"].values())
    with pytest.raises(ValueError, match="evidence-backed range"):
        profile.scale_density_moments(np.asarray([1.0]), np.asarray([0.01]), 7.0)
