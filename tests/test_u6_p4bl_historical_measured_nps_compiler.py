from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.historical_measured_nps_compiler import (
    HistoricalMeasuredNPSCompilerError,
    evaluate_historical_measured_nps_compiler,
    load_contract,
)
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
    profile_from_source_row,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bl_historical_measured_nps_compiler_v1.json"
DECISION = ROOT / "configs/u6_p4bl_historical_measured_nps_compiler_decision_v1.json"
SOURCE = ROOT / "configs/u6_p4bk_fuji_noise_spectrum_source_v1.json"
SOURCE_EVIDENCE_ID = (
    "8561040664c2511b350ba6259dc9c32acba2c5733804be4993cafe96702eda1b"
)


def test_p4bl_contract_is_frozen_before_compilation() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bl_historical_measured_nps_compiler_contract.v1"
    )
    assert payload["compiler"]["frequency_maximum_lines_per_mm"] == 500.0
    assert payload["compiler"]["density_interpolation_allowed"] is False
    assert payload["compiler"]["coefficient_refit_allowed"] is False
    assert payload["evaluation"]["maximum_projection_relative_error"] == 1e-10


def test_historical_profile_reproduces_published_equations() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    profile = profile_from_source_row(source["table_1"][0], SOURCE_EVIDENCE_ID)
    assert float(profile.one_dimensional(0.0)) == 266.0
    expected_q0 = (140.0 / 57.2 + 126.0 / 113.0) / np.sqrt(2.0 * np.pi)
    assert float(profile.aperture_convolved_2d(0.0, 0.0)) == pytest.approx(
        expected_q0, abs=0.0, rel=1e-15
    )
    assert float(profile.circular_aperture_mtf(0.0)) == 1.0
    assert float(profile.intrinsic_2d(0.0, 0.0)) == pytest.approx(
        expected_q0, abs=0.0, rel=1e-15
    )


def test_p4bl_decision_binds_formal_compiler() -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    assert payload["decision"] == "retain_historical_bw_measured_nps_profiles"
    assert payload["bundle_sha256"] == (
        "2299a1acd914aaa865910fc2a7601f286db3df69ffa642d3bc31de8dbd0f310d"
    )
    assert payload["report_sha256"] == (
        "e36bfc27a01f7ce20cd33bfab88f92542034ddcb3be777033f2ab6980413ea36"
    )
    assert payload["stable_evidence_id"] == (
        "47408aa84dc01b5a5b30b3a342752ddaa6e7c781162ecf625af7c0066b71d201"
    )


def test_historical_profile_roundtrip_and_bounds() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    profile = profile_from_source_row(source["table_1"][1], SOURCE_EVIDENCE_ID)
    restored = HistoricalBWNoiseSpectrumProfile.from_dict(profile.to_dict())
    assert restored.to_dict() == profile.to_dict()
    assert restored.identity() == profile.identity()
    assert float(restored.circular_aperture_mtf(500.0)) > 0.7
    with pytest.raises(ValueError, match="outside the frozen interval"):
        restored.intrinsic_2d(501.0, 0.0)


def test_p4bl_contract_rejects_frequency_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["frequency_maximum_lines_per_mm"] = 600.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HistoricalMeasuredNPSCompilerError, match="contract drift"):
        load_contract(path)


def test_historical_measured_nps_compiler_passes() -> None:
    bundle, report = evaluate_historical_measured_nps_compiler(
        load_contract(CONTRACT), ROOT
    )
    assert report["automatic_pass"] is True
    assert report["decision"] == "retain_historical_bw_measured_nps_profiles"
    assert report["profile_count"] == 5
    assert report["component_count"] == 11
    assert len(bundle["profiles"]) == 5
    assert bundle["render_allowed"] is False
    assert all(report["gate_results"].values())
