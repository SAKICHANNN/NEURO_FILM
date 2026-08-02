from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.measured_nps_field_synthesis import (
    MeasuredNPSFieldSynthesisError,
    evaluate_measured_nps_field_synthesis,
    load_contract,
)
from src.film_physics.historical_noise_spectrum import (
    HistoricalBWNoiseSpectrumProfile,
)
from src.film_physics.measured_nps_field import (
    physical_periodogram,
    synthesize_measured_nps_field_pair,
    target_nps_grids,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bm_measured_nps_field_synthesis_v1.json"
BUNDLE = (
    ROOT
    / "outputs/experiments/u6_p4bl_historical_measured_nps_compiler_v1/run_a/bundle.json"
)


def _first_profile() -> HistoricalBWNoiseSpectrumProfile:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    return HistoricalBWNoiseSpectrumProfile.from_dict(bundle["profiles"][0])


def test_p4bm_contract_is_frozen_before_synthesis() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "neuro_film.u6_p4bm_measured_nps_field_synthesis_contract.v1"
    )
    assert payload["synthesis"]["field_shape"] == [512, 512]
    assert payload["synthesis"]["sample_pitch_millimetres"] == 0.001
    assert payload["synthesis"]["realized_amplitude_renormalization_allowed"] is False
    assert payload["synthesis"]["microscopic_geometry_claimed"] is False
    assert payload["evaluation"]["required_row_count"] == 15


def test_measured_nps_field_has_exact_second_order_target() -> None:
    profile = _first_profile()
    shape = (64, 64)
    pitch = 0.001
    fields = synthesize_measured_nps_field_pair(
        profile, shape=shape, sample_pitch_millimetres=pitch, seed=2608022301
    )
    intrinsic_target, observed_target, supported = target_nps_grids(
        profile, shape, pitch
    )
    intrinsic_periodogram = physical_periodogram(fields.intrinsic, pitch)
    observed_periodogram = physical_periodogram(fields.aperture_observed, pitch)
    selected = supported & (intrinsic_target > 0.0)
    assert np.max(
        np.abs(intrinsic_periodogram[selected] / intrinsic_target[selected] - 1.0)
    ) < 1e-10
    assert np.max(
        np.abs(observed_periodogram[selected] / observed_target[selected] - 1.0)
    ) < 1e-10
    assert fields.construction_fourier_outside_band_exact_zero is True


def test_measured_nps_field_repeat_and_seed_behavior() -> None:
    profile = _first_profile()
    first = synthesize_measured_nps_field_pair(
        profile, shape=(32, 34), sample_pitch_millimetres=0.001, seed=7
    )
    repeated = synthesize_measured_nps_field_pair(
        profile, shape=(32, 34), sample_pitch_millimetres=0.001, seed=7
    )
    different = synthesize_measured_nps_field_pair(
        profile, shape=(32, 34), sample_pitch_millimetres=0.001, seed=8
    )
    assert first.field_hashes() == repeated.field_hashes()
    assert first.field_hashes() != different.field_hashes()
    assert abs(float(np.mean(first.intrinsic))) < 1e-12
    assert np.sqrt(np.mean(np.square(first.aperture_observed))) <= np.sqrt(
        np.mean(np.square(first.intrinsic))
    )


def test_p4bm_contract_rejects_realized_renormalization(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["synthesis"]["realized_amplitude_renormalization_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MeasuredNPSFieldSynthesisError, match="contract drift"):
        load_contract(path)


def test_measured_nps_field_synthesis_passes() -> None:
    report = evaluate_measured_nps_field_synthesis(load_contract(CONTRACT), ROOT)
    assert report["automatic_pass"] is True
    assert report["decision"] == "retain_offline_measured_nps_field_synthesizer"
    assert report["row_count"] == 15
    assert all(report["gate_results"].values())

