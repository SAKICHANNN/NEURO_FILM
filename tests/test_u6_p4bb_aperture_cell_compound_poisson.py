from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.aperture_cell_compound_poisson import (
    ApertureCellCompoundPoissonError,
    compile_and_evaluate,
    load_contract,
)
from src.film_physics.transmittance_granularity import (
    ApertureCellCompoundPoissonProfile,
    compile_aperture_cell_compound_poisson,
    density_compound_poisson_to_transmittance_moments,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bb_aperture_cell_compound_poisson_v1.json"


def test_contract_rejects_spatial_sampling(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["spatial_sampling_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ApertureCellCompoundPoissonError, match="contract drift"):
        load_contract(path)


def test_compiler_exactly_replays_density_moments() -> None:
    mean = np.asarray([0.2, 1.0, 2.5])
    rms = np.asarray([0.004, 0.01, 0.015])
    parameters = compile_aperture_cell_compound_poisson(mean, rms)
    assert np.allclose(parameters.poisson_rate * parameters.density_mark, mean)
    assert np.allclose(
        np.sqrt(parameters.poisson_rate) * parameters.density_mark, rms
    )


def test_compound_poisson_moments_are_positive_and_distinct() -> None:
    mean = np.asarray([0.25, 1.25])
    rms = np.asarray([0.005, 0.012])
    parameters, moments = density_compound_poisson_to_transmittance_moments(
        mean, rms
    )
    assert np.all(parameters.poisson_rate > 1.0)
    assert np.all(parameters.density_mark > 0.0)
    assert np.all(moments.transmittance_mean > 0.0)
    assert np.all(moments.transmittance_rms > 0.0)


def test_profile_roundtrip_and_exact_evaluator_repeat() -> None:
    contract = load_contract(CONTRACT)
    first_bundle, first_report = compile_and_evaluate(contract, ROOT)
    second_bundle, second_report = compile_and_evaluate(contract, ROOT)
    assert first_bundle == second_bundle
    assert first_report == second_report
    serialized = dict(first_bundle)
    profile_id = serialized.pop("profile_id")
    profile = ApertureCellCompoundPoissonProfile.from_dict(serialized)
    assert profile.identity() == profile_id
    assert first_report["automatic_pass"]
    assert first_report["decision"] == (
        "retain_aperture_cell_compound_poisson_amplitude_compiler"
    )
    assert first_report["probe_count"] == 15
    assert first_bundle["microstructure_status"] == "unidentified"
    assert first_bundle["spatial_nps_status"] == "unidentified"
    assert first_bundle["render_allowed"] is False
