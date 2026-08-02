from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.density_to_transmittance_granularity import (
    DensityTransmittanceGranularityError,
    compile_and_evaluate,
    load_contract,
)
from src.film_physics.transmittance_granularity import (
    TransmittanceGranularityProfile,
    density_gaussian_to_transmittance_moments,
    transmittance_moments_to_density_gaussian,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4az_density_to_transmittance_granularity_v1.json"


def test_contract_rejects_display_rgb_noise(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["clipping_or_display_rgb_noise_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DensityTransmittanceGranularityError, match="contract drift"):
        load_contract(path)


def test_exact_moment_mapping_roundtrips() -> None:
    density_mean = np.asarray([0.2, 1.0, 2.8])
    density_rms = np.asarray([0.004, 0.008, 0.012])
    moments = density_gaussian_to_transmittance_moments(density_mean, density_rms)
    recovered_mean, recovered_rms = transmittance_moments_to_density_gaussian(
        moments.transmittance_mean, moments.transmittance_rms
    )
    assert np.max(np.abs(recovered_mean - density_mean)) < 1e-14
    assert np.max(np.abs(recovered_rms - density_rms)) < 1e-14
    assert moments.transmittance_mean[0] > moments.transmittance_mean[-1]


def test_profile_schema_rejects_spatial_claim() -> None:
    contract = load_contract(CONTRACT)
    bundle, _ = compile_and_evaluate(contract, ROOT)
    bundle["spatial_structure_status"] = "measured"
    bundle.pop("profile_id")
    with pytest.raises(ValueError, match="profile schema"):
        TransmittanceGranularityProfile.from_dict(bundle)


def test_exact_compiler_repeats() -> None:
    contract = load_contract(CONTRACT)
    first_bundle, first_report = compile_and_evaluate(contract, ROOT)
    second_bundle, second_report = compile_and_evaluate(contract, ROOT)
    assert first_bundle == second_bundle
    assert first_report == second_report
    assert first_report["automatic_pass"]
    assert first_report["decision"] == (
        "retain_density_to_transmittance_amplitude_compiler"
    )
    assert first_report["probe_count"] == 15
    assert first_report["transmittance_rms_dynamic_range_ratio"] > 100.0
    assert first_bundle["spatial_structure_status"] == "unidentified"
