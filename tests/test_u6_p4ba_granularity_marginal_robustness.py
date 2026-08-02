from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.granularity_marginal_robustness import (
    GranularityMarginalRobustnessError,
    compile_and_evaluate,
    load_contract,
)
from src.film_physics.transmittance_granularity import (
    density_gamma_to_transmittance_moments,
    density_gaussian_to_transmittance_moments,
    density_uniform_to_transmittance_moments,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ba_granularity_marginal_robustness_v1.json"


def test_contract_rejects_spatial_selection(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["shared_constraints"]["spatial_structure_selection_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GranularityMarginalRobustnessError, match="contract drift"):
        load_contract(path)


def test_three_marginals_share_density_moments_but_not_exact_transform_moments() -> (
    None
):
    mean = np.asarray([0.25, 1.0, 2.5])
    rms = np.asarray([0.005, 0.01, 0.012])
    gaussian = density_gaussian_to_transmittance_moments(mean, rms)
    gamma = density_gamma_to_transmittance_moments(mean, rms)
    uniform = density_uniform_to_transmittance_moments(mean, rms)
    assert np.array_equal(gaussian.density_mean, gamma.density_mean)
    assert np.array_equal(gamma.density_rms, uniform.density_rms)
    assert not np.array_equal(gaussian.transmittance_rms, gamma.transmittance_rms)
    assert (
        np.max(np.abs(gamma.transmittance_rms / gaussian.transmittance_rms - 1.0))
        < 5e-4
    )


def test_exact_interval_compiler_repeats() -> None:
    contract = load_contract(CONTRACT)
    first_bundle, first_report = compile_and_evaluate(contract, ROOT)
    second_bundle, second_report = compile_and_evaluate(contract, ROOT)
    assert first_bundle == second_bundle
    assert first_report == second_report
    assert first_report["automatic_pass"]
    assert first_report["decision"] == ("retain_marginal_robust_transmittance_interval")
    assert first_report["probe_count"] == 15
    assert first_bundle["spatial_structure_status"] == "unidentified"
