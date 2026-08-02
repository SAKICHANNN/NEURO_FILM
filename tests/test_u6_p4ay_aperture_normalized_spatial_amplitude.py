from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.aperture_normalized_spatial_amplitude import (
    ApertureNormalizationError,
    evaluate_normalizer,
    load_contract,
)
from src.film_physics.aperture_normalized_structure import (
    ApertureNormalizedSpatialHypothesis,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4ay_aperture_normalized_spatial_amplitude_v1.json"
PARENT_BUNDLE = (
    ROOT
    / "outputs/experiments/u6_p4ax_kodak_250d_granularity_amplitude_profile_v1/bundle_a.json"
)


def test_contract_rejects_post_result_selection(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["compiler"]["hypothesis_selection_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ApertureNormalizationError, match="contract drift"):
        load_contract(path)


def test_analytic_normalizer_hits_target_for_distinct_shapes() -> None:
    for identifier, family, sigma in (
        ("white", "delta", 0.0),
        ("fine", "gaussian", 2.0),
        ("coarse", "gaussian", 6.0),
    ):
        hypothesis = ApertureNormalizedSpatialHypothesis(
            identifier, family, sigma, 1.0, 48.0
        )
        innovation = hypothesis.innovation_sigma(0.01)
        assert hypothesis.analytic_aperture_sigma(innovation) == pytest.approx(
            0.01, abs=1e-15
        )
    white = ApertureNormalizedSpatialHypothesis("white", "delta", 0.0, 1.0, 48.0)
    coarse = ApertureNormalizedSpatialHypothesis(
        "coarse", "gaussian", 6.0, 1.0, 48.0
    )
    assert white.power_transfer(0.1) - coarse.power_transfer(0.1) > 0.9


@pytest.mark.skipif(not PARENT_BUNDLE.is_file(), reason="P4AX bundle unavailable")
def test_exact_normalizer_experiment_repeats() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_normalizer(contract, ROOT)
    second = evaluate_normalizer(contract, ROOT)
    assert first == second
    assert not first["automatic_pass"]
    assert first["decision"] == "close_generic_aperture_normalization_without_rescue"
    assert [name for name, passed in first["gate_results"].items() if not passed] == [
        "monte_carlo_median"
    ]
    assert len(first["profile_probes"]) == 9


def test_hypothesis_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        ApertureNormalizedSpatialHypothesis("bad", "gaussian", 0.0, 1.0, 48.0)
