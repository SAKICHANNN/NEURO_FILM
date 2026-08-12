from pathlib import Path

import numpy as np
import pytest

from src.eval.density_conditioned_cross_layer_counts import evaluate, load_contract
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    density_conditioned_component_rates,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dc_density_conditioned_cross_layer_counts_v1.json"


def test_contract_parent() -> None:
    assert (
        load_contract(ROOT, CONTRACT)["experiment_id"]
        == "u6.p4dc-density-conditioned-cross-layer-counts-v1"
    )


def test_rates_reject_bad_scale() -> None:
    profile = CrossLayerPoissonProfile(
        (12.0, 18.0, 15.0), 2.0, (2.0, 1.0, 1.5), (0.02, 0.018, 0.022), 1
    )
    with pytest.raises(ValueError):
        density_conditioned_component_rates(profile, np.ones((2, 2, 3)) * 1.1)


def test_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_parent_drift(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        CONTRACT.read_text().replace(
            "retain_versioned_cross_layer_cloud_reference_profile", "wrong"
        )
    )
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
