from pathlib import Path

import numpy as np
import pytest

from src.eval.cross_layer_compound_poisson import evaluate, load_contract
from src.film_physics.cross_layer_compound_poisson import (
    CrossLayerPoissonProfile,
    sample_cross_layer_poisson_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4cx_cross_layer_compound_poisson_v1.json"


def test_p4cx_profile_rejects_impossible_shared_rates() -> None:
    with pytest.raises(ValueError, match="shared rates"):
        CrossLayerPoissonProfile(
            (1.0, 1.0, 1.0), 1.0, (1.0, 1.0, 1.0), (0.1, 0.1, 0.1), 1
        )


def test_p4cx_sampler_is_partition_exact() -> None:
    profile = CrossLayerPoissonProfile(
        (12.0, 18.0, 15.0), 2.0, (2.0, 1.0, 1.5), (0.02, 0.018, 0.022), 99
    )
    full = sample_cross_layer_poisson_region(
        profile, (32, 48), origin_yx=(0, 0), shape=(32, 48)
    )
    parts = np.concatenate(
        [
            sample_cross_layer_poisson_region(
                profile, (32, 48), origin_yx=(y, 0), shape=(min(7, 32 - y), 48)
            )
            for y in range(0, 32, 7)
        ]
    )
    assert np.array_equal(full, parts)


def test_p4cx_frozen_evaluation() -> None:
    report = evaluate(ROOT, CONTRACT)
    assert report["automatic_pass"] is all(report["stable"]["gates"].values())


def test_p4cx_parent_drift_fails_closed(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"required_decision": "retain_scanner_unmixing_as_required_layer_correlation_identifiability_control"',
        '"required_decision": "wrong"',
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent drift"):
        load_contract(ROOT, path)
