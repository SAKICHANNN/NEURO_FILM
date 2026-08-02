from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.thomas_photographic_stress import (
    _adapt_display_linear_to_layer_exposure,
    _derive_seed,
    evaluate_thomas_photographic_stress,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bz_thomas_photographic_stress_v1.json"
PRIOR = ROOT / "outputs/u6_p2q_kodak_250d_characteristic_prior/bundle_run1.json"


def test_per_source_seed_derivation_is_stable_and_distinct() -> None:
    values = {
        _derive_seed(source_id, channel, base)
        for source_id in ("source-a", "source-b")
        for channel, base in zip(
            ("red", "green", "blue"),
            (2608024501, 11400714821016565696, 15111065704576689758),
            strict=True,
        )
    }
    assert len(values) == 6
    assert all(0 <= value < 2**64 for value in values)


def test_fixed_adapter_maps_endpoints_inside_each_curve_domain() -> None:
    payload = json.loads(PRIOR.read_text(encoding="utf-8"))
    prior = ManufacturerCharacteristicPrior.from_dict(payload["prior"])
    source = np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float64)
    exposure, relative_log = _adapt_display_linear_to_layer_exposure(
        source,
        prior,
        minimum_fraction=0.15,
        maximum_fraction=0.85,
        pixel_pitch_um=6.35,
    )
    assert exposure.values.shape == (1, 1, 3)
    for index, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        assert lower < relative_log[0, 0, index] < upper


def test_contract_rejects_per_image_normalization() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["adapter"]["per_image_normalization_allowed"] = True
    with pytest.raises(RuntimeError, match="contract drift"):
        evaluate_thomas_photographic_stress(contract, ROOT)
