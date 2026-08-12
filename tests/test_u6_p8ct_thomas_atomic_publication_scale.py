from __future__ import annotations

import hashlib

import numpy as np

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import _parent_payloads
from scripts.evaluate_u6_p8ct_thomas_atomic_publication_scale import (
    _domain_valid_exposure_fixture,
    _exposure_fixture,
    _linear_layer_exposure_fixture,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_thomas_input import RelativeLayerLogExposure


def test_exposure_fixture_matches_frozen_scalar_formula() -> None:
    height, width = 7, 11
    actual = _exposure_fixture(height, width)
    expected = np.empty((3, height, width), dtype=np.float32)
    for channel in range(3):
        for index in range(height * width):
            expected[channel].flat[index] = np.float32(
                ((index * 17 + channel * 13) % 193 - 96) / 64.0
            )
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


def test_exposure_fixture_replays_exactly() -> None:
    first = _exposure_fixture(65, 67)
    second = _exposure_fixture(65, 67)
    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(
        second.tobytes()
    ).digest()


def test_domain_valid_fixture_stays_inside_profile_domains() -> None:
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    exposure = _domain_valid_exposure_fixture(prior, (65, 67))
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        assert float(exposure[channel].min()) >= lower
        assert float(exposure[channel].max()) <= upper


def test_linear_layer_fixture_roundtrips_exact_log_exposure() -> None:
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    exposure = _domain_valid_exposure_fixture(prior, (65, 67))
    linear = _linear_layer_exposure_fixture(exposure)
    assert not linear.values.flags.writeable
    restored = RelativeLayerLogExposure.from_layer_exposure(linear)
    assert restored.values_chw.tobytes() == exposure.tobytes()
