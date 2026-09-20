import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.fable_status_m_positive import StatusMPositive, redistribute_highlights
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model():
    config = json.loads((ROOT / "configs/fable_status_m_positive_v1.json").read_text(encoding="utf-8"))
    prior = ManufacturerCharacteristicPrior.from_dict(json.loads((ROOT / config["prior"]["path"]).read_text(encoding="utf-8"))["prior"])
    return StatusMPositive.build(prior, config["parameters"])


def test_curve_anchors_slope_and_precision(model):
    x = np.repeat(np.array([0., .18, 1.])[:, None], 3, axis=1)
    np.testing.assert_allclose(model.apply(x), x, atol=1e-12)
    delta = 1e-6
    derivative = (model.apply(x[1:2] + delta) - model.apply(x[1:2] - delta)) / (2 * delta)
    np.testing.assert_allclose(derivative, 1.15, atol=1e-7)
    ramp = np.repeat(np.linspace(0, 1, 65537)[:, None], 3, axis=1)
    out = model.apply(ramp)
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    assert np.diff(out, axis=0).min() >= 0
    np.testing.assert_allclose(model.apply(ramp.astype(np.float32)).astype(np.float32), out, atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        model.apply(np.array([[0., 0., 1.01]]))


def test_highlight_conservation_bounds_and_replay(model):
    eps = model.references["epsilon"]
    for value in (0., .5, .75, 1.):
        e = model.pseudo_exposure(np.full((29, 37, 3), value))
        np.testing.assert_allclose(redistribute_highlights(e, eps, model.parameters["optics"]), e, atol=1e-15)
    x = np.zeros((43, 51, 3))
    x[0, 0] = (1, 0, 1)
    x[21, 25] = 1
    x[-5:, -7:, 1] = 1
    e = model.pseudo_exposure(x)
    h = redistribute_highlights(e, eps, model.parameters["optics"])
    assert h.min() >= eps and h.max() <= 1
    np.testing.assert_allclose(h.sum(axis=(0, 1)), e.sum(axis=(0, 1)), atol=1e-10, rtol=0)
    assert h[21, 25, 0] < e[21, 25, 0] and h[21, 24, 0] > e[21, 24, 0]
    np.testing.assert_array_equal(model.apply(x, optics=True), model.apply(x, optics=True))
