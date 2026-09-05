import colorsys
import json
import math
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from src.color_engine.creative_ordered_hue_look import (
    OrderedHueLook,
    map_ordered_hue,
    render_ordered_hue_look,
)


def spec():
    root = Path(__file__).resolve().parents[1]
    return OrderedHueLook(
        **json.loads(
            (root / "configs/creative_ordered_hue_development_v1.json").read_text()
        )["looks"]["petrol_ordered"]
    )


def scalar_hue(h, look):
    x, y = look.source_hue, look.mapped_hue
    d = [(b - a) / (v - u) for a, b, u, v in zip(y, y[1:], x, x[1:])]
    m = [min(d[-1], d[0])] + [min(a, b) for a, b in pairwise(d)] + [min(d[-1], d[0])]
    i = next((i for i in range(len(x) - 1) if h < x[i + 1]), len(x) - 2)
    w = x[i + 1] - x[i]
    t = (h - x[i]) / w
    # Independent Hermite basis rather than implementation Bezier basis.
    return (
        (2 * t**3 - 3 * t * t + 1) * y[i]
        + (t**3 - 2 * t * t + t) * w * m[i]
        + (-2 * t**3 + 3 * t * t) * y[i + 1]
        + (t**3 - t * t) * w * m[i + 1]
    )


def scalar_rgb(rgb, look, amount):
    h, s, v = colorsys.rgb_to_hsv(*map(float, rgb))
    h *= 360

    def weight(c, r):
        t = max(0, 1 - abs((h - c + 180) % 360 - 180) / r)
        return t * t * (3 - 2 * t)

    exponent = amount * (
        look.green_logsat * weight(105, 55) + look.blue_logsat * weight(235, 45)
    )
    scale = s * math.exp(exponent)
    s = scale / (1 - s + scale)
    v = v + amount * look.value_lift * v * (1 - v)
    return colorsys.hsv_to_rgb(
        ((1 - amount) * h + amount * scalar_hue(h, look)) / 360, s, v
    )


@pytest.mark.parametrize("amount", [0, 0.25, 0.5, 1])
def test_scalar_cube_tiles_and_value_retention(amount):
    look = spec()
    axis = np.linspace(0, 1, 17, dtype=np.float32)
    x = np.stack(np.meshgrid(axis, axis, axis), -1).reshape(17, 289, 3)
    expected = np.array(
        [scalar_rgb(p, look, amount) for p in x.reshape(-1, 3)], np.float32
    ).reshape(x.shape)
    y = render_ordered_hue_look(x, spec(), amount=amount)
    np.testing.assert_allclose(y, expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(y.max(-1), x.max(-1))
    np.testing.assert_array_equal(
        y,
        np.concatenate(
            [
                render_ordered_hue_look(x[i : i + 1], spec(), amount=amount)
                for i in range(17)
            ]
        ),
    )
    np.testing.assert_array_equal(
        y, render_ordered_hue_look(x[::-1], spec(), amount=amount)[::-1]
    )
    assert y.min() == 0 and y.max() == 1


@pytest.mark.parametrize("amount", [0, 0.25, 0.5, 1])
def test_whole_hue_domain_order_knots_and_c1_seam(amount):
    look = spec()
    h = np.linspace(0, 360, 360001)
    y = map_ordered_hue(h, look, amount=amount)
    slopes = np.diff(y) / np.diff(h)
    assert slopes.min() >= 0.375 - 1e-6
    assert slopes.max() < 3
    expected = (1 - amount) * np.array(look.source_hue) + amount * np.array(
        look.mapped_hue
    )
    np.testing.assert_allclose(
        map_ordered_hue(np.array(look.source_hue), look, amount=amount),
        expected,
        atol=1e-12,
    )
    eps = 1e-5
    for knot in look.source_hue[1:-1]:
        yy = map_ordered_hue(
            np.array([knot - eps, knot, knot + eps]), look, amount=amount
        )
        assert abs((yy[2] - yy[1]) / eps - (yy[1] - yy[0]) / eps) < 1e-5
    assert abs((y[1] - y[0]) - (y[-1] - y[-2])) < 1e-10


def test_warm_purple_and_grey_unchanged_owned_and_finite():
    h = np.r_[np.linspace(0, 50, 101), np.linspace(280, 360, 161)]
    x = np.array([colorsys.hsv_to_rgb(v / 360, 0.55, 0.7) for v in h], np.float32)[None]
    np.testing.assert_array_equal(render_ordered_hue_look(x, spec()), x)
    grey = np.repeat(np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None], 3, -1)
    np.testing.assert_array_equal(render_ordered_hue_look(grey, spec()), grey)
    x = np.random.default_rng(5).random((23, 31, 3), dtype=np.float32)[:, ::2]
    before = x.copy()
    for amount in (0, 0.5, 1):
        y = render_ordered_hue_look(x, spec(), amount=amount)
        assert y.flags.c_contiguous and not np.shares_memory(x, y)
        assert np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("amount", [0, 0.5, 1])
def test_daylight_lift_scalar_monotone_and_no_shadow_darkening(amount):
    look = replace(spec(), value_lift=0.32)
    x = np.random.default_rng(44).random((19, 31, 3), dtype=np.float32)
    expected = np.array(
        [scalar_rgb(p, look, amount) for p in x.reshape(-1, 3)], np.float32
    ).reshape(x.shape)
    y = render_ordered_hue_look(x, look, amount=amount)
    np.testing.assert_allclose(y, expected, atol=1e-7, rtol=0)
    v = x.max(-1).astype(np.float64)
    np.testing.assert_array_equal(
        y.max(-1), (v + amount * 0.32 * v * (1 - v)).astype(np.float32)
    )
    ramp = np.repeat(np.linspace(0, 1, 65536, dtype=np.float32)[None, :, None], 3, -1)
    lifted = render_ordered_hue_look(ramp, look, amount=amount)
    assert np.all(np.diff(lifted[0, :, 0]) > 0)
    assert np.all(lifted >= ramp)
    np.testing.assert_array_equal(lifted[..., 0], lifted[..., 1])
    assert lifted.min() == 0 and lifted.max() == 1


@pytest.mark.parametrize(
    "change",
    [
        {"source_hue": [0, 90, 90, 360]},
        {"mapped_hue": [0, 40, 30, 360]},
        {"mapped_hue": [0, 90, 360]},
        {"source_hue": [1, 90, 360]},
        {"green_logsat": float("nan")},
        {"value_lift": -0.1},
        {"value_lift": 0.6},
        {"blue_logsat": True},
        {"source_hue": [0, 1, 360], "mapped_hue": [0, 300, 360]},
    ],
)
def test_invalid_spec(change):
    with pytest.raises((ValueError, TypeError)):
        OrderedHueLook(**(asdict(spec()) | change))


@pytest.mark.parametrize("amount", [-0.1, 1.1, True, float("nan")])
def test_invalid_amount(amount):
    with pytest.raises((ValueError, TypeError)):
        render_ordered_hue_look(np.zeros((1, 1, 3), np.float32), spec(), amount=amount)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros((1, 1, 3)),
        np.zeros((0, 1, 3), np.float32),
        np.full((1, 1, 3), float("nan"), np.float32),
        np.ones((1, 1, 4), np.float32),
        np.full((1, 1, 3), 1.01, np.float32),
    ],
)
def test_invalid_rgb(x):
    with pytest.raises(ValueError):
        render_ordered_hue_look(x, spec())


@pytest.mark.parametrize("h", [np.array([-1]), np.array([361]), np.array([np.nan])])
def test_invalid_hue(h):
    with pytest.raises(ValueError):
        map_ordered_hue(h, spec())
