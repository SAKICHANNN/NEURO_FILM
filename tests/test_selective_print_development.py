"""Authored development composition, no appeal or calibration assertion."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_creative_looks_v2 import selective_print
from src.color_engine.creative_ordered_hue_look import (
    OrderedHueLook,
    render_ordered_hue_look,
)
from src.color_engine.creative_print_look import CreativePrintLook, render_print_look


def specs():
    cfg = json.loads(Path("configs/creative_print_development_v5.json").read_text())
    return CreativePrintLook(**cfg["looks"]["gold_daylight"]), OrderedHueLook(
        **cfg["selective_colour"]["gold_daylight"]
    )


def test_composition_order_bounds_ownership_repeat_tiles():
    p, h = specs()
    x = np.random.default_rng(8).random((31, 37, 3), dtype=np.float32)
    before = x.copy()
    a = selective_print(x, p, h)
    expected = render_print_look(render_ordered_hue_look(x, h), p)
    assert np.array_equal(a, expected)
    assert np.array_equal(a, selective_print(x, p, h))
    assert np.array_equal(
        a,
        np.concatenate([selective_print(x[:11], p, h), selective_print(x[11:], p, h)]),
    )
    assert np.array_equal(before, x) and not np.shares_memory(a, x)
    assert a.dtype == np.float32 and np.isfinite(a).all()
    assert a.min() >= 0 and a.max() <= 1
    assert np.array_equal(selective_print(x, p, h, amount=0), x)
    assert np.array_equal(selective_print(x, p, h, amount=0.5), 0.5 * x + 0.5 * a)


def test_neutral_axis_and_tone_monotonicity():
    p, h = specs()
    x = np.repeat(np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None], 3, axis=2)
    a = selective_print(x, p, h)
    assert np.array_equal(a[..., 0], a[..., 1])
    assert np.array_equal(a[..., 0], a[..., 2])
    assert (np.diff(a[0, :, 0]) > 0).all()
    assert a[0, 0, 0] == 0 and a[0, -1, 0] == 1


@pytest.mark.parametrize("amount", [-1, 2, float("nan")])
def test_invalid_strength(amount):
    with pytest.raises(ValueError):
        selective_print(np.zeros((2, 3, 3), np.float32), *specs(), amount=amount)
