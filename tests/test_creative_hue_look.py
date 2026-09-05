import colorsys
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.color_engine.creative_hue_look import CreativeHueLook, render_creative_hue_look


def spec():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/creative_hue_look_development_v1.json"
    )
    return CreativeHueLook(**json.loads(path.read_text())["looks"]["petrol_contrast"])


def oracle(rgb, look):
    h, s, v = colorsys.rgb_to_hsv(*map(float, rgb))

    def weight(centre):
        d = abs((h * 360 - centre + 180) % 360 - 180)
        t = max(0, 1 - d / 65)
        return t * t * (3 - 2 * t)

    green, blue = weight(140), weight(235)
    h = (h + (look.green_shift * green + look.blue_shift * blue) / 360) % 1
    scale = s * math.exp(look.green_logsat * green + look.blue_logsat * blue)
    s = scale / (1 - s + scale)
    a, b = look.tone
    v = 3 * a * (1 - v) ** 2 * v + 3 * b * (1 - v) * v * v + v**3
    return colorsys.hsv_to_rgb(h, s, v)


def test_scalar_oracle_cube_and_deterministic_tiles():
    look = spec()
    axis = np.linspace(0, 1, 17, dtype=np.float32)
    x = np.stack(np.meshgrid(axis, axis, axis), -1).reshape(17, 289, 3)
    expected = np.array(
        [oracle(p, look) for p in x.reshape(-1, 3)], np.float32
    ).reshape(x.shape)
    actual = render_creative_hue_look(x, spec())
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    tiled = np.concatenate(
        [render_creative_hue_look(x[i : i + 1], spec()) for i in range(17)]
    )
    np.testing.assert_array_equal(tiled, actual)
    np.testing.assert_array_equal(
        render_creative_hue_look(x[::-1], spec())[::-1], actual
    )
    assert actual.min() == 0 and actual.max() == 1


def test_grey_neutral_warm_hues_and_local_continuity():
    grey = np.repeat(
        np.linspace(0, 1, 1024, dtype=np.float32)[None, :, None], 3, axis=-1
    )
    y = render_creative_hue_look(grey, spec())
    np.testing.assert_array_equal(y[..., 0], y[..., 1])
    np.testing.assert_array_equal(y[..., 1], y[..., 2])
    assert np.all(np.diff(y[0, :, 0]) > 0)
    warm = np.array(
        [colorsys.hsv_to_rgb(h / 360, 0.55, 0.6) for h in range(76)], np.float32
    )[None]
    mapped = render_creative_hue_look(warm, spec())
    for original, result in zip(warm[0], mapped[0]):
        np.testing.assert_allclose(
            colorsys.rgb_to_hsv(*original)[:2],
            colorsys.rgb_to_hsv(*result)[:2],
            atol=1e-7,
        )
    wheel = np.array(
        [colorsys.hsv_to_rgb(h / 360, 0.7, 0.8) for h in np.linspace(0, 360, 3601)],
        np.float32,
    )[None]
    mapped = render_creative_hue_look(wheel, spec())
    assert np.abs(np.diff(mapped, axis=1)).max() < 0.004
    np.testing.assert_array_equal(mapped[0, 0], mapped[0, -1])


def test_owned_zero_amount_and_input_immutability():
    x = np.random.default_rng(6).random((19, 23, 3), dtype=np.float32)[:, ::2]
    before = x.copy()
    y = render_creative_hue_look(x, spec(), amount=0)
    np.testing.assert_array_equal(y, x)
    assert not np.shares_memory(x, y)
    mapped = render_creative_hue_look(x, spec(), amount=0.5)
    assert mapped.flags.c_contiguous and not np.shares_memory(mapped, x)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("amount", [-0.1, 1.1, True, float("nan"), float("inf")])
def test_invalid_amount(amount):
    with pytest.raises((TypeError, ValueError)):
        render_creative_hue_look(np.zeros((1, 1, 3), np.float32), spec(), amount=amount)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((1, 1, 3)),
        np.zeros((0, 1, 3), np.float32),
        np.full((1, 1, 3), np.nan, np.float32),
        np.full((1, 1, 3), 1.01, np.float32),
    ],
)
def test_invalid_input(data):
    with pytest.raises(ValueError):
        render_creative_hue_look(data, spec())


@pytest.mark.parametrize(
    "update",
    [
        {"green_shift": 61},
        {"blue_logsat": True},
        {"tone": [0.7, 0.2]},
        {"tone": [0, 1]},
        {"green_logsat": float("inf")},
    ],
)
def test_invalid_spec(update):
    from dataclasses import asdict

    with pytest.raises((TypeError, ValueError)):
        CreativeHueLook(**(asdict(spec()) | update))
