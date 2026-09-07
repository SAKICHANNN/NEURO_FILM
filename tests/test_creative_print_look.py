import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.compare_creative_looks_v2 import compare
from src.color_engine.creative_print_look import (
    CreativePrintLook,
    _radial_scale,
    render_print_look,
)


def test_soft_radial_scalar_oracle_and_no_plateau():
    q = np.array([0, 0.4, 0.8, 0.9, 1, 1.2, 2, 10], dtype=float)
    radius = q * _radial_scale(q, "soft-knee-v1")
    expected = np.array([v if v <= 0.8 else 1 - 0.04 / (v - 0.6) for v in q])
    np.testing.assert_allclose(radius, expected, atol=2e-16, rtol=0)
    assert np.all(np.diff(radius) > 0) and np.all(radius < 1)
    dense = np.linspace(0.7, 2, 10001)
    mapped = dense * _radial_scale(dense, "soft-knee-v1")
    assert np.all(np.diff(mapped) > 0)
    epsilon = 1e-6
    sides = np.array([0.8 - epsilon, 0.8, 0.8 + epsilon])
    slopes = np.diff(sides * _radial_scale(sides, "soft-knee-v1")) / epsilon
    np.testing.assert_allclose(slopes, 1, rtol=0, atol=6e-6)


def test_soft_radial_highlight_retains_variation_and_chroma_direction():
    values = np.linspace(0.65, 0.95, 1024, dtype=np.float32)
    x = np.stack([values, values * 0.8, values * 0.65], -1)[None]
    old = CreativePrintLook(tint_neutral_power=2)
    new = replace(old, gamut_policy="soft-knee-v1")
    hard, soft = render_print_look(x, old), render_print_look(x, new)
    assert np.count_nonzero(hard[..., 0] > 1 - 1e-6) > 100
    assert np.all(np.diff(soft[0, :, 0]) > 0)
    assert soft.max() < 1 - 1e-4
    yh, ys = luma(hard)[..., None], luma(soft)[..., None]
    np.testing.assert_allclose(yh, ys, atol=5e-8, rtol=0)
    np.testing.assert_allclose(np.cross(hard - yh, soft - ys), 0, atol=2e-8, rtol=0)


def luma(x):
    return np.sum(x.astype(float) * [0.2126, 0.7152, 0.0722], axis=-1)


def test_chromatic_tint_protection_and_v1_preservation():
    spec = CreativePrintLook()
    v2 = replace(spec, tint_neutral_power=2)
    x = np.array([[[0.4, 0.15, 0.4], [0.3, 0.3, 0.3], [0.1, 0.8, 0]]], dtype=np.float32)
    plain = render_print_look(x, replace(spec, shadow=(0, 0, 0), highlight=(0, 0, 0)))
    one, two = render_print_look(x, spec), render_print_look(x, v2)
    assert np.linalg.norm(two[0, 0] - plain[0, 0]) < np.linalg.norm(
        one[0, 0] - plain[0, 0]
    )
    np.testing.assert_array_equal(one[0, 1], two[0, 1])
    np.testing.assert_array_equal(two[0, 2], plain[0, 2])
    np.testing.assert_allclose(luma(one), luma(two), rtol=0, atol=4e-8)
    probe = np.random.default_rng(733).random((17, 19, 3), dtype=np.float32)
    assert (
        hashlib.sha256(render_print_look(probe, spec).tobytes()).hexdigest()
        == "979dbd04accd6084ce816ff62e345acabb0235999a899138d7df1ea73a9de11f"
    )


@pytest.mark.parametrize("policy", ["hard-v1", "soft-knee-v1"])
@pytest.mark.parametrize("power", [0, 2])
def test_luma_tone_and_gamut_independent_of_colour(power, policy):
    x = np.random.default_rng(733).random((128, 129, 3), dtype=np.float32)
    y = luma(x)
    spec = CreativePrintLook(tint_neutral_power=power, gamut_policy=policy)
    a, b = spec.tone
    target = 3 * a * (1 - y) ** 2 * y + 3 * b * (1 - y) * y * y + y**3
    result = render_print_look(x, spec)
    np.testing.assert_allclose(luma(result), target, rtol=0, atol=4e-8)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    assert not np.shares_memory(result, x)
    assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(result, render_print_look(x, spec))
    np.testing.assert_array_equal(
        result,
        np.concatenate(
            [render_print_look(x[:49], spec), render_print_look(x[49:], spec)]
        ),
    )
    np.testing.assert_array_equal(result[::-1], render_print_look(x[::-1], spec))
    np.testing.assert_array_equal(render_print_look(x, spec, amount=0), x)


@pytest.mark.parametrize("policy", ["hard-v1", "soft-knee-v1"])
@pytest.mark.parametrize("power", [0, 2])
def test_neutral_ramp_tone_order_and_zero_endpoints(power, policy):
    r = np.linspace(0, 1, 65537, dtype=np.float32)
    x = np.repeat(r[None, :, None], 3, axis=-1)
    out = render_print_look(
        x, CreativePrintLook(tint_neutral_power=power, gamut_policy=policy)
    )
    assert np.all(np.diff(luma(out).ravel()) > 0)
    np.testing.assert_array_equal(out[:, 0], [[0, 0, 0]])
    np.testing.assert_array_equal(out[:, -1], [[1, 1, 1]])
    # Smooth tonal grading, not a maximum-difference objective.
    assert np.max(np.abs(np.diff(out, axis=1))) < 1e-4


def test_cube_and_zero_tint_scalar_contract():
    v = np.linspace(0, 1, 33, dtype=np.float32)
    x = np.stack(np.meshgrid(v, v, v, indexing="ij"), -1).reshape(33, -1, 3)
    spec = CreativePrintLook()
    original = x.copy()
    for amount in (0, 0.25, 0.5, 1):
        result = render_print_look(x, spec, amount=amount)
        assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(x, original)
    # Without tint and tone, contraction must be exactly around encoded luma.
    neutral = replace(spec, tone=(1 / 3, 2 / 3), shadow=(0, 0, 0), highlight=(0, 0, 0))
    y = luma(x)[..., None]
    expected = y + spec.chroma * (x - y)
    np.testing.assert_allclose(
        render_print_look(x, neutral), expected, rtol=0, atol=4e-8
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3), np.float32),
        np.zeros((2, 2, 3)),
        np.zeros((0, 2, 3), np.float32),
        np.full((2, 2, 3), np.nan, np.float32),
        np.full((2, 2, 3), -0.01, np.float32),
        np.full((2, 2, 3), 1.01, np.float32),
    ],
)
def test_invalid_pixels(bad):
    with pytest.raises(ValueError):
        render_print_look(bad, CreativePrintLook())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tone": (0.9, 0.1)},
        {"chroma": True},
        {"shadow": (0, 0, float("nan"))},
        {"highlight": (0, 0, 0.31)},
        {"gamut_policy": "unknown"},
    ],
)
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError):
        CreativePrintLook(**kwargs)


@pytest.mark.parametrize("mode", ["assessment", "ordered_assessment", "subject_detail"])
def test_no_confirmation_access_for_development(monkeypatch, mode):
    original = Path.read_bytes

    def read(path):
        if path.suffix.lower() == ".png":
            pytest.fail("development mode must reject before pixels")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="development only"):
        compare(
            "never-create",
            creative_config="creative_print_development_v1.json",
            **{mode: True},
        )
