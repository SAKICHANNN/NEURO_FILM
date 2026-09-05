from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.color_engine.creative_look_v2 import (
    CreativeLookError,
    CreativeLookV2,
    render_creative_look_v2,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/creative_looks_v2_development.json"
ROWS = json.loads(CONFIG.read_text(encoding="utf-8"))["looks"]


@pytest.fixture(params=tuple(ROWS))
def spec(request: pytest.FixtureRequest) -> CreativeLookV2:
    return CreativeLookV2(**ROWS[request.param])


def _oracle(pixel: np.ndarray, spec: CreativeLookV2, amount: float) -> list[float]:
    # Scalar math, no production helpers or vector implementation calls.
    a, b = spec.tone
    toned = [
        math.fsum(
            [
                3 * a * (1 - float(v)) ** 2 * float(v),
                3 * b * (1 - float(v)) * float(v) ** 2,
                float(v) ** 3,
            ]
        )
        for v in pixel
    ]
    y = math.fsum(v * w for v, w in zip(toned, (0.2126, 0.7152, 0.0722), strict=True))
    output = []
    for i, value in enumerate(toned):
        exponent = (
            (1 - y) ** 2 * spec.shadow[i]
            + y**2 * spec.highlight[i]
            + spec.chroma * (value - y)
        )
        # Algebraically equivalent denominator in independent scalar form.
        e = math.exp(exponent)
        target = value * e / ((1 - value) + value * e)
        target = spec.black[i] + target * (spec.white[i] - spec.black[i])
        output.append((1 - amount) * float(pixel[i]) + amount * target)
    return output


@pytest.mark.parametrize("name", ["cyan_matte", "sunbleached_print", "deep_chrome"])
def test_bold_endpoints_oracle_and_tiles(name):
    rows = json.loads(
        (ROOT / "configs/creative_looks_v2_bold_development.json").read_text()
    )
    spec = CreativeLookV2(**rows["looks"][name])
    source = np.random.default_rng(617).random((17, 19, 3), dtype=np.float32)
    result = render_creative_look_v2(source, spec)
    expected = np.asarray(
        [_oracle(p, spec, 1) for p in source.reshape(-1, 3)], dtype=np.float32
    ).reshape(source.shape)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(
        result,
        np.concatenate(
            [
                render_creative_look_v2(source[:8], spec),
                render_creative_look_v2(source[8:], spec),
            ]
        ),
    )
    np.testing.assert_array_equal(
        render_creative_look_v2(source, spec, amount=0), source
    )
    endpoints = render_creative_look_v2(
        np.asarray([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32), spec
    )
    np.testing.assert_array_equal(
        endpoints[0], np.asarray([spec.black, spec.white], dtype=np.float32)
    )
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1


@pytest.mark.parametrize(
    "field,value", [("black", [-0.1, 0, 0]), ("white", [1, 1.1, 1]), ("black", [0, 0])]
)
def test_invalid_creative_endpoints(field, value):
    with pytest.raises(CreativeLookError):
        CreativeLookV2((0.3, 0.7), (0, 0, 0), (0, 0, 0), 0, **{field: value})


def test_development_config_is_not_a_stock_or_product_approval() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "development-not-product-approved"
    assert config["input_domain"] == "encoded-display-srgb-f32"
    assert set(ROWS) == {"amber_soft", "cool_clear", "copper_rich"}
    assert "no calibrated stock response" in config["claim"]


def test_scalar_oracle_and_repeat(spec: CreativeLookV2) -> None:
    source = np.random.default_rng(516).random((9, 13, 3), dtype=np.float32)
    for amount in (0.0, 0.25, 0.65, 1.0):
        expected = np.asarray(
            [_oracle(p, spec, amount) for p in source.reshape(-1, 3)], dtype=np.float32
        ).reshape(source.shape)
        result = render_creative_look_v2(source, spec, amount=amount)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(
            result, render_creative_look_v2(source, spec, amount=amount)
        )


def test_cube_bounded_and_endpoints(spec: CreativeLookV2) -> None:
    axis = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        33, -1, 3
    )
    original = cube.copy()
    output = render_creative_look_v2(cube, spec)
    assert output.dtype == np.float32 and output.flags.c_contiguous
    assert output.shape == cube.shape and np.isfinite(output).all()
    assert output.min() == 0.0 and output.max() == 1.0
    np.testing.assert_array_equal(output[0, 0], np.zeros(3, np.float32))
    np.testing.assert_array_equal(output[-1, -1], np.ones(3, np.float32))
    np.testing.assert_array_equal(cube, original)
    assert not np.shares_memory(output, cube)


def test_strided_zero_amount_owned_and_exact(spec: CreativeLookV2) -> None:
    source = np.random.default_rng(37).random((21, 23, 3), dtype=np.float32)[::2, ::2]
    source.flags.writeable = False
    output = render_creative_look_v2(source, spec, amount=0.0)
    np.testing.assert_array_equal(output, source)
    assert output.flags.c_contiguous and output.flags.writeable
    assert not np.shares_memory(source, output)


def test_tiled_strided_and_reverse_exact(spec: CreativeLookV2) -> None:
    source = np.random.default_rng(47).random((37, 53, 3), dtype=np.float32)
    expected = render_creative_look_v2(source, spec, amount=0.65)
    actual = np.empty_like(source)
    for y in reversed(range(0, 37, 7)):
        for x in reversed(range(0, 53, 11)):
            actual[y : y + 7, x : x + 11] = render_creative_look_v2(
                source[y : y + 7, x : x + 11], spec, amount=0.65
            )
    np.testing.assert_array_equal(actual, expected)
    reverse = render_creative_look_v2(source[::-1, ::-1], spec, amount=0.65)
    np.testing.assert_array_equal(reverse[::-1, ::-1], expected)


def test_neutral_ramp_smoothly_increases_for_development_specs(
    spec: CreativeLookV2,
) -> None:
    ramp = np.repeat(np.linspace(0, 1, 2049, dtype=np.float32)[:, None], 3, axis=1)[
        None
    ]
    output = render_creative_look_v2(ramp, spec)[0]
    assert np.all(np.diff(output, axis=0) > 0)
    assert float(np.max(np.diff(output, axis=0))) < 0.002


def test_amount_is_convex_strength_not_new_render_selection(
    spec: CreativeLookV2,
) -> None:
    source = np.random.default_rng(61).random((11, 17, 3), dtype=np.float32)
    full = render_creative_look_v2(source, spec)
    half = render_creative_look_v2(source, spec, amount=0.5)
    np.testing.assert_allclose(half, 0.5 * source + 0.5 * full, rtol=0, atol=1e-7)


@pytest.mark.parametrize("amount", [True, None, "0.5", -0.01, 1.01, math.nan, math.inf])
def test_reject_invalid_amount(amount: object) -> None:
    with pytest.raises(CreativeLookError):
        render_creative_look_v2(
            np.zeros((1, 1, 3), np.float32),
            CreativeLookV2(**ROWS["amber_soft"]),
            amount=amount,
        )


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((2, 2, 3), np.float64),
        np.zeros((2, 2, 4), np.float32),
        np.zeros((2, 3), np.float32),
        np.zeros((0, 2, 3), np.float32),
        np.full((2, 2, 3), -0.01, np.float32),
        np.full((2, 2, 3), 1.01, np.float32),
        np.full((2, 2, 3), np.nan, np.float32),
        np.full((2, 2, 3), np.inf, np.float32),
    ],
)
def test_reject_invalid_image_even_at_zero(source: np.ndarray) -> None:
    with pytest.raises(CreativeLookError):
        render_creative_look_v2(
            source, CreativeLookV2(**ROWS["amber_soft"]), amount=0.0
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("tone", [0.8, 0.2]),
        ("tone", [0, 1]),
        ("tone", [0.2]),
        ("tone", [True, 0.8]),
        ("shadow", [0, 0, 0.51]),
        ("highlight", [0, 0, math.nan]),
        ("chroma", 1.51),
        ("chroma", False),
    ],
)
def test_reject_invalid_spec(key: str, value: object) -> None:
    row = dict(ROWS["amber_soft"])
    row[key] = value
    with pytest.raises(CreativeLookError):
        CreativeLookV2(**row)


def test_spec_owns_immutable_vectors() -> None:
    row = {"tone": [0.3, 0.7], "shadow": [0, 0, 0], "highlight": [0, 0, 0], "chroma": 0}
    spec = CreativeLookV2(**row)
    row["tone"][0] = 0.9
    assert spec.tone == (0.3, 0.7)
    with pytest.raises(FrozenInstanceError):
        spec.chroma = 1


def test_development_directions_have_different_patch_actions_not_value_proof() -> None:
    patches = np.array(
        [[[0.2, 0.2, 0.2], [0.7, 0.55, 0.4], [0.2, 0.6, 0.25]]], np.float32
    )
    outputs = [
        render_creative_look_v2(patches, CreativeLookV2(**row)) for row in ROWS.values()
    ]
    assert all(
        not np.array_equal(a, b)
        for i, a in enumerate(outputs)
        for b in outputs[i + 1 :]
    )
    # A synthetic neutral shadow witnesses cool-clear intent; not a photo score.
    cool = outputs[1][0, 0]
    assert cool[2] > cool[1] > cool[0]
