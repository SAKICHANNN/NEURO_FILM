import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.color_lut.nlut_reference import (
    basis_chunk,
    fused_lut,
    interpolate,
    load_official,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def official():
    torch.set_num_threads(4)
    cfg = json.loads(
        (ROOT / "configs/ai_nlut_reference_development_v1.json").read_text()
    )
    return load_official(ROOT, cfg)


def test_fused_basis_and_gradient_equal_original(official):
    torch.manual_seed(8)
    base = official.OriginalCLUT(7, 5, 3, 4).double()
    with torch.no_grad():
        base.LUTs.normal_()
    other = copy.deepcopy(base)
    weights = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    w2 = weights.detach().clone().requires_grad_()
    expected, _ = base.combine(weights, None)
    actual = fused_lut(other, w2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        torch.cat([basis_chunk(other, 0, 3), basis_chunk(other, 3, 7)]),
        base.reconstruct_luts(),
        atol=1e-12,
        rtol=1e-12,
    )
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(weights.grad, w2.grad, atol=1e-11, rtol=1e-11)
    for a, b in zip(base.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-10)


def test_chunk_regularizer_gradient_matches_full(official):
    torch.manual_seed(9)
    base = official.OriginalCLUT(7, 5, 3, 4).double()
    with torch.no_grad():
        base.LUTs.normal_()
    other = copy.deepcopy(base)
    regularizer = official.TVMN(5).double()
    expected = regularizer(base.reconstruct_luts())[:2].sum()
    expected.backward()
    values = []
    for start, end in ((0, 3), (3, 7)):
        value = regularizer(basis_chunk(other, start, end))[:2].sum() * (
            (end - start) / 7
        )
        values.append(value.detach())
        value.backward()
    torch.testing.assert_close(sum(values), expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(base.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-12, rtol=1e-12)


def scalar_reference(lut, images):
    # Independent scalar transcription of the pinned C++ float-bin oracle.
    n, _, h, w = images.shape
    d = lut.shape[-1]
    step = np.float32(1.000001 / (d - 1))
    out = np.empty_like(images)
    for row in range(n):
        for y in range(h):
            for x in range(w):
                rgb = images[row, :, y, x]
                ids = np.floor(rgb / step).astype(int)
                frac = np.fmod(rgb, step) / step
                value = np.zeros(3, dtype=np.float32)
                for dr, dg, db in (
                    (0, 0, 0),
                    (1, 0, 0),
                    (0, 1, 0),
                    (1, 1, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                    (0, 1, 1),
                    (1, 1, 1),
                ):
                    weight = (
                        (frac[0] if dr else 1 - frac[0])
                        * (frac[1] if dg else 1 - frac[1])
                        * (frac[2] if db else 1 - frac[2])
                    )
                    value += weight * lut[row, :, ids[2] + db, ids[1] + dg, ids[0] + dr]
                out[row, :, y, x] = value
    return out


def test_interpolation_against_scalar_cpp_equations():
    torch.manual_seed(10)
    lut = torch.randn(2, 3, 5, 5, 5, requires_grad=True)
    image = torch.rand(2, 3, 5, 7)
    image[:, :, 0, 0] = 0
    image[:, :, -1, -1] = 1
    actual = interpolate(lut, image)
    expected = scalar_reference(lut.detach().numpy(), image.numpy())
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
    actual.sum().backward()
    assert torch.isfinite(lut.grad).all()
    torch.testing.assert_close(lut.grad.sum((2, 3, 4)), torch.full((2, 3), 35.0))
    with pytest.raises(ValueError, match="domain"):
        interpolate(lut, image + 2)
