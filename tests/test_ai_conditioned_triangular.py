import torch

from src.models.color_lut.conditioned_triangular import (
    ConditionedTriangular,
    apply_parameters,
)
from src.models.color_lut.triangular_photo import TriangularPhoto


def test_batched_matches_original_formula_with_float32_roundoff():
    torch.manual_seed(5)
    x = torch.rand(3, 3, 8, 9)
    raw = torch.randn(3, 3, 12)
    old = TriangularPhoto()
    expected = []
    for i in range(3):
        with torch.no_grad():
            old.parameters_raw.copy_(raw[i])
        expected.append(old(x[i : i + 1]))
    # Batched tensor kernels are not the historical scalar-kernel byte contract.
    torch.testing.assert_close(
        apply_parameters(x, raw),
        torch.cat(expected),
        rtol=0,
        atol=2 * torch.finfo(torch.float32).eps,
    )


def test_conditional_initial_identity_and_gradient():
    model = ConditionedTriangular()
    x = torch.rand(3, 3, 16, 16)
    styles = torch.arange(3)
    y = model(x, styles)
    torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
    (y - x * 0.8).square().mean().backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert model.global_parameters.grad.abs().sum() > 0


def test_endpoint_and_bound_preservation():
    x = torch.rand(2, 3, 8, 8)
    x[..., 0] = 0
    x[..., -1] = 1
    y = apply_parameters(x, torch.randn(2, 3, 12) * 20)
    assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    assert torch.equal(x[..., 0], y[..., 0])
    assert torch.equal(x[..., -1], y[..., -1])
