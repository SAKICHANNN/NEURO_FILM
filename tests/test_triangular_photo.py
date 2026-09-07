import pytest
import torch

from src.models.color_lut.triangular_photo import TriangularPhoto


def test_identity_and_endpoints():
    model = TriangularPhoto().double()
    x = torch.linspace(0, 1, 300, dtype=torch.float64).reshape(1, 3, 10, 10)
    assert torch.allclose(model(x), x, atol=1e-14, rtol=0)
    with torch.no_grad():
        model.parameters_raw.fill_(100)
    y = model(x)
    assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    assert torch.equal(y[(x == 0) | (x == 1)], x[(x == 0) | (x == 1)])


def test_extreme_parameter_jacobians_positive():
    torch.manual_seed(28)
    model = TriangularPhoto().double()
    for _ in range(12):
        with torch.no_grad():
            model.parameters_raw.copy_(torch.randn_like(model.parameters_raw) * 20)
        x = torch.rand(3, dtype=torch.float64) * 0.98 + 0.01
        jac = torch.autograd.functional.jacobian(
            lambda q: model(q.reshape(1, 3, 1, 1)).flatten(), x
        )
        assert torch.linalg.det(jac) > 0
        assert torch.equal(torch.triu(jac, diagonal=1), torch.zeros_like(jac))


def test_learnable_and_invalid():
    model = TriangularPhoto()
    x = torch.full((1, 3, 2, 2), 0.4)
    model(x).sum().backward()
    assert torch.isfinite(model.parameters_raw.grad).all()
    assert model.parameters_raw.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model(x + 1)
