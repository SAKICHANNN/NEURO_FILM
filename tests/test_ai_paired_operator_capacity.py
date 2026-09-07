import pytest
import torch

from scripts.audit_ai_paired_operator_capacity import (
    cell_centre_determinants,
    checkerboard,
    curvature,
    select_fit,
)
from src.models.color_lut.lut import apply_lut, identity_lut


def test_lut_identity_and_rgb_coupling():
    x = torch.rand(2, 3, 7, 9)
    lut = identity_lut(9).unsqueeze(0).repeat(2, 1, 1, 1, 1)
    torch.testing.assert_close(apply_lut(x, lut), x, rtol=0, atol=2e-7)
    swapped = lut[..., [1, 2, 0]]
    torch.testing.assert_close(
        apply_lut(x, swapped), x[:, [1, 2, 0]], rtol=0, atol=2e-7
    )
    assert float(curvature(lut)) == 0
    assert torch.equal(cell_centre_determinants(lut), torch.ones(2, 8, 8, 8))
    flipped = lut.clone()
    flipped[..., 0] = 1 - flipped[..., 0]
    assert (cell_centre_determinants(flipped) < 0).all()


def test_partition_and_gradient():
    mask = checkerboard(32, 32, 8)
    assert mask.sum() == 512 and not (mask & ~mask).any()
    lut = identity_lut(9).requires_grad_()
    x = torch.rand(1, 3, 32, 32)
    y = apply_lut(x, lut)
    y[:, :, mask].mean().backward()
    assert torch.isfinite(lut.grad).all() and lut.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        checkerboard(4, 4, 8)


def test_fit_role_selection_rejects_shortage_and_duplicates():
    files = [{"path": "train/input/a.png"}] * 4
    rows = [
        {"role": "paired_development_evaluation", "group": "sealed", "files": []},
        {"role": "paired_fit", "group": "a", "files": files},
        {"role": "paired_fit", "group": "b", "files": files},
    ]
    assert [r["group"] for r in select_fit({"rows": rows}, 2)] == ["a", "b"]
    with pytest.raises(ValueError, match="Insufficient"):
        select_fit({"rows": rows}, 3)
    rows[-1]["group"] = "a"
    with pytest.raises(ValueError, match="Repeated"):
        select_fit({"rows": rows}, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke")
def test_deterministic_cuda_backward_smoke():
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        x = torch.linspace(0, 1, 192, device="cuda").reshape(1, 3, 8, 8)
        gradients = []
        for _ in range(2):
            lut = identity_lut(9, device=x.device).requires_grad_()
            apply_lut(x, lut).mean().backward()
            gradients.append(lut.grad)
        assert torch.equal(*gradients)
    finally:
        torch.use_deterministic_algorithms(old)
