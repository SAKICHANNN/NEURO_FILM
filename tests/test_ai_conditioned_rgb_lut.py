import pytest
import torch

from scripts.audit_ai_paired_operator_capacity import (
    cell_centre_determinants,
    curvature,
)
from src.models.color_lut.conditioned_rgb_lut import ConditionedRGBLUT


def test_identity_gradient_and_bound():
    model = ConditionedRGBLUT()
    x = torch.rand(3, 3, 16, 16)
    styles = torch.arange(3)
    y = model(x, styles)
    torch.testing.assert_close(x, y, rtol=0, atol=2e-7)
    (y - x * 0.8).square().mean().backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert model.global_residual.grad.abs().sum() > 0
    with torch.no_grad():
        model.global_residual.normal_(0, 10)
    y = model(x, styles)
    assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1


def test_curvature_and_fold_gradients_finite():
    lut = torch.rand(2, 9, 9, 9, 3, requires_grad=True)
    loss = curvature(lut) + (0.1 - cell_centre_determinants(lut)).relu().square().mean()
    loss.backward()
    assert torch.isfinite(lut.grad).all() and lut.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke")
def test_cuda_model_regularized_update():
    model = ConditionedRGBLUT().cuda()
    x = torch.linspace(0, 1, 768, device="cuda").reshape(1, 3, 16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(3):
        lut = model.predict_lut(x, torch.tensor([0], device="cuda"))
        loss = (model(x, torch.tensor([0], device="cuda")) - x * 0.8).abs().mean()
        loss = (
            loss
            + 0.001 * curvature(lut)
            + 0.01 * (0.1 - cell_centre_determinants(lut)).relu().square().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )
        optimizer.step()


def test_new_roles_exclude_prior_groups(tmp_path, monkeypatch):
    import json

    from scripts import run_ai_conditioned_rgb_lut as run

    def row(group, role):
        return {
            "group": group,
            "role": role,
            "files": [{"path": f"train/input/{group}.png"}],
        }

    monkeypatch.setattr(run, "ROOT", tmp_path)
    prior = {
        "rows": [
            row("fit", "paired_fit"),
            row("old_eval", "paired_development_evaluation"),
        ]
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior))
    candidates = {
        "inventory_sha256": "fixture",
        "rows": prior["rows"]
        + [
            row("preflight", "paired_development_evaluation"),
            row("new", "paired_development_evaluation"),
        ],
    }
    monkeypatch.setattr(run, "freeze", lambda cfg: candidates)
    cfg = {
        "prior_manifest": "prior.json",
        "prior_manifest_sha256": run.sha(path),
        "fit_count": 1,
        "evaluation_count": 1,
        "excluded_preflight_names": ["preflight.png"],
    }
    manifest = run.new_manifest(cfg)
    assert [r["group"] for r in manifest["rows"]] == ["fit", "new"]
