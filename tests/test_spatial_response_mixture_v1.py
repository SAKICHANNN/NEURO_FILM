import copy
import json
from pathlib import Path

import torch

from scripts.run_spatial_response_mixture_v1 import fit_loss, make_model, render_native, secondary_split
from src.models.color_lut.spatial_response_mixture_v1 import SpatialResponseMixture, effective_probe_luts


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/spatial_response_mixture_v1.json").read_text())
torch.set_num_threads(4)


def test_matched_models_same_parameters_and_initialization():
    local, pooled = make_model("local", CONFIG), make_model("pooled", CONFIG)
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in pooled.parameters())
    assert all(torch.equal(value, pooled.state_dict()[key]) for key, value in local.state_dict().items())
    x, style = torch.rand(2, 3, 37, 61), torch.tensor([0, 2])
    assert local.predict_weights(x, style).shape == (2, 4, 8, 8)
    assert pooled.predict_weights(x, style).shape == (2, 4, 1, 1)
    assert torch.allclose(local.predict_weights(x, style).sum(1), torch.ones(2, 8, 8), atol=2e-7)


def test_symmetry_break_has_gate_gradient_and_experts_diverge_on_update():
    model = make_model("local", CONFIG)
    torch.manual_seed(4)
    x = torch.rand(2, 3, 32, 32) * 0.8 + 0.1
    target = x.clone()
    target[:, 0, :, :16] *= 0.6
    target[:, 2, :, 16:] = target[:, 2, :, 16:] * 0.6 + 0.3
    style = torch.tensor([0, 1])
    before = model.basis_residual.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss, _ = fit_loss(model, "local", x, style, target, CONFIG)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head[-1].weight.grad.abs().sum() > 1e-7
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    update = model.basis_residual.detach() - before
    assert float((update[:, 0] - update[:, 1]).abs().max()) > 1e-7


def test_native_tiling_matches_one_global_weight_field_and_misalignment():
    model = make_model("local", CONFIG).eval()
    x, style = torch.rand(1, 3, 47, 63), torch.tensor([2])
    with torch.no_grad():
        direct = model(x, style)
        tiled, auxiliary = render_native(model, "local", x, style, 13)
        shifted = model(x, style, misalign=True)
        shifted_tiled, _ = render_native(model, "local", x, style, 13, True)
    assert tiled.shape == x.shape and auxiliary["weights"].shape == (1, 4, 8, 8)
    assert torch.allclose(direct, tiled, atol=2e-7, rtol=0)
    assert torch.allclose(shifted, shifted_tiled, atol=2e-7, rtol=0)
    assert torch.isfinite(tiled).all() and tiled.min() >= 0 and tiled.max() <= 1


def test_same_rgb_receives_spatially_different_full_color_response():
    model = SpatialResponseMixture()
    x = torch.full((1, 3, 20, 40), 0.5)
    bases = torch.zeros(1, 4, 9, 9, 9, 3)
    bases[:, 0, ..., 0] = 1
    bases[:, 1, ..., 2] = 1
    weights = torch.zeros(1, 4, 2, 2)
    weights[:, 0, :, 0] = 1
    weights[:, 1, :, 1] = 1
    y = model.render(x, bases, weights)
    assert y[0, 0, 10, 0] == 1 and y[0, 2, 10, -1] == 1
    assert y[0, 2, 10, 0] == 0 and y[0, 0, 10, -1] == 0
    probes = effective_probe_luts(bases, weights)
    assert probes.shape == (4, 9, 9, 9, 3)
    assert torch.isfinite(probes).all()


def test_secondary_split_excludes_all_old_evaluation_roles_and_is_group_disjoint():
    rows = [{"role": "paired_fit", "group": f"group{i}", "files": [{"path": f"train/{style}/{i}.png"} for style in ["input", "Cinema", "ClassNeg", "Velvia"]]} for i in range(64)]
    rows.append({"role": "paired_development_evaluation", "group": "forbidden", "files": [{"path": "test/forbidden.png"}]})
    original = copy.deepcopy(rows)
    fit, check = secondary_split({"rows": rows}, CONFIG)
    assert rows == original
    assert len(fit) == 48 and len(check) == 16
    assert not ({r["group"] for r in fit} & {r["group"] for r in check})
    assert all(r["group"] != "forbidden" for r in fit + check)
    reversed_fit, reversed_check = secondary_split({"rows": list(reversed(rows))}, CONFIG)
    assert fit == reversed_fit and check == reversed_check


def test_all_four_training_losses_and_global_native_render_are_finite():
    torch.manual_seed(13)
    x, target, style = torch.rand(2, 3, 21, 29), torch.rand(2, 3, 21, 29), torch.tensor([0, 1])
    for arm in CONFIG["arms"]:
        model = make_model(arm, CONFIG)
        loss, terms = fit_loss(model, arm, x, style, target, CONFIG)
        loss.backward()
        assert torch.isfinite(loss) and set(terms) == {"l1", "curvature", "fold", "weight_spatial"}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if arm in ("full_global", "shared_global"):
            direct = model(x[:1], style[:1])
            tiled, _ = render_native(model, arm, x[:1], style[:1], 11)
            assert torch.allclose(direct, tiled, atol=2e-7, rtol=0)
