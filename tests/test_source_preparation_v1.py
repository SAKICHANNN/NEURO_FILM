import json
from pathlib import Path

import pytest
import torch

from scripts.run_source_preparation_v1 import load_development_image, make_model, training_loss
from src.models.source_preparation_v1 import SourcePreparation, apply_coefficients, constrained_coefficients, masked_l1, mean_field, preview_frame, render_frames, render_native, sample_valid_frame


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/source_preparation_v1.json").read_text())
torch.set_num_threads(2)


def test_constraints_identity_endpoints_and_finite_gradients():
    torch.manual_seed(7)
    raw = (torch.randn(3, 4, 16, 16) * 30).requires_grad_()
    coefficients = constrained_coefficients(raw, 1, 0.25)
    assert coefficients[:, :1].abs().max() <= 1
    assert coefficients[:, 1:].abs().max() <= 0.2500001
    assert coefficients[:, 1:].sum(1).abs().max() < 1e-7
    x = torch.rand(3, 3, 16, 16)
    x[..., 0, :] = 0
    x[..., -1, :] = 1
    y = apply_coefficients(x, coefficients)
    assert torch.equal(y[..., 0, :], x[..., 0, :])
    assert torch.equal(y[..., -1, :], x[..., -1, :])
    assert y.min() >= 0 and y.max() <= 1 and torch.isfinite(y).all()
    assert torch.equal(apply_coefficients(x, torch.zeros_like(coefficients)), x)
    y.mean().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


def test_matched_models_initialize_identity_and_learn_encoder_after_head_update():
    torch.set_num_threads(2)
    local, global_model = make_model("local", CONFIG), make_model("global", CONFIG)
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in global_model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(local.state_dict().values(), global_model.state_dict().values()))
    x, mask, box = preview_frame(torch.rand(1, 3, 43, 67) * 0.6 + 0.1, 64)
    target = (x + 0.1 * mask).clamp(0, 1)
    assert local(x, mask).count_nonzero() == 0
    optimizer = torch.optim.Adam(local.parameters(), lr=CONFIG["learning_rate"])
    initial = local.encoder[0].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = training_loss(local, x, target, mask, [box], CONFIG)
        loss.backward()
        assert local.head.weight.grad.abs().sum() > 0
        optimizer.step()
    assert (local.encoder[0].weight - initial).abs().sum() > 0
    assert local.encoder[0].weight.grad.abs().sum() > 0
    global_model.load_state_dict(local.state_dict())
    torch.testing.assert_close(global_model(x, mask), mean_field(local(x, mask)))
    assert (local(x, mask).amax((-2, -1)) - local(x, mask).amin((-2, -1))).max() > 1e-6


def test_masked_loss_ignores_padding_and_weights_sources_equally():
    x = torch.zeros(2, 3, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, :, :2, :2] = 1
    mask[1] = 1
    target = torch.full_like(x, 100)
    target[0, :, :2, :2] = 0.1
    target[1] = 0.3
    torch.testing.assert_close(masked_l1(x, target, mask), torch.tensor(0.2))


def test_whole_image_coefficients_native_tiled_and_padded_frame_agree():
    torch.manual_seed(9)
    x = torch.rand(1, 3, 79, 113)
    coefficients = constrained_coefficients(torch.randn(1, 4, 16, 16), 1, 0.25)
    torch.testing.assert_close(render_native(x, coefficients), render_native(x, coefficients, 17), rtol=0, atol=0)
    preview, mask, box = preview_frame(x, 96)
    top, left, h, w = box
    framed = render_frames(preview, coefficients, [box])
    expected = render_native(preview[..., top:top + h, left:left + w], coefficients)
    torch.testing.assert_close(framed[..., top:top + h, left:left + w], expected)
    assert (framed * (1 - mask)).count_nonzero() == 0


def test_meanfield_uses_constrained_coefficients_and_preserves_zero_sum():
    raw = torch.zeros(1, 4, 16, 16)
    raw[..., :8, :] = 4
    raw[:, 2, 8:, :] = -2
    coefficients = constrained_coefficients(raw, 1, 0.25)
    mean = mean_field(coefficients)
    torch.testing.assert_close(mean, coefficients.mean((-2, -1), keepdim=True).expand_as(mean))
    assert mean[:, 1:].sum(1).abs().max() < 1e-7
    assert not torch.allclose(mean, constrained_coefficients(mean_field(raw), 1, 0.25))
    x = torch.full((1, 3, 91, 57), 0.5)
    assert (render_native(x, mean).amax((-2, -1)) - render_native(x, mean).amin((-2, -1))).max() < 1e-6


def test_forbidden_role_rejected_before_any_asset_read():
    with pytest.raises(ValueError, match="Forbidden role"):
        load_development_image({"split": "confirmation"}, "source")
    with pytest.raises(ValueError, match="Forbidden role"):
        load_development_image({"split": "development"}, "target")


def test_valid_frame_sampling_maps_padded_ramps_to_source_coordinates():
    yy, xx = torch.meshgrid(torch.arange(8) * 8.0, torch.arange(8) * 8.0, indexing="ij")
    feature_ramps = torch.stack((xx, yy))[None]
    for h, w in ((64, 32), (32, 64), (48, 64)):
        _, mask, (top, left, ph, pw) = preview_frame(torch.zeros(1, 3, h, w), 64)
        sampled = sample_valid_frame(feature_ramps, mask, 4)
        unit = (torch.arange(4) + 0.5) / 4
        expected_x = (left + unit * pw - 0.5)[None, :].expand(4, -1)
        expected_y = (top + unit * ph - 0.5)[:, None].expand(-1, 4)
        torch.testing.assert_close(sampled[0, 0], expected_x, rtol=0, atol=1e-5)
        torch.testing.assert_close(sampled[0, 1], expected_y, rtol=0, atol=1e-5)
