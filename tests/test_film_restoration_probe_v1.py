import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_film_restoration_probe_v1 import corruption_schedule, make_model, state_sha, target_rows, training_loss
from src.models.source_preparation_v1 import apply_coefficients, masked_l1, preview_frame, render_frames, render_native, sample_valid_frame


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/film_restoration_probe_v1.json").read_text())
torch.set_num_threads(2)


def test_paired_seed_schedule_and_initialization():
    for seed in CONFIG["seeds"]:
        a, b = corruption_schedule(seed, CONFIG), corruption_schedule(seed, CONFIG)
        assert all(np.array_equal(a[key], b[key]) for key in a)
        assert a["indices"].shape == (1200, 4)
        assert a["identity_batches"].sum() == 60
        assert np.count_nonzero(a["coefficients"][a["identity_batches"]]) == 0
        assert np.abs(a["coefficients"][..., :1, :, :]).max() <= 0.35
        assert np.abs(a["coefficients"][..., 1:, :, :].sum(-3)).max() < 3e-8
        assert state_sha(make_model(seed, CONFIG)) == state_sha(make_model(seed, CONFIG))
    assert state_sha(make_model(CONFIG["seeds"][0], CONFIG)) != state_sha(make_model(CONFIG["seeds"][1], CONFIG))
    assert not np.array_equal(corruption_schedule(CONFIG["seeds"][0], CONFIG)["indices"], corruption_schedule(CONFIG["seeds"][1], CONFIG)["indices"])
    evaluation = corruption_schedule(CONFIG["evaluation_seed"], CONFIG, evaluation=True)
    assert np.array_equal(np.bincount(evaluation["indices"].ravel()), np.full(7, 8))
    assert not evaluation["identity_batches"].any()


@pytest.mark.parametrize("shape", [(31, 79), (79, 31)])
def test_stable_gain_known_inverse_and_native_geometry(shape):
    torch.manual_seed(15)
    target = torch.rand(1, 3, *shape)
    target[..., 0, :] = 0
    target[..., -1, :] = 1
    corruption = torch.tensor([0.31, 0.09, -0.02, -0.07])[None, :, None, None]
    corrupted = apply_coefficients(target, corruption)
    restored = render_native(corrupted, -corruption, tile_size=13)
    assert restored.shape == target.shape
    assert (restored - target).abs().max() < 2e-7
    assert torch.equal(restored[..., 0, :], target[..., 0, :])
    assert torch.equal(restored[..., -1, :], target[..., -1, :])
    assert torch.equal(render_native(target, torch.zeros(1, 4, 16, 16)), target)


def test_synthetic_known_inverse_is_learnable_without_target_leakage():
    torch.manual_seed(19)
    target, mask, box = preview_frame(torch.rand(1, 3, 35, 61) * 0.6 + 0.2, 64)
    corruption = torch.tensor([0.25, 0.07, -0.04, -0.03])[None, :, None, None]
    corrupted = apply_coefficients(target, corruption)
    model = make_model(CONFIG["seeds"][0], CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    baseline = float(masked_l1(corrupted, target, mask))
    unchanged = target.clone()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_loss(model, target, mask, [box], corruption, CONFIG)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    with torch.no_grad():
        nodes = model(corrupted, mask)
        result = render_frames(corrupted, nodes, [box])
    assert torch.equal(target, unchanged)
    assert float(masked_l1(result, target, mask)) < baseline * 0.25
    assert result.shape == target.shape and torch.isfinite(result).all()


def test_inherited_stride8_coordinates_follow_unpadded_rectangle():
    _, mask, (top, left, height, width) = preview_frame(torch.zeros(1, 3, 96, 192), 192)
    rows = torch.arange(24, dtype=torch.float32) * 8
    cols = torch.arange(24, dtype=torch.float32) * 8
    features = torch.stack((cols[None].expand(24, -1), rows[:, None].expand(-1, 24)))[None]
    sampled = sample_valid_frame(features, mask, 16)
    unit = (torch.arange(16) + 0.5) / 16
    assert torch.allclose(sampled[0, 0], (left + width * unit - 0.5).clamp(0, 184)[None].expand(16, -1), atol=2e-5)
    assert torch.allclose(sampled[0, 1], (top + height * unit - 0.5).clamp(0, 184)[:, None].expand(-1, 16), atol=2e-5)


def test_roles_are_exact_seven_development_targets_and_corrected_index():
    rows = target_rows(CONFIG)
    assert CONFIG["digital_indices"] == [220, 39, 250, 101, 213, 78, 210]
    assert [row["id"] for row in rows["film"]] == CONFIG["film_ids"]
    assert len(rows["digital"]) == 7
    assert all(row["provenance"]["split"] == "development" for row in rows["digital"])
    assert all(row["decode"] == "digital" and "expert/" in row["path"] for row in rows["digital"])
