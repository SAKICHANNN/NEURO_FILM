from __future__ import annotations

import numpy as np

from src.real_film.gold_matrix_transplant import (
    DigitalSample,
    build_archive_ood_model,
    colour_distribution_descriptor,
    composite_affine,
    evaluate_transplant,
)
from src.real_film.gold_transform_consistency import BoundedAffineOperator, PairedFrameSamples


def _frame(frame_id: str, roll_id: str, shift: float = 0.0) -> PairedFrameSamples:
    seed = sum((index + 1) * value for index, value in enumerate(f"{frame_id}:{roll_id}".encode()))
    rng = np.random.default_rng(seed)
    source = np.clip(rng.uniform(0.05, 0.85, size=(300, 3)) + shift, 0.0, 1.0)
    target = np.clip(source @ np.array([[0.9, 0.08, 0.02], [0.03, 0.92, 0.04], [0.05, 0.03, 0.88]]).T + 0.03, 0.0, 1.0)
    return PairedFrameSamples(frame_id, roll_id, source, target)


def test_descriptor_has_frozen_nine_dimensions() -> None:
    pixels = np.linspace(0.0, 1.0, 300).reshape(100, 3)
    descriptor = colour_distribution_descriptor(pixels)
    assert descriptor.shape == (9,)
    assert np.all(np.isfinite(descriptor))


def test_ood_threshold_uses_cross_roll_archive_only() -> None:
    frames = [_frame("a1", "a"), _frame("a2", "a", 0.01), _frame("b1", "b", 0.02), _frame("b2", "b", 0.03)]
    first = build_archive_ood_model(frames, threshold_quantile=0.95)
    second = build_archive_ood_model(frames, threshold_quantile=0.95)
    assert first == second
    assert len(first["leave_one_roll_nearest_distances"]) == 4
    assert first["threshold"] > 0.0


def test_composite_affine_is_identity_interpolation() -> None:
    operator = BoundedAffineOperator(np.diag([1.2, 0.8, 1.1]), np.array([0.1, -0.05, 0.02]), "test")
    half = composite_affine(operator, 0.5)
    assert half is not None
    assert np.allclose(half.matrix, np.diag([1.1, 0.9, 1.05]))
    assert np.allclose(half.bias, [0.05, -0.025, 0.01])


def test_evaluation_fails_closed_on_ood_coverage() -> None:
    frames = [_frame("a", "a"), _frame("b", "b", 0.02), _frame("c", "c", 0.04)]
    rng = np.random.default_rng(11)
    digital = [
        DigitalSample("g", "gold", rng.uniform(0.94, 1.0, size=(300, 3)), "g.png"),
        DigitalSample("s", "stress", rng.uniform(0.92, 0.99, size=(300, 3)), "s.png"),
    ]
    config = {
        "working_space": "test",
        "all_roll_fit": {"matrix_coefficient": [-2, 2], "channel_bias": [-0.25, 0.25], "ridge": 0.001},
        "ood": {"threshold_quantile": 0.95},
        "transplant_strengths_strongest_first": [1.0, 0.5],
        "anchor_derived_floors_frozen_before_candidate_render": {
            "minimum_candidate_gold_median_style_delta_e76": 0.0,
            "minimum_candidate_gold_median_residual_delta_e76_after_matched_basic": 0.0,
        },
        "gates": {
            "minimum_gold_ood_eligible_images": 1,
            "minimum_stress_ood_eligible_fraction": 1.0,
            "maximum_gold_per_image_raw_clip_fraction": 1.0,
        },
    }
    result = evaluate_transplant(archive_frames=frames, digital_samples=digital, config=config)
    assert result["automatic_passed"] is False
    assert result["decision"] == "ood_coverage_fail"
