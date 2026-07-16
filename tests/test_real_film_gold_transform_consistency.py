from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.real_film.gold_transform_consistency import (
    PairedFrameSamples,
    _positive_orientation,
    balanced_training_arrays,
    fit_per_channel_affine,
    fit_seplut17,
    load_paired_frame_samples,
    operator_from_dict,
    run_whole_roll_evaluation,
)


BOUNDS = {
    "channel_gain": [0.5, 2.0],
    "channel_bias": [-0.25, 0.25],
    "matrix_coefficient": [-2.0, 2.0],
    "lut_knots": 17,
    "lut_output": [0.0, 1.0],
    "ridge": 0.001,
}


def _frame(frame_id: str, roll_id: str, source: np.ndarray, target: np.ndarray) -> PairedFrameSamples:
    return PairedFrameSamples(frame_id, roll_id, source.reshape(-1, 3), target.reshape(-1, 3))


def test_grid_sampling_is_deterministic_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "pixels"
    preview = root / "negative" / "frame.png"
    proxy = root / "proxy" / "frame.png"
    preview.parent.mkdir(parents=True)
    proxy.parent.mkdir(parents=True)
    values = np.arange(12 * 10 * 3, dtype=np.uint16).reshape(10, 12, 3).astype(np.uint8)
    Image.fromarray(values).save(preview)
    Image.fromarray(values[2:8, 3:11]).save(proxy)
    records = [{
        "frame_id": "frame",
        "roll_id": "roll",
        "preview_path": "negative/frame.png",
        "proxy_path": "proxy/frame.png",
        "bbox": [3, 2, 11, 8],
    }]
    first = load_paired_frame_samples(download_root=root, pair_records=records, maximum_pixels_per_frame=64)
    second = load_paired_frame_samples(download_root=root, pair_records=records, maximum_pixels_per_frame=64)
    assert 4 <= len(first[0].source) <= 64
    assert np.array_equal(first[0].source, first[0].target)
    assert np.array_equal(first[0].source, second[0].source)


def test_balancing_gives_equal_roll_and_frame_mass() -> None:
    pixels = np.linspace(0.0, 1.0, 12).reshape(4, 3)
    frames = [
        _frame("a1", "a", pixels, pixels),
        _frame("a2", "a", np.tile(pixels, (2, 1)), np.tile(pixels, (2, 1))),
        _frame("b1", "b", pixels, pixels),
    ]
    _, _, weights = balanced_training_arrays(frames)
    assert np.isclose(weights[:4].sum(), 0.25)
    assert np.isclose(weights[4:12].sum(), 0.25)
    assert np.isclose(weights[12:].sum(), 0.5)


def test_bounded_channel_affine_recovers_known_mapping() -> None:
    rng = np.random.default_rng(4)
    source = rng.uniform(0.1, 0.7, size=(500, 3))
    target = source * np.array([1.2, 0.8, 1.1]) + np.array([0.03, 0.08, -0.02])
    operator = fit_per_channel_affine([_frame("f", "r", source, target)], BOUNDS, "test")
    assert np.allclose(np.diag(operator.matrix), [1.2, 0.8, 1.1], atol=1e-8)
    assert np.allclose(operator.bias, [0.03, 0.08, -0.02], atol=1e-8)


def test_seplut_is_strictly_monotone_and_models_nonlinearity() -> None:
    rng = np.random.default_rng(5)
    source = rng.uniform(0.0, 1.0, size=(4000, 3))
    target = np.clip(source ** np.array([0.65, 1.35, 0.8]), 0.0, 1.0)
    frames = [_frame("f", "r", source, target)]
    affine = fit_per_channel_affine(frames, BOUNDS, "test")
    seplut = fit_seplut17(frames, BOUNDS, "test")
    assert np.all(np.diff(seplut.y_knots, axis=1) > 0.0)
    assert np.linalg.det(seplut.affine.matrix) > 0.0
    assert np.mean(np.abs(seplut.apply(source) - target)) < np.mean(np.abs(affine.apply(source) - target))
    restored = operator_from_dict(seplut.to_dict())
    assert np.array_equal(restored.apply(source), seplut.apply(source))


def test_orientation_repair_is_positive_and_deterministic() -> None:
    matrix = np.diag([-1.0, 1.0, 1.0])
    first, alpha = _positive_orientation(matrix)
    second, second_alpha = _positive_orientation(matrix)
    assert alpha == second_alpha
    assert np.array_equal(first, second)
    assert np.linalg.det(first) > 0.0


def test_whole_roll_evaluation_is_deterministic() -> None:
    rng = np.random.default_rng(6)
    frames = []
    for roll_index in range(3):
        source = rng.uniform(0.02, 0.98, size=(600, 3))
        target = np.clip(source ** np.array([0.7, 1.25, 0.85]), 0.0, 1.0)
        frames.append(_frame(f"f{roll_index}", f"r{roll_index}", source, target))
    config = {
        "working_space": "test",
        "operator_bounds": BOUNDS,
        "evaluation": {"bootstrap_seed": 1411, "cluster_bootstrap_resamples": 100},
        "gates": {
            "minimum_primary_relative_improvement_over_simple_affine": 0.0,
            "minimum_rolls_improved_over_simple_affine": 1,
            "minimum_cluster_bootstrap_lower_improvement": -1.0,
            "minimum_rolls_beating_median_wrong_roll_operator": 0,
            "minimum_median_render_delta_e76_from_identity": 0.0,
            "maximum_output_clip_fraction_increase": 1.0,
        },
    }
    first = run_whole_roll_evaluation(frames, config)
    second = run_whole_roll_evaluation(frames, config)
    assert first == second
    assert first["rolls_improved_over_simple_affine"] == 3
