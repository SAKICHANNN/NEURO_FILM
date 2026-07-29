from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2aq2_colorreference_velvia100f_held_group_baselines import (
    CONFIG_SHA256,
    _fold_masks,
    load_config,
)
from src.roll2film.colorreference_proxy_baselines import (
    apply_proxy_model,
    fit_proxy_model,
    xyz_to_lab_d50,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq2_colorreference_velvia100f_held_group_baselines_v1.json"
)


def test_contract_freezes_joint_group_holdout() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["split_contract"]["primary_fold_count"] == 30
    assert (
        config["split_contract"]["primary_expected_development_rows"]
        == 5760
    )
    assert config["fit_allowed"] is True
    assert config["training_allowed"] is False
    assert config["render_allowed"] is False


def test_joint_split_excludes_held_set_and_slide() -> None:
    sets = np.repeat(np.array([1, 2, 3, 4, 5, 9]), 5 * 288)
    slides = np.tile(np.repeat(np.arange(1, 6), 288), 6)
    table = {"test_set": sets, "slide_index": slides}
    development, confirmation = _fold_masks(
        table, "joint", held_set=3, held_slide=2
    )
    assert int(development.sum()) == 5760
    assert int(confirmation.sum()) == 288
    assert not np.any(development & confirmation)
    assert np.all(sets[development] != 3)
    assert np.all(slides[development] != 2)


@pytest.mark.parametrize(
    "model_id",
    [
        "diagonal_affine",
        "nonnegative_affine",
        "full_affine",
        "quadratic_full",
    ],
)
def test_explicit_models_recover_known_affine(model_id: str) -> None:
    rng = np.random.default_rng(20260729)
    source = rng.uniform(0.0, 1.0, size=(128, 3))
    matrix = np.array(
        [[0.6, 0.1, 0.05], [0.05, 0.7, 0.1], [0.1, 0.05, 0.65]]
    )
    if model_id == "diagonal_affine":
        matrix = np.diag(np.diag(matrix))
    target = source @ matrix.T + np.array([0.02, 0.03, 0.01])
    model = fit_proxy_model(
        source,
        target,
        model_id=model_id,
        quadratic_ridge=1e-8,
        nonnegative_maximum_iterations=1000,
    )
    prediction = apply_proxy_model(model, source)
    np.testing.assert_allclose(prediction, target, atol=2e-8)


def test_xyz_to_lab_matches_reference_white() -> None:
    lab = xyz_to_lab_d50(np.array([[0.9642, 1.0, 0.8251]]))
    np.testing.assert_allclose(lab, np.array([[100.0, 0.0, 0.0]]), atol=1e-12)
    black = xyz_to_lab_d50(np.zeros((1, 3)))
    np.testing.assert_allclose(black, np.zeros((1, 3)), atol=1e-12)


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["render_allowed"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
