from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_u5_r2s4_diversified_distribution_operator_development import (
    _make_condition_distributions,
    _method_groups,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (
            ROOT
            / "configs"
            / "u5_r2s4_diversified_distribution_operator_development_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_condition_distributions_repeat_without_cross_domain_reuse() -> None:
    config = _config()
    content = dict(config["content_condition_generator"])
    content["scenes_per_condition_per_domain"] = 1
    content["samples_per_scene"] = 16
    axis_size = int(config["operator"]["velocity_grid_axis_size"])
    grids = np.zeros((1, axis_size, axis_size, axis_size, 3))
    first = _make_condition_distributions(
        grids, seed=17, content_config=content, integration_steps=2
    )
    second = _make_condition_distributions(
        grids, seed=17, content_config=content, integration_steps=2
    )
    for left_styles, right_styles in zip(first, second, strict=True):
        assert len(left_styles) == 1
        assert len(left_styles[0]) == int(content["condition_count"])
        for left, right in zip(left_styles[0], right_styles[0], strict=True):
            assert left.shape == (16, 3)
            assert np.array_equal(left, right)
    neutral, styled = first
    assert all(
        not np.array_equal(left, right)
        for left, right in zip(neutral[0], styled[0], strict=True)
    )


def test_method_groups_keep_fixed_control_semantics() -> None:
    sources = [np.full((4, 3), value) for value in (0.1, 0.2, 0.3, 0.4)]
    targets = [np.full((4, 3), value) for value in (0.5, 0.6, 0.7, 0.8)]
    pooled_sources, pooled_targets = _method_groups(
        {"id": "pooled_rff_mmd_192"}, sources, targets
    )
    assert len(pooled_sources) == len(pooled_targets) == 1
    assert pooled_sources[0].shape == pooled_targets[0].shape == (16, 3)
    correct_sources, correct_targets = _method_groups(
        {"id": "conditional_rff_mmd_192"}, sources, targets
    )
    assert correct_sources is sources
    assert correct_targets is targets
    _, shuffled_targets = _method_groups(
        {
            "id": "shuffled_condition_rff_mmd_192",
            "target_condition_permutation": [1, 3, 0, 2],
        },
        sources,
        targets,
    )
    assert all(
        actual is targets[index]
        for actual, index in zip(shuffled_targets, [1, 3, 0, 2], strict=True)
    )
