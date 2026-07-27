from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.dorf_film_response import (
    parse_curves,
    read_archive,
    strict_rgb_triplets,
)
from src.eval.dorf_response_diversity import (
    _median_de,
    apply_triplet,
    shared_curve_output,
    synthetic_rgb,
)


ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[dict, dict, dict]:
    config = json.loads(
        (ROOT / "configs/u5_r2ab1_dorf_response_diversity_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(
        (ROOT / config["source"]["archive_config"]).read_text(encoding="utf-8")
    )
    payload, _ = read_archive(ROOT / source["source"]["path"])
    triplets = strict_rgb_triplets(
        parse_curves(payload),
        eligible_scales=set(config["source"]["eligible_scale_labels"]),
    )
    return config, source, triplets


def test_frozen_population_and_candidate_count() -> None:
    config, _source, triplets = _inputs()
    rgb = synthetic_rgb(config["synthetic_population"]["encoded_rgb_levels"])
    assert rgb.shape == (1331, 3)
    assert len(triplets) == 46
    assert all(
        set(channels) == {"red", "green", "blue"}
        for channels in triplets.values()
    )


def test_all_raw_candidates_are_finite_and_inside_cube() -> None:
    config, _source, triplets = _inputs()
    rgb = synthetic_rgb(config["synthetic_population"]["encoded_rgb_levels"])
    for channels in triplets.values():
        output = apply_triplet(rgb, channels)
        assert output.shape == rgb.shape
        assert np.all(np.isfinite(output))
        assert np.min(output) >= 0.0
        assert np.max(output) <= 1.0


def test_exact_per_channel_monotone_ceiling_is_candidate() -> None:
    config, _source, triplets = _inputs()
    rgb = synthetic_rgb(config["synthetic_population"]["encoded_rgb_levels"])
    channels = triplets["Gold-100CD"]
    first = apply_triplet(rgb, channels)
    second = apply_triplet(rgb, channels)
    np.testing.assert_array_equal(first, second)


def test_shared_curve_control_uses_one_curve_for_all_channels() -> None:
    _config, _source, triplets = _inputs()
    levels = np.asarray([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.8, 0.8, 0.8]])
    output = shared_curve_output(levels, triplets["Portra-400VCCD"])
    np.testing.assert_array_equal(output[:, 0], output[:, 1])
    np.testing.assert_array_equal(output[:, 1], output[:, 2])


def test_delta_e_is_computed_once_in_lab_space() -> None:
    black = np.zeros((1, 3), dtype=np.float64)
    white = np.ones((1, 3), dtype=np.float64)
    assert abs(_median_de(black, white) - 100.0) < 1e-5
