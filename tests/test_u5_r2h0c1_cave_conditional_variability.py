from __future__ import annotations

import numpy as np

from src.eval.cave_conditional_variability import (
    RepresentativePopulation,
    _normalized_official_path,
    array_sha256,
    scene_bootstrap,
    select_conditional_pairs,
)


def _config() -> dict:
    return {
        "seed": 7,
        "pair_policy": {
            "input_delta_e76_max": 1.0,
            "spectral_rms_min": 0.01,
            "pairs_per_scene_pair_max": 16,
            "pair_incidences_per_scene_max": 64,
        },
    }


def test_official_path_normalization_removes_archive_duplicate_scene() -> None:
    assert (
        _normalized_official_path("face_ms/face_ms/face_ms_01.png")
        == "face_ms/face_ms_01.png"
    )
    assert _normalized_official_path("face_ms/face_ms/") == "face_ms/"


def test_pair_selection_is_cross_scene_unique_and_deterministic() -> None:
    spectra = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
        ]
    )
    lab = np.array([[50.0, 0.0, 0.0], [50.1, 0.0, 0.0], [50.2, 0.0, 0.0], [50.3, 0.0, 0.0]])
    population = RepresentativePopulation(
        spectra=spectra,
        xyz=np.zeros((4, 3)),
        lab=lab,
        scene=np.array(["a", "b", "a", "c"]),
        cell_row=np.zeros(4, dtype=np.int16),
        cell_column=np.arange(4, dtype=np.int16),
    )
    first = select_conditional_pairs(population, _config())
    second = select_conditional_pairs(population, _config())
    assert np.array_equal(first.first, second.first)
    assert np.array_equal(first.second, second.second)
    assert len(set(first.first) | set(first.second)) == 2 * len(first.first)
    assert np.all(population.scene[first.first] != population.scene[first.second])


def test_scene_bootstrap_is_repeatable_and_array_hash_includes_shape() -> None:
    population = RepresentativePopulation(
        spectra=np.zeros((6, 3)),
        xyz=np.zeros((6, 3)),
        lab=np.zeros((6, 3)),
        scene=np.array(["a", "b", "a", "c", "b", "c"]),
        cell_row=np.zeros(6, dtype=np.int16),
        cell_column=np.arange(6, dtype=np.int16),
    )
    pairs = type("Pairs", (), {"first": np.array([0, 2, 4]), "second": np.array([1, 3, 5])})()
    values = np.array([1.0, 2.0, 3.0])
    first = scene_bootstrap(population, pairs, values, 20, 11)
    second = scene_bootstrap(population, pairs, values, 20, 11)
    assert first == second
    assert array_sha256(np.zeros((2, 3))) != array_sha256(np.zeros((3, 2)))
