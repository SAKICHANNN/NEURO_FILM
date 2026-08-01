from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.spektrafilm_langmuir import (
    adjudicate_blind_observations,
    evaluate_manifests,
    paired_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u5_r2bs0_spektrafilm_langmuir_ablation_v1.json").read_text(
        encoding="utf-8"
    )
)


def _image() -> np.ndarray:
    y, x = np.mgrid[:32, :48]
    return np.stack(
        (
            0.1 + 0.8 * x / 47,
            0.15 + 0.7 * y / 31,
            0.2 + 0.5 * (x + y) / 78,
        ),
        axis=-1,
    )


def test_frozen_config_binds_latest_external_mechanism() -> None:
    assert CONFIG["external_source"]["revision"] == (
        "6cd00c8d4f30b5b550f50f4bbd3753c9f2a48507"
    )
    assert CONFIG["arms"]["linear_donor"]["langmuir_donor_k_rgb"] == (
        "infinite_linear_limit"
    )
    assert CONFIG["arms"]["langmuir_donor"]["langmuir_donor_k_rgb"] == [
        1.0,
        1.0,
        1.0,
    ]
    assert CONFIG["diagnostics"]["metric_sample_pixels_per_image"] == 65536


def test_identity_pair_has_zero_mechanism_effect() -> None:
    image = _image()
    metrics = paired_metrics(image, image, image, CONFIG)
    assert metrics["linear_finite_and_bounded"]
    assert metrics["langmuir_finite_and_bounded"]
    assert metrics["langmuir_effect_delta_e76"] == 0.0
    assert metrics["new_hard_boundary_fraction_vs_linear"] == 0.0
    assert metrics["new_isolated_red_speckle_fraction_vs_linear"] == 0.0


def test_evaluator_requires_distinct_external_runs() -> None:
    manifest = {
        "schema_version": "u5-r2bs0-external-render-manifest-v1",
        "run_id": "same",
        "external_revision": CONFIG["external_source"]["revision"],
        "external_file_sha256": CONFIG["external_source"]["file_sha256"],
        "python_version": CONFIG["runtime"]["python_version"],
        "input_manifest_sha256": CONFIG["inputs"]["frozen_set_sha256"],
        "package_versions": {},
        "records": [{}] * CONFIG["inputs"]["expected_samples"],
    }
    with pytest.raises(ValueError, match="distinct external run IDs"):
        evaluate_manifests(manifest, manifest, CONFIG, root=ROOT)


def test_blind_adjudication_decodes_pairwise_preferences() -> None:
    observations = {
        "mapping_read_before_observations": False,
        "records": [
            {
                "sample_id": "x",
                "ranking": ["B", "A", "C"],
                "severe_by_candidate": {"A": False, "B": False, "C": False},
                "note": "locked",
            }
        ],
    }
    result = adjudicate_blind_observations(
        observations,
        {"x": ["linear", "langmuir", "ao6"]},
    )
    assert result["pairwise_preference_counts"]["langmuir_over_linear"] == 1
    assert result["pairwise_preference_counts"]["langmuir_over_ao6"] == 1
    assert result["decision"] == "retain_external_mechanism_development_evidence"
