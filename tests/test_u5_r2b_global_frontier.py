from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.global_frontier import (
    GlobalFrontierError,
    evaluate_manifest,
    new_hard_clipping_fraction,
)


def _write_rgb(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_hard_clipping_ignores_source_endpoints() -> None:
    source = np.array([[0.0, 0.5, 1.0], [0.5, 0.5, 0.5]])
    output = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    assert new_hard_clipping_fraction(source, output, 0.001) == pytest.approx(2 / 6)


def test_new_hard_clipping_rejects_shape_mismatch() -> None:
    with pytest.raises(GlobalFrontierError, match="grids differ"):
        new_hard_clipping_fraction(np.zeros((2, 3)), np.zeros((3, 3)), 0.0)


def test_evaluate_manifest_requires_complete_fixed_bank(tmp_path: Path) -> None:
    renderer = tmp_path / "renderer.py"
    renderer.write_text("# frozen\n", encoding="utf-8")
    source = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 3)).reshape(8, 8, 3)
    source_path = tmp_path / "source.png"
    source_hash = _write_rgb(source_path, source)
    frozen = {
        "frozen_set": {
            "samples": [
                {
                    "id": "g",
                    "split": "gold",
                    "availability": "available",
                    "source_path": "source.png",
                    "source_sha256": source_hash,
                }
            ]
        }
    }
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "frozen_set_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "frozen_set": "frozen.json",
        "frozen_set_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "expected_gold_samples": 1,
        "expected_stress_samples": 0,
        "renderer_script": "renderer.py",
        "renderer_script_sha256": hashlib.sha256(renderer.read_bytes()).hexdigest(),
        "primary_candidate_ids": ["candidate"],
        "metrics": {
            "maximum_pixels_per_image": 64,
            "new_hard_clipping_epsilon": 0.001,
            "minimum_gold_median_style_delta_e76": 0.0,
            "minimum_gold_median_non_basic_residual_delta_e76": 0.0,
            "maximum_worst_gold_new_hard_clipping_fraction": 1.0,
        },
    }
    with pytest.raises(GlobalFrontierError, match="incomplete"):
        evaluate_manifest(root=tmp_path, config=config, manifest_path=manifest_path)
