from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.filmmatch_paired_source import (
    FilmMatchSourceAuditError,
    ordered_mapping_audit,
    sequence_number,
    validate_download_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2aw0_filmmatch_ektachrome_paired_source_v1.json"
    ).read_text(encoding="utf-8")
)


def test_sequence_number_is_exact() -> None:
    assert sequence_number("Still 2025-08-21 150231_1.315.1.tif") == 315
    with pytest.raises(FilmMatchSourceAuditError):
        sequence_number("315.tif")


def test_ordered_mapping_audit_accepts_shared_order() -> None:
    rng = np.random.default_rng(7)
    film = rng.normal(size=(17, 12))
    digital = film * np.linspace(0.5, 2.0, 12) + np.linspace(-3.0, 4.0, 12)
    result = ordered_mapping_audit(
        film,
        digital,
        epsilon=1e-6,
        permutations=1000,
        permutation_seed=20260730,
    )
    assert result["best_circular_shift"] == 0
    assert result["permutation_p"] <= 0.01
    assert result["cost_to_permutation_median_ratio"] < 0.01


def test_ordered_mapping_audit_exposes_shifted_order() -> None:
    rng = np.random.default_rng(11)
    film = rng.normal(size=(13, 8))
    digital = np.roll(film, 3, axis=0)
    result = ordered_mapping_audit(
        film,
        digital,
        epsilon=1e-6,
        permutations=100,
        permutation_seed=3,
    )
    assert result["best_circular_shift"] != 0


def test_manifest_validation_rejects_identity_before_io(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "wrong",
        "experiment_id": CONFIG["experiment_id"],
        "file_count": 140,
        "bytes": 4_255_132_151,
        "files": [],
    }
    with pytest.raises(FilmMatchSourceAuditError):
        validate_download_manifest(manifest, root=tmp_path, config=CONFIG)


def test_manifest_validation_rejects_unsafe_path_before_io(tmp_path: Path) -> None:
    config = copy.deepcopy(CONFIG)
    config["acquisition"]["expected_total_files"] = 1
    config["acquisition"]["expected_total_bytes"] = 1
    config["acquisition"]["folders"] = [
        {
            "lane": "sony_reflective",
            "expected_files": 1,
            "expected_sequence": [233, 233],
        }
    ]
    manifest = {
        "schema_version": "u5-r2aw0-filmmatch-download-manifest-v1",
        "experiment_id": config["experiment_id"],
        "file_count": 1,
        "bytes": 1,
        "files": [
            {
                "lane": "sony_reflective",
                "name": "Still 2025-09-08 144458_1.233.1.tif",
                "relative_path": "../foreign.tif",
                "bytes": 1,
                "sha256": "0" * 64,
            }
        ],
    }
    with pytest.raises(FilmMatchSourceAuditError):
        validate_download_manifest(manifest, root=tmp_path, config=config)
