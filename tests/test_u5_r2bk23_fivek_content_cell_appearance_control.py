from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.eval.fivek_content_cell_appearance_control import (
    FiveKContentCellControlError,
    _policy_result,
    _sample_tiff,
    load_config,
    load_row_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "u5_r2bk23_fivek_content_cell_appearance_control_v1.json"
)


def test_contract_binds_bk22_and_research_only_fivek() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["parent"]["required_decision"] == (
        "retain_content_cell_worst_case_gate_prior"
    )
    assert config["source"]["licence_scope"] == "research-only, non-commercial"
    assert config["sampling"]["paired_pixel_coordinates_used"] is False
    assert config["policy"]["minimum_cell_pass_fraction"] == 1.0


def test_inventory_is_disjoint_licensed_and_complete() -> None:
    config = load_config(ROOT, CONFIG)
    development, confirmation = load_row_inventory(ROOT, config)
    assert len(development) == len(confirmation) == 64
    assert not (
        {row["cell_id"] for row in development}
        & {row["cell_id"] for row in confirmation}
    )
    assert all((ROOT / row["source_path"]).is_file() for row in development)
    assert all((ROOT / row["target_path"]).is_file() for row in confirmation)


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["parent"]["decision_sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FiveKContentCellControlError, match="hash drift"):
        load_config(ROOT, path)


def test_tiff_sampling_is_repeat_exact_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "sample.tif"
    values = np.arange(16 * 12 * 3, dtype=np.uint16).reshape(16, 12, 3)
    tifffile.imwrite(path, values, photometric="rgb")
    first, digest = _sample_tiff(
        path, count=32, seed=12, expected_sha256=None
    )
    second, _ = _sample_tiff(
        path, count=32, seed=12, expected_sha256=digest
    )
    assert np.array_equal(first, second)
    with pytest.raises(FiveKContentCellControlError, match="hash drift"):
        _sample_tiff(path, count=32, seed=12, expected_sha256="0" * 64)


def test_policy_preserves_k1_and_rejects_confounded_view() -> None:
    config = load_config(ROOT, CONFIG)
    valid_policy, valid_reasons = _policy_result(
        config,
        disagreement=0.01,
        pooled_improvement=0.5,
        cell_improvements=[0.4] * 63 + [0.05],
    )
    assert valid_policy == "identity_fallback"
    assert valid_reasons == ["content-cell-worst-case"]
    positive_policy, positive_reasons = _policy_result(
        config,
        disagreement=0.01,
        pooled_improvement=0.5,
        cell_improvements=[0.4] * 64,
    )
    assert positive_policy == "hard_explicit_candidate"
    assert positive_reasons == []
