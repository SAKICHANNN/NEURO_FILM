from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.content_cell_appearance_gate import (
    ContentCellAppearanceGateError,
    load_config,
    make_cell_distributions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk22_content_cell_appearance_gate_v1.json"


def _set_hash(cells: list[np.ndarray]) -> str:
    values = np.concatenate(cells, axis=0)
    rounded = np.round(values, decimals=14)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return hashlib.sha256(
        np.ascontiguousarray(rounded[order], dtype="<f8").tobytes()
    ).hexdigest()


def test_contract_binds_bk21_and_external_cells() -> None:
    config = load_config(ROOT, CONFIG)
    assert config["parent"]["required_decision"] == "close_frozen_consensus_policy"
    assert config["content_cells"]["identity_source"].startswith(
        "generated latent scene identity"
    )
    assert config["policy"]["minimum_cell_pass_fraction"] == 1.0
    assert config["optimization"]["device"] == "cpu"


def test_invalid_permutation_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["content_cells"]["confounded_target_permutation"] = [0, 1, 2, 3]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContentCellAppearanceGateError, match="invalid"):
        load_config(ROOT, path)


def test_valid_and_confounded_views_have_exact_pooled_sets() -> None:
    config = load_config(ROOT, CONFIG)
    cells = make_cell_distributions(config)
    assert _set_hash(cells["valid_target"]) == _set_hash(
        cells["confounded_target"]
    )
    assert np.array_equal(
        np.concatenate(cells["valid_target"], axis=0),
        np.concatenate(cells["confounded_target"], axis=0)[
            np.concatenate(
                [
                    np.arange(
                        index * config["population"]["samples_per_cell"],
                        (index + 1) * config["population"]["samples_per_cell"],
                    )
                    for index in [3, 0, 1, 2]
                ]
            )
        ],
    )


def test_cell_correspondence_changes_without_changing_arrays() -> None:
    config = load_config(ROOT, CONFIG)
    cells = make_cell_distributions(config)
    permutation = config["content_cells"]["confounded_target_permutation"]
    for index, target_index in enumerate(permutation):
        assert np.array_equal(
            cells["confounded_target"][index],
            cells["valid_target"][target_index],
        )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            cells["valid_target"], cells["confounded_target"], strict=True
        )
    )


def test_generation_is_repeat_exact() -> None:
    config = load_config(ROOT, CONFIG)
    first = make_cell_distributions(config)
    second = make_cell_distributions(config)
    for key in first:
        assert all(
            np.array_equal(left, right)
            for left, right in zip(first[key], second[key], strict=True)
        )
