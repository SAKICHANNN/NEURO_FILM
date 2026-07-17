from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.filmstylesafe_synthetic import (
    SyntheticFailureError,
    canonical_parameter_hash,
    highlight_chroma_island_v0,
    highlight_chroma_speckle_v1,
    run_highlight_chroma_island_v0,
    run_highlight_chroma_speckle_v1,
)

ROOT = Path(__file__).resolve().parents[1]


def test_highlight_chroma_island_is_deterministic(tmp_path: Path) -> None:
    src = Image.new("RGB", (64, 64), (240, 240, 240))
    params = dict(
        seed=7,
        center_xy_norm=(0.5, 0.5),
        radius_norm=0.25,
        luma_threshold=0.5,
        chroma_boost=1.0,
    )
    a = highlight_chroma_island_v0(src, **params)
    b = highlight_chroma_island_v0(src, **params)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(src))


def test_empty_mask_fails_closed() -> None:
    src = Image.new("RGB", (32, 32), (10, 10, 10))
    with pytest.raises(SyntheticFailureError, match="empty"):
        highlight_chroma_island_v0(
            src,
            seed=1,
            center_xy_norm=(0.5, 0.5),
            radius_norm=0.2,
            luma_threshold=0.95,
            chroma_boost=1.0,
        )


def test_run_writes_png_and_hashes(tmp_path: Path) -> None:
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (48, 48), (250, 250, 250)).save(inp)
    params = {
        "operator_id": "explicit-highlight-chroma-island-v0",
        "seed": 3,
        "center_xy_norm": [0.5, 0.5],
        "radius_norm": 0.3,
        "luma_threshold": 0.2,
        "chroma_boost": 1.2,
    }
    result = run_highlight_chroma_island_v0(inp, out, params)
    assert out.is_file()
    assert result["parameter_hash"] == canonical_parameter_hash(params)
    assert len(result["output_hash"]) == 64


def test_highlight_chroma_speckle_is_deterministic_and_sparse(tmp_path: Path) -> None:
    src = Image.new("RGB", (64, 64), (245, 245, 245))
    params = dict(
        seed=11,
        center_xy_norm=(0.5, 0.5),
        radius_norm=0.4,
        luma_threshold=0.5,
        chroma_boost=1.2,
        speckle_density=0.05,
        blob_radius_px=0,
    )
    a = highlight_chroma_speckle_v1(src, **params)
    b = highlight_chroma_speckle_v1(src, **params)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    changed = np.any(np.asarray(a) != np.asarray(src), axis=2)
    assert 1 <= int(changed.sum()) < 64 * 64 // 4


def test_run_speckle_writes_png_and_hashes(tmp_path: Path) -> None:
    inp = tmp_path / "in.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (48, 48), (250, 250, 250)).save(inp)
    params = {
        "operator_id": "explicit-highlight-chroma-speckle-v1",
        "seed": 5,
        "center_xy_norm": [0.5, 0.5],
        "radius_norm": 0.35,
        "luma_threshold": 0.2,
        "chroma_boost": 1.4,
        "speckle_density": 0.08,
        "blob_radius_px": 1,
    }
    result = run_highlight_chroma_speckle_v1(inp, out, params)
    assert out.is_file()
    assert result["parameter_hash"] == canonical_parameter_hash(params)
    assert len(result["output_hash"]) == 64


def test_r1b4_inventory_binds_executed_synthetic_member() -> None:
    from src.eval.filmstylesafe_r1b import load_r1b_contract, validate_suite_member, audit_suite_leakage
    from src.roll2film.blueneg_download import sha256_file

    contract = load_r1b_contract(ROOT / "configs" / "filmstylesafe_r1b_contract_v1.json")
    inventory = json.loads(
        (ROOT / "configs" / "filmstylesafe_r1b4_a0_inventory_v1.json").read_text(encoding="utf-8")
    )
    for row in inventory["members"]:
        validate_suite_member(row, contract)
    assert audit_suite_leakage(inventory["members"])["passed"] is True
    synth = next(row for row in inventory["members"] if row["member_id"] == "a0-synth-speckle-proto-001")
    assert synth["binding_status"] == "bound"
    assert synth["operator_card"]["implementation_status"] == "executed_prototype_v0"
    path = ROOT / synth["artifact_path"]
    assert path.is_file()
    assert sha256_file(path) == synth["exact_hash"]


def test_r1b5_inventory_binds_speckle_and_rf2c0_control() -> None:
    from src.eval.filmstylesafe_r1b import load_r1b_contract, validate_suite_member, audit_suite_leakage
    from src.roll2film.blueneg_download import sha256_file

    contract = load_r1b_contract(ROOT / "configs" / "filmstylesafe_r1b_contract_v1.json")
    inventory = json.loads(
        (ROOT / "configs" / "filmstylesafe_r1b5_a0_inventory_v1.json").read_text(encoding="utf-8")
    )
    for row in inventory["members"]:
        validate_suite_member(row, contract)
    assert audit_suite_leakage(inventory["members"])["passed"] is True
    synth = next(row for row in inventory["members"] if row["member_id"] == "a0-synth-speckle-hf-001")
    assert synth["binding_status"] == "bound"
    assert synth["operator_card"]["operator_id"] == "explicit-highlight-chroma-speckle-v1"
    assert sha256_file(ROOT / synth["artifact_path"]) == synth["exact_hash"]
    control = next(
        row for row in inventory["members"] if row["member_id"] == "a0-rf2c0-ektar-fixed-e0-01"
    )
    assert control["role"] == "external_style_control"
    assert control["binding_status"] == "bound"
    assert sha256_file(ROOT / control["artifact_path"]) == control["exact_hash"]
