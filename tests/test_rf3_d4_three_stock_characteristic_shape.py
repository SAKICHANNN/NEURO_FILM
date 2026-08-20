from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.real_film.three_stock_characteristic_shape import (
    ThreeStockCharacteristicShapeError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "rf3_d4_three_stock_characteristic_shape_v1.json"


def test_rf3_d4_contract_and_formal_result(tmp_path: Path) -> None:
    report = evaluate(load_contract(CONTRACT), ROOT, overlay_dir=tmp_path / "overlays")
    assert report["schema"] == "neuro-film.rf3-three-stock-characteristic-shape-report.v1"
    assert len(report["traces"]) == 3
    assert len(report["pairwise_comparisons"]) == 3
    assert report["automatic_pass"] is False
    assert report["decision"] == (
        "close_datasheet_characteristic_shape_as_a_three_stock_discriminator_and_require_pixel_evidence"
    )


def test_rf3_d4_replay_is_exact(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    left = evaluate(contract, ROOT, overlay_dir=tmp_path / "a")
    right = evaluate(contract, ROOT, overlay_dir=tmp_path / "b")
    assert left["stable_evidence_id"] == right["stable_evidence_id"]
    assert left["pairwise_comparisons"] == right["pairwise_comparisons"]
    assert left["overlay_sha256"] == right["overlay_sha256"]


def test_rf3_d4_rejects_nonmonotone_trace(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_contract(CONTRACT))
    contract["sources"][1]["sample_midpoint_y_pixels"][3] = 800.0
    with pytest.raises(ThreeStockCharacteristicShapeError, match="not strict"):
        evaluate(contract, ROOT, overlay_dir=tmp_path)


def test_rf3_d4_rejects_source_hash_drift(tmp_path: Path) -> None:
    contract = copy.deepcopy(load_contract(CONTRACT))
    contract["sources"][0]["page_image_sha256"] = "0" * 64
    with pytest.raises(ThreeStockCharacteristicShapeError, match="page drift"):
        evaluate(contract, ROOT, overlay_dir=tmp_path)
