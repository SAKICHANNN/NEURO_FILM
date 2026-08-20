from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.three_stock_mtf_prior import (
    ThreeStockMtfError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/rf3_d3_three_stock_mtf_prior_v1.json"


def test_contract_freezes_stock_order_and_materiality_gates() -> None:
    contract = load_contract(CONTRACT)
    assert [row["stock_id"] for row in contract["sources"]] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert contract["gates"][
        "minimum_pairwise_median_absolute_log10_response_difference"
    ] == 0.03


def test_contract_rejects_gate_relaxation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["gates"]["minimum_pairwise_rmse_log10_response_difference"] = 0.0
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ThreeStockMtfError, match="contract drift"):
        load_contract(path)


@pytest.mark.skipif(
    not all((ROOT / row["page_image"]).is_file() for row in load_contract(CONTRACT)["sources"]),
    reason="RF3.D3 exact page rasters are unavailable",
)
def test_exact_mtf_comparison_is_repeatable(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = evaluate(contract, ROOT, overlay_dir=tmp_path / "first")
    second = evaluate(contract, ROOT, overlay_dir=tmp_path / "second")
    assert first == second
    assert first["decision"] in {
        contract["decision_if_pass"],
        contract["decision_if_fail"],
    }
    assert len(first["pairwise_comparisons"]) == 3


def test_parent_hash_drift_fails(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    project = tmp_path / "project"
    parent = project / contract["parent"]["path"]
    parent.parent.mkdir(parents=True)
    parent.write_text("{}", encoding="utf-8")
    with pytest.raises(ThreeStockMtfError, match="parent evidence drift"):
        evaluate(contract, project, overlay_dir=tmp_path / "overlays")
