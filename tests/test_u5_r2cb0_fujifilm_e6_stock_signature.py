from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_e6_stock_signature import (
    CONTRACT_SHA256,
    FujifilmE6SignatureError,
    _axis_residual,
    _curve_values,
    audit_signature,
    hash_file,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb0_fujifilm_e6_stock_source_signature_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    contract = load_contract(CONFIG)
    assert contract["velvia50_parent_trace"]["forbidden_observation"] == (
        "mtf_curve_or_axis_calibration"
    )


def test_log_axis_and_curve_are_deterministic() -> None:
    row = {
        "graph_axes": {
            "x_value_pixels": [[1.0, 0.0], [100.0, 100.0]],
            "y_value_pixels": [[100.0, 0.0], [1.0, 100.0]],
        },
        "curve": [[0, 0], [50, 50], [100, 100]],
    }
    frequencies, responses = _curve_values(row)
    assert np.allclose(frequencies, [1.0, 10.0, 100.0])
    assert np.allclose(responses, [100.0, 10.0, 1.0])
    assert _axis_residual([[1.0, 0.0], [10.0, 50.0], [100.0, 100.0]], row["graph_axes"]["x_value_pixels"]) == pytest.approx(0.0)


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmE6SignatureError, match="contract hash drift"):
        load_contract(changed)


def test_formal_source_audit_when_exact_assets_exist(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    assets = [ROOT / contract["trace"]["path"]]
    for row in __import__("json").loads(assets[0].read_text(encoding="utf-8"))["stocks"].values():
        assets.extend(
            [ROOT / row["source_pdf"]["path"], ROOT / row["source_graph"]["path"]]
        )
        if "derivation" in row["source_graph"]:
            assets.append(ROOT / row["source_graph"]["derivation"]["full_page_render_path"])
    if not all(path.is_file() for path in assets):
        pytest.skip("exact first-party source artifacts are not local")
    first = audit_signature(contract, ROOT, overlay_dir=tmp_path / "a")
    second = audit_signature(contract, ROOT, overlay_dir=tmp_path / "b")
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
    assert set(first["granularity"].values()) == {8.0, 9.0}
    assert first["gate_results"]["velvia50_invalid_mtf_excluded"] is True
