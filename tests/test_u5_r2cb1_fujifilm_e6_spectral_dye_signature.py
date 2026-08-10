from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_e6_spectral_dye_signature import (
    CONTRACT_SHA256,
    FujifilmSpectralDyeError,
    _axis_residual,
    _curve_values,
    audit_signature,
    hash_file,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb1_fujifilm_e6_spectral_dye_signature_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB1"


def test_linear_axis_and_curve_are_deterministic() -> None:
    row = {
        "graph_axes": {
            "x_value_pixels": [[400.0, 0.0], [700.0, 300.0]],
            "y_value_pixels": [[1.0, 0.0], [0.0, 100.0]],
        },
        "curves": {"yellow": [[0, 0], [150, 50], [300, 100]]},
    }
    wavelengths, densities = _curve_values(row, "yellow")
    assert np.allclose(wavelengths, [400.0, 550.0, 700.0])
    assert np.allclose(densities, [1.0, 0.5, 0.0])
    assert _axis_residual([[400.0, 0.0], [550.0, 150.0], [700.0, 300.0]], row["graph_axes"]["x_value_pixels"]) == pytest.approx(0.0)


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmSpectralDyeError, match="contract hash drift"):
        load_contract(changed)


def test_formal_source_audit_when_exact_assets_exist(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    trace_path = ROOT / contract["trace"]["path"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assets = [trace_path]
    for row in trace["stocks"].values():
        assets.extend([ROOT / row["source_pdf"]["path"], ROOT / row["source_graph"]["path"]])
        if "derivation" in row["source_graph"]:
            assets.append(ROOT / row["source_graph"]["derivation"]["full_page_render_path"])
    if not all(path.is_file() for path in assets):
        pytest.skip("exact first-party source artifacts are not local")
    first = audit_signature(contract, ROOT, overlay_dir=tmp_path / "a")
    second = audit_signature(contract, ROOT, overlay_dir=tmp_path / "b")
    assert first == second
    assert first["stable_evidence_id"] == second["stable_evidence_id"]
    assert len(first["pairs"]) == 3
