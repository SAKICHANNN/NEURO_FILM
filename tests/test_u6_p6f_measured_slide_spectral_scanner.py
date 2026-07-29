from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_slide_spectral_scanner import (
    evaluate_measured_slide_spectral_scanner,
    load_contract,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs" / "u6_p6f_measured_slide_spectral_scanner_v1.json"
)


def test_measured_inventory_and_split_cells_are_exact() -> None:
    report = evaluate_measured_slide_spectral_scanner(
        ROOT, load_contract(CONTRACT)
    )
    assert report["measured_data"]["rows"] == 8640
    assert report["measured_data"]["wavelength_count"] == 31
    assert report["measured_data"]["test_sets"] == [1, 2, 3, 4, 5, 9]
    assert report["measured_data"]["slides"] == [1, 2, 3, 4, 5]
    assert report["split_rows"] == {
        "development": 3456,
        "held-set": 1728,
        "held-slide": 2304,
        "joint-held": 1152,
    }
    assert report["checks"]["strict_transmittance_domain"] is True


def test_matrix_and_all_predictions_remain_bounded() -> None:
    report = evaluate_measured_slide_spectral_scanner(
        ROOT, load_contract(CONTRACT)
    )
    matrix = np.asarray(report["fit"]["bounded_matrix"], dtype=np.float64)
    assert np.all(matrix >= 0.0)
    assert np.all(np.sum(matrix, axis=1) <= 1.0 + 1e-15)
    assert report["checks"]["candidate_bounded"] is True
    for cell in report["output_extrema"].values():
        for interval in cell.values():
            assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_report_is_repeatable_and_oracle_exact(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_measured_slide_spectral_scanner(ROOT, contract)
    second = evaluate_measured_slide_spectral_scanner(ROOT, contract)
    assert first == second
    assert first["checks"]["spectral_oracle"] is True
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(first, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_parent_table_hash_drift_fails_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["pair_table_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pair_table_path hash mismatch"):
        evaluate_measured_slide_spectral_scanner(ROOT, contract)


def test_reopening_p6e_nonlinear_route_fails_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["p6e_nonlinear_rgb_route_closed"] = False
    with pytest.raises(ValueError, match="P6E nonlinear closure"):
        evaluate_measured_slide_spectral_scanner(ROOT, contract)
