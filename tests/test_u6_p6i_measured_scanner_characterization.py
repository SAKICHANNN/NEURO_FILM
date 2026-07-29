from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_scanner_characterization import (
    _fit_positive_matrix,
    _load_target_table,
    _select_common_power,
    _slide_member,
    evaluate_measured_scanner_characterization,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs" / "u6_p6i_measured_scanner_characterization_v1.json"
)


def test_slide_member_parses_all_published_filename_families() -> None:
    names = [
        "x/testscan_1_3scsaled.tif",
        "x/testscan2-4scaled.tif",
        "x/Test9-5scaled.tif",
    ]
    assert _slide_member(names, 1, 3).endswith("1_3scsaled.tif")
    assert _slide_member(names, 2, 4).endswith("2-4scaled.tif")
    assert _slide_member(names, 9, 5).endswith("9-5scaled.tif")


def test_target_table_selects_exact_colour_grid() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    table = _load_target_table(
        ROOT / contract["parents"]["pair_table_path"], [1, 2, 9]
    )
    assert sorted(table) == [1, 2, 9]
    for target_set in table:
        for slide in range(1, 6):
            assert table[target_set][slide]["source_rgb"].shape == (264, 3)
            assert table[target_set][slide]["xyz"].shape == (264, 3)
            assert table[target_set][slide]["lab"].shape == (264, 3)
            assert table[target_set][slide]["sample_ids"][0] == "A1"
            assert table[target_set][slide]["sample_ids"][-1] == "L22"


def test_positive_matrix_and_power_selection_recover_synthetic_transfer() -> None:
    rng = np.random.default_rng(17)
    source = rng.uniform(0.05, 0.9, size=(256, 3))
    expected_power = 1.5
    expected_matrix = np.asarray(
        [[0.7, 0.1, 0.0], [0.0, 0.8, 0.1], [0.1, 0.0, 0.6]]
    )
    target = np.power(source, expected_power) @ expected_matrix.T
    matrix = _fit_positive_matrix(source, target, [0.0, 4.0])
    assert np.all(matrix >= 0.0)
    power, selected, loss = _select_common_power(
        source,
        target,
        [1.0, 1.5, 2.0],
        [0.0, 4.0],
    )
    assert power == expected_power
    assert loss < 1e-6
    np.testing.assert_allclose(selected, expected_matrix, atol=1e-5)


@pytest.fixture(scope="module")
def report() -> dict:
    return evaluate_measured_scanner_characterization(ROOT, CONTRACT)


def test_real_support_and_reference_consistency_are_exact(report: dict) -> None:
    assert report["support"]["pipelines"] == 13
    assert report["support"]["alignment_cells"] == 65
    assert report["support"]["target_sets"] == [1, 2, 9]
    assert report["checks"]["source_patch_index_exact"]
    assert report["checks"]["published_xyz_lab_consistency"]
    assert report["alignment_summary"]["retry_cells"] >= 1


def test_candidates_are_explicit_nonnegative_and_unclipped(report: dict) -> None:
    assert report["checks"]["matrix_coefficient_domain"]
    assert report["fit_diagnostics"]["minimum_matrix_coefficient"] >= 0.0
    for name, interval in report["output_extrema"].items():
        if name != "identity-device-rgb-as-xyz-control":
            assert interval["minimum"] >= 0.0


def test_report_is_repeatable(report: dict, tmp_path: Path) -> None:
    second = evaluate_measured_scanner_characterization(ROOT, CONTRACT)
    assert report == second
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(report, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["p6h_report_sha256"] = "0" * 64
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="p6h_report_path hash mismatch"):
        evaluate_measured_scanner_characterization(ROOT, drifted)
