from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_scanner_nuisance_operator import (
    evaluate_measured_scanner_nuisance_operator,
    write_report,
)
from src.real_film.scanner_nuisance import build_aligned_patch_bank


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "configs" / "u6_p6g_measured_scanner_nuisance_operator_v1.json"
)


@pytest.fixture(scope="module")
def report() -> dict:
    return evaluate_measured_scanner_nuisance_operator(ROOT, CONTRACT)


def test_patch_bank_reuses_exact_parent_inventory() -> None:
    parent = ROOT / "configs" / "sf2_7a_scanner_nuisance_quantification_v1.json"
    config, bank, alignments = build_aligned_patch_bank(ROOT, parent)
    assert len(alignments) == 20
    assert sorted(bank) == sorted(config["pipeline_roles"])
    assert all(
        bank[role][slide].shape == (264, 3)
        for role in config["pipeline_roles"]
        for slide in config["slide_ids"]
    )


def test_parent_replay_is_exact(report: dict) -> None:
    assert all(report["parent_replay_checks"].values())
    assert report["aggregate"]["identity"]["median_l2"] == pytest.approx(
        0.06054916498813366, abs=1e-15
    )
    assert report["aggregate"][
        "legacy-clipped-full-affine-parent-replay"
    ]["median_l2"] == pytest.approx(0.014462732544002719, abs=1e-15)


def test_safe_candidate_is_structurally_and_numerically_bounded(
    report: dict,
) -> None:
    assert report["checks"]["safe_matrix_structure"]
    assert report["checks"]["safe_output_bounded"]
    assert report["fit_diagnostics"]["safe_matrix_minimum_coefficient"] >= 0.0
    assert (
        report["fit_diagnostics"]["safe_matrix_maximum_row_sum"]
        <= 1.0 + 1e-15
    )
    output = report["aggregate"]["nonnegative-row-sum-bounded-3x3"]["output"]
    assert output["outside_scalar_count"] == 0
    assert 0.0 <= output["minimum"] <= output["maximum"] <= 1.0


def test_legacy_unclipped_diagnostics_preserve_out_of_range_facts(
    report: dict,
) -> None:
    legacy = report["aggregate"][
        "legacy-unclipped-full-affine-diagnostic-only"
    ]["output"]
    assert legacy["outside_scalar_count"] >= 0
    assert np.isfinite(legacy["maximum_boundary_excursion"])
    assert report["checks"]["legacy_unclipped_factual_diagnostics_present"]


def test_report_is_repeatable(report: dict, tmp_path: Path) -> None:
    second = evaluate_measured_scanner_nuisance_operator(ROOT, CONTRACT)
    assert report == second
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    assert write_report(report, path_a) == write_report(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["parents"]["sf2_7a_report_sha256"] = "0" * 64
    drifted = tmp_path / "contract.json"
    drifted.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="sf2_7a_report_path hash mismatch"):
        evaluate_measured_scanner_nuisance_operator(ROOT, drifted)
