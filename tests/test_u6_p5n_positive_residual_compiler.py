from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_positive_mtf_residual import (
    PositiveMtfResidualError,
    compile_and_evaluate_residual,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p5n_positive_residual_compiler_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_rejects_overlap_and_radius_drift() -> None:
    config = _config()
    validate_contract(config)
    config["frequency_split"]["confirmation_frequencies_cycles_per_mm"][0] = 25
    with pytest.raises(PositiveMtfResidualError, match="frequency split"):
        validate_contract(config)
    config = _config()
    config["candidate_family"]["candidate_radii_px"].append(9)
    with pytest.raises(PositiveMtfResidualError, match="contract drift"):
        validate_contract(config)


def test_compiler_closes_on_two_dimensional_mismatch(tmp_path: Path) -> None:
    first_bundle, first = compile_and_evaluate_residual(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "first.png"
    )
    second_bundle, second = compile_and_evaluate_residual(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "second.png"
    )
    assert first_bundle == second_bundle
    assert first == second
    assert first["automatic_pass"] is False
    assert first["decision"] == "close_residual_use_measured_total_replacement"
    assert first["gate_results"]["development_fit"] is True
    assert first["gate_results"]["confirmation_fit"] is True
    assert first["gate_results"]["two_dimensional_chart_max"] is False
    assert first["gate_results"]["two_dimensional_chart_mean"] is False
    assert all(first["row_partition_exact"].values())
    assert [
        first["channel_rows"][name]["selected_radius_px"]
        for name in ("red", "green", "blue")
    ] == [2, 7, 1]
    assert (tmp_path / "first.png").read_bytes() == (
        tmp_path / "second.png"
    ).read_bytes()


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    config = _config()
    config["parents"]["p5m_decision_sha256"] = "0" * 64
    with pytest.raises(PositiveMtfResidualError, match="parent hash mismatch"):
        compile_and_evaluate_residual(
            root=ROOT, config=config, diagnostic_path=tmp_path / "x.png"
        )
