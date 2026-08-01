from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.physical_measured_mtf_domain_placement import (
    MeasuredMtfPlacementError,
    evaluate_domain_placement,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p5o_measured_mtf_domain_placement_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_contract_rejects_probe_and_source_claim_drift() -> None:
    config = _config()
    validate_contract(config)
    config["probes"]["relative_modulation_amplitudes"][1] = 0.6
    with pytest.raises(MeasuredMtfPlacementError, match="contract drift"):
        validate_contract(config)
    config = _config()
    config["source_boundary"][
        "measurement_protocol_identifies_exact_operator_domain"
    ] = True
    with pytest.raises(MeasuredMtfPlacementError, match="contract drift"):
        validate_contract(config)


def test_domain_placement_is_deterministic_and_numerically_bounded(
    tmp_path: Path,
) -> None:
    first = evaluate_domain_placement(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "first.png"
    )
    second = evaluate_domain_placement(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "second.png"
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert first["decision"] == "placement_unidentified_open_bounded_envelope"
    assert all(first["gate_results"].values())
    assert first["row_partition_exact"] is True
    assert (tmp_path / "first.png").read_bytes() == (
        tmp_path / "second.png"
    ).read_bytes()


def test_parent_hash_drift_fails_closed(tmp_path: Path) -> None:
    config = _config()
    config["parents"]["sensitometry_contract_sha256"] = "0" * 64
    with pytest.raises(MeasuredMtfPlacementError, match="parent hash mismatch"):
        evaluate_domain_placement(
            root=ROOT, config=config, diagnostic_path=tmp_path / "x.png"
        )
