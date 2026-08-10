from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.kodak_intermediate_role_signature import (
    IntermediateRoleSignatureError,
    audit_intermediate_role_signature,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2by0_kodak_2242_intermediate_role_signature_v1.json"


def test_contract_and_exact_source_audit(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    report = audit_intermediate_role_signature(
        config, ROOT, overlay_path=tmp_path / "overlay.png"
    )
    assert report["passed"] is False
    assert (
        report["decision"] == "close_quantitative_intermediate_role_profile_expansion"
    )
    assert len(report["material_stock_pairs"]) == 2
    assert len(report["stable_evidence_id"]) == 64
    assert (tmp_path / "overlay.png").is_file()


def test_contract_rejects_threshold_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["comparison"]["material_threshold"] = 0.079
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IntermediateRoleSignatureError, match="contract drift"):
        load_contract(path)


def test_source_hash_drift_fails_before_pdf_decode(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    source = ROOT / config["source"]["path"]
    drift = tmp_path / "source.pdf"
    drift.write_bytes(source.read_bytes() + b"drift")
    config["source"]["path"] = (
        drift.relative_to(ROOT).as_posix()
        if drift.is_relative_to(ROOT)
        else "outputs/source_recon/u5_r2by0_kodak_2242/missing.pdf"
    )
    with pytest.raises(IntermediateRoleSignatureError, match="source integrity"):
        audit_intermediate_role_signature(
            config, ROOT, overlay_path=tmp_path / "overlay.png"
        )


def test_trace_point_drift_fails_closed(tmp_path: Path) -> None:
    config = load_contract(CONFIG)
    config["raster_sources"]["mtf"]["samples"]["channels"]["blue"][0] = 200
    with pytest.raises(IntermediateRoleSignatureError, match="left graph ink"):
        audit_intermediate_role_signature(
            config, ROOT, overlay_path=tmp_path / "overlay.png"
        )


def test_conservative_material_gate_is_not_nominal_only(tmp_path: Path) -> None:
    report = audit_intermediate_role_signature(
        load_contract(CONFIG), ROOT, overlay_path=tmp_path / "overlay.png"
    )
    uncertainty_reclassifications = 0
    for domain in ("mtf", "granularity"):
        for row in report["pairwise"][domain].values():
            for channel in row["channels"].values():
                assert (
                    channel["conservative_lower_bound"]
                    <= channel["mean_centered_log_rmse"]
                )
                assert (
                    channel["combined_trace_uncertainty_log_rmse_bound"]
                    > channel["material_trace_uncertainty_log_rmse_bound"]
                )
                if (
                    channel["mean_centered_log_rmse"] >= 0.08
                    and not channel["material"]
                ):
                    uncertainty_reclassifications += 1
    assert uncertainty_reclassifications >= 1
