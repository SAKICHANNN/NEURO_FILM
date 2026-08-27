from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_p303_eyefultower_dci_p3_sidecar_feasibility import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p303_eyefultower_dci_p3_sidecar_feasibility_v1.json"
EVIDENCE = (
    ROOT / "docs/evidence/P303_EYEFULTOWER_DCI_P3_SIDECAR_FEASIBILITY_RESULT.json"
)


def test_p303_freezes_exact_p293_source_without_pixel_authorization() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["source"]["sha256"] == (
        "c7b65fe69568b4d04bbf162381fd6d74c9b9682057a0f9dd8b931ec6b038f174"
    )
    assert config["boundaries"]["exr_header_reads"] == 0
    assert config["boundaries"]["exr_pixel_reads"] == 0
    assert config["boundaries"]["sidecar_publications"] == 0


def test_p303_contract_does_not_silently_choose_a_white_point() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    variants = config["frozen_color_facts"]["authoritative_white_variants"]
    assert variants == {"P3D65": [0.3127, 0.329], "P3DCI": [0.314, 0.351]}
    assert config["dataset_authority"]["white_point_declared"] is False
    assert "Do not select P3D65, P3DCI" in config["stop_rule"]


def test_p303_formal_logic_fails_before_sidecar_or_pixels() -> None:
    forward = execute(CONFIG)
    reverse = execute(CONFIG, reverse=True)
    assert forward == reverse
    assert forward["authoritative_variants"] == ["P3D65", "P3DCI"]
    assert forward["decision"] == (
        "FAIL_CLOSED_AMBIGUOUS_DCI_P3_WHITE_POINT_BEFORE_SIDECAR"
    )
    assert forward["source_header_reads"] == 0
    assert forward["source_pixel_reads"] == 0
    assert forward["sidecar_publications"] == 0


def test_p303_evidence_is_frozen() -> None:
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == (
        "a15ace7a5109d446886f2098075bbb48be68fcfde02f3b80b07ece0f6512d595"
    )
