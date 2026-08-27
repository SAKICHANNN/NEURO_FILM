from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p281_hdrplus_monotone_tone_d0_v1.json"


def test_p281_parent_and_roles_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = ROOT / "docs/evidence/P280_HDRPLUS_RESULT_PAIR_GEOMETRY_RESULT.json"
    assert (
        hashlib.sha256(parent.read_bytes()).hexdigest()
        == config["parent_p280_evidence_sha256"]
    )
    assert config["development"]["version"] == "20161014"
    assert config["confirmation"]["version"] == "20171023"


def test_p281_representation_and_gates_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["knot_count"] == 17
    assert config["fit_stride"] == 8
    assert (config["fit_parity"], config["heldout_parity"]) == (0, 1)
    assert config["gates"] == {
        "minimum_identity_rmse_reduction": 0.2,
        "maximum_control_rmse_ratio": 1.05,
        "maximum_new_boundary_components": 0,
    }


def test_p281_evidence_is_exact_when_present() -> None:
    path = ROOT / "docs/evidence/P281_HDRPLUS_MONOTONE_TONE_D0_RESULT.json"
    if not path.is_file():
        return
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] in {
        "PASS_PRIVATE_HDRPLUS_MONOTONE_TONE_D0",
        "FAIL_CLOSED_HDRPLUS_MONOTONE_TONE_D0",
    }
    assert evidence["execution"]["forward_reverse_report_exact"] is True
