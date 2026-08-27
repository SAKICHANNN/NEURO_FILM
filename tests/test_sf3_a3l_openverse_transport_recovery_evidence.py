from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "evidence" / "SF3_A3L_OPENVERSE_TRANSPORT_RECOVERY_RESULT.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sf3_a3l_evidence_binds_reports_and_fail_closed_result() -> None:
    value = json.loads(EVIDENCE.read_text())
    assert value["status"] == "FAIL_CLOSED_SOURCE_UNAVAILABLE"
    assert value["formal_runs"]["scientific_payload_exact"] is True
    for key in ("report_a", "report_b"):
        binding = value["formal_runs"][key]
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
    assert value["result"]["http_status_by_request"] == [401] * 6
    assert value["result"]["pixel_reads"] == 0


def test_sf3_a3l_disclosure_forbids_post_result_transport_rescue() -> None:
    value = json.loads(EVIDENCE.read_text())
    assert value["contract"]["post_result_rescue_allowed"] is False
    assert "page_size=5" in value["attempt_disclosure"]["precontract_observation"]
    assert "page_size=50" in value["attempt_disclosure"]["formal_difference"]
    assert "Do not change" in value["attempt_disclosure"]["decision"]
