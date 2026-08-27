from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/P295_OPENEXR_CARROTS_ACES2065_INTEROPERABILITY_RESULT.json"
)


def test_p295_evidence_closes_before_pixels_on_three_strict_header_failures() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_OPENEXR_CARROTS_ACES2065_INTEROPERABILITY"
    gates = evidence["gates"]
    assert gates["ap0_chromaticities_exact"]
    assert gates["adopted_neutral_exact"]
    assert not gates["float_rgb_only"]
    assert not gates["aces_image_container_flag_exact"]
    assert not gates["color_interop_id_exact"]
    assert gates["zero_pixel_reads"]
    assert gates["zero_working_image_outputs"]


def test_p295_evidence_preserves_source_and_claim_boundaries() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["official_source"]["payload_requests"] == 1
    assert evidence["official_source"]["jpeg_requests"] == 0
    assert evidence["official_source"]["replacement_sample_requests"] == 0
    assert evidence["candidate_count_after_result"] == "2/3"
    assert evidence["consumer_mapping"] is False
    assert (
        evidence["cleanup"]["diagnostic_site_cleanup_status"]
        == "policy_rejected_before_execution"
    )
