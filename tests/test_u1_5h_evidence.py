from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U1_5H_STRICT_SDR_AVIF_INGRESS_RESULT.json"


def test_u1_5h_evidence_is_bounded_and_bound_to_immutable_files() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == (
        "PASS_PRIVATE_STRICT_SDR_AVIF_LOOK_APPROXIMATION_INGRESS"
    )
    assert all(evidence["gates"][key] is True for key in (
        "committed_head_clean",
        "official_source_identity_exact",
        "official_source_immutable",
        "official_sdr_container_exact",
        "working_image_finite_linear_srgb",
        "all_p278_reject_before_pillow",
        "product_render_recipe_replay_exact",
        "look_approximation_claim_exact",
        "scratch_residue_zero",
    ))
    assert evidence["formal"]["reports_byte_exact"] is True
    assert evidence["formal"]["p278_negative_count"] == 6
    assert evidence["formal"]["p278_pixel_decoder_calls"] == 0
    assert evidence["official_source"]["container"]["nclx"] == [1, 13, 6, True]
    assert "No arbitrary AVIF/HEIF" in evidence["claim_ceiling"]
    assert "calibrated stock response" in evidence["claim_ceiling"]
    for binding in evidence["bindings"]:
        assert_historical_evidence_binding(ROOT, binding)


def test_u1_5h_evidence_file_has_stable_lf_identity() -> None:
    payload = EVIDENCE.read_bytes()
    assert b"\r\n" not in payload
    assert hashlib.sha256(payload).hexdigest() == (
        "c1de594d3f65c722877524e46c793a9f1020e0bfb64a44bd269c6abecf580bd9"
    )
