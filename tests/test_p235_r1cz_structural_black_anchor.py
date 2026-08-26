from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_p235_r1cz_structural_black_anchor import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p235_r1cz_structural_black_anchor_v1.json"
EVIDENCE = ROOT / "docs/evidence/P235_R1CZ_STRUCTURAL_BLACK_ANCHOR_RESULT.json"


def test_p235_forward_reverse_science_is_exact() -> None:
    forward = run(CONFIG, reverse=False)
    reverse = run(CONFIG, reverse=True)
    assert forward == reverse
    assert forward["status"] == "FAIL_CLOSED_R1CZ_STRUCTURAL_BLACK_ANCHOR"


def test_p235_failure_is_structural_and_before_application_pixels() -> None:
    report = run(CONFIG)
    gates = report["gates"]
    assert gates["p233_and_p234_parent_identities_exact"]
    assert gates["payload_and_envelope_exact"]
    assert not gates["black_preclamp_output_exact_zero_all_channels"]
    assert not gates["black_postclamp_output_exact_zero_all_channels"]
    assert not gates["all_channel_zero_crossings_within_one_micro_nit_of_zero"]
    assert not gates["no_mixed_zero_nonzero_channels_on_fixed_neutral_shadow_probe"]
    assert gates["application_pixels_build_network_and_media_reads_zero"]

    black_raw = np.asarray(
        report["structural_probe"]["black_preclamp_output_nits_float32"]
    )
    black_clamped = np.asarray(
        report["structural_probe"]["black_postclamp_output_nits_float32"]
    )
    assert black_raw[0] > 0.0
    assert np.all(black_raw[1:] < 0.0)
    assert black_clamped[0] > 0.0
    assert np.array_equal(black_clamped[1:], np.zeros(2))


def test_p235_contract_forbids_rescue_and_new_domain() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    no_rescue = set(config["decision"]["no_rescue"])
    assert "black floor or offset" in no_rescue
    assert "knot insertion or endpoint anchoring" in no_rescue
    assert "strength reduction" in no_rescue
    assert "additional application domain" in no_rescue
    assert config["execution"]["application_pixel_reads"] == 0


def test_p235_evidence_binds_formal_result() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = run(CONFIG)
    assert evidence["status"] == report["status"]
    assert evidence["bindings"]["scientific_identity"] == report["scientific_identity"]
    assert (
        evidence["structural_probe"]["black_postclamp_output_nits_float32"]
        == report["structural_probe"]["black_postclamp_output_nits_float32"]
    )
    assert evidence["gates"]["forward_reverse_reports_exact"]
