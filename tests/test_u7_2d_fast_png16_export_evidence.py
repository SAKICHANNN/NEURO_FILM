from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2D_FAST_PNG16_EXPORT_RESULT.json"


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_u7_2d_evidence_binds_exact_formal_replay() -> None:
    evidence = _evidence()
    formal = evidence["formal_execution"]
    outputs = evidence["exact_outputs"]

    assert evidence["decision"] == "PASS_OPT_IN_FAST_PNG16_EXPORT"
    assert formal["stable_identity"] == (
        "41ea55cec8beb64b9bfa8e3976510158876b6b5d1f1ea503d28ef98701ab6c2f"
    )
    assert formal["forward_decision"] == formal["reverse_decision"] == "PASS"
    assert formal["forward_wall_ratio"] <= 0.85
    assert formal["reverse_wall_ratio"] <= 0.85
    assert outputs["maximum_candidate_to_baseline_file_size_ratio"] <= 1.2

    for gate in (
        "all_three_decoded_rgb16_samples_exact",
        "all_three_srgb_icc_profiles_exact",
        "candidate_repeat_output_bytes_exact",
        "default_encoder_output_bytes_unchanged",
        "recipe_records_exact_png_compression",
        "scratch_empty",
    ):
        assert outputs[gate] is True


def test_u7_2d_evidence_keeps_claim_ceiling_narrow() -> None:
    evidence = _evidence()
    ceiling = evidence["claim_ceiling"]

    assert "Look Approximation" in ceiling
    assert "not stock evidence" in ceiling
    assert "default is unchanged" in ceiling
