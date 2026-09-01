from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads(
    (
        ROOT / "docs/evidence/U4_2A_CURRENT_THREE_LOOK_STYLE_APPEAL_RESULT.json"
    ).read_text(encoding="utf-8")
)


def _git_blob_sha256(path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_u4_2a_negative_result_is_exact_and_claim_bounded() -> None:
    assert EVIDENCE["status"] == (
        "FAIL_CLOSED_CURRENT_THREE_LOOK_AUTONOMOUS_STYLE_APPEAL_VALUE"
    )
    formal = EVIDENCE["formal"]
    assert formal["reports_byte_exact"] is True
    assert (
        formal["forward_sha256"]
        == formal["reverse_sha256"]
        == ("46d3a0b2396c30367c24173772c47e82768aa5e79e63ba6e98bbb55dacb33619")
    )
    assert formal["reviewed_observations"] == 108
    assert formal["missing_observations"] == 0
    assert formal["passing_named_looks"] == 0
    assert formal["portfolio_pass"] is False
    assert EVIDENCE["execution"]["production_default_changed"] is False
    assert any(
        "Look Approximation" in boundary
        for boundary in EVIDENCE["preserved_boundaries"]
    )


def test_every_named_look_fails_only_the_frozen_style_advantage() -> None:
    assert set(EVIDENCE["look_results"]) == {
        "velvia_50",
        "portra_400",
        "ektar_100",
    }
    for row in EVIDENCE["look_results"].values():
        assert row["pass"] is False
        assert row["severe_yes"] == 0
        assert row["severe_uncertain"] == 0
        assert row["paired_appeal_delta_median_vs_identity"] >= 0.0
        assert row["failed_checks"] == ["style_delta_median", "style_delta_count"]


def test_observation_blob_was_committed_before_unblinding() -> None:
    path = "configs/u4_2a_current_three_look_style_appeal_observations_v1.json"
    assert (
        _git_blob_sha256(path) == EVIDENCE["source_lock"]["observation_git_blob_sha256"]
    )
    provenance = EVIDENCE["blind_provenance"]
    assert provenance["accepted_public_review_sheet_reads"] == 27
    assert provenance["private_mapping_reads_before_freeze"] == 0
    assert provenance["mapping_reconstructed_before_freeze"] is False
    assert provenance["observations_committed_before_formal_unblinding"] is True


def test_output_identity_claim_excludes_path_specific_recipes() -> None:
    scope = EVIDENCE["artifact_identity_scope"]
    assert scope["identity_pixels_exact_across_builds"] is True
    assert scope["review_media_exact_across_builds"] is True
    assert scope["formal_reports_exact_across_processes"] is True
    assert scope["entire_output_tree_or_recipe_bytes_exact_claimed"] is False
