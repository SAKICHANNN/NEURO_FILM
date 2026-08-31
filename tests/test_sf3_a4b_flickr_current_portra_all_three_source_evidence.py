from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def test_sf3_a4b_evidence_binds_frozen_objects_and_formal_result() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (
            root
            / "docs/evidence/SF3_A4B_FLICKR_CURRENT_PORTRA_ALL_THREE_SOURCE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["decision"] == (
        "FAIL_CLOSED_SPRKLG_THREE_STOCK_ALTERATION_SUPPORT_AND_GROUP_GAP"
    )
    assert evidence["formal_replay"] == {
        "processes": 2,
        "orders": ["forward", "reverse"],
        "bytes_each": 42643,
        "byte_exact": True,
        "sha256": "33fca4b648bf44fd5b19d380f298c071972d0ac98736f6a0f772be941a67a746",
        "stable_evidence_id": "37f5cede5928ecb65a203ae3a7edcbbb015992cc77dedce132912484f19c927b",
    }
    implementation = evidence["commits"]["implementation"]
    assert _git(root, "cat-file", "-t", implementation) == "commit"
    expected_blobs = {
        "config_blob": "configs/sf3_a4b_flickr_current_portra_all_three_source_v1.json",
        "contract_blob": "docs/planning/SF3_A4B_FLICKR_CURRENT_PORTRA_ALL_THREE_SOURCE_CONTRACT.md",
        "core_blob": "src/real_film/flickr_current_portra_all_three_source.py",
        "runner_blob": "scripts/audit_sf3_a4b_flickr_current_portra_all_three_source.py",
        "test_blob": "tests/test_sf3_a4b_flickr_current_portra_all_three_source.py",
    }
    for key, path in expected_blobs.items():
        assert (
            _git(root, "rev-parse", f"{implementation}:{path}")
            == evidence["source_objects"][key]
        )
    assert evidence["source"]["target_counts"] == {
        "fujifilm_velvia_50": 1,
        "kodak_ektar_100": 47,
        "kodak_portra_400_current": 29,
    }
    assert evidence["source"]["unaltered_target_counts"]["fujifilm_velvia_50"] == 0
    assert evidence["connectivity"]["augmented_qualifying_author_count"] == 5
    assert evidence["connectivity"]["gate_passed"] is True
    assert all(evidence["source_gates"].values())
    assert all(evidence["connectivity_gates"].values())
    assert not any(evidence["pixel_admission_gates"].values())
    for key in (
        "image_url_requests",
        "image_head_requests",
        "image_range_requests",
        "image_body_requests",
        "exif_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert evidence["operation_counts"][key] == 0
