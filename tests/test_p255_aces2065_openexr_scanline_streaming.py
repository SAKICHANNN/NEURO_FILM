from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/eval/p255_aces2065_openexr_scanline_writer.cpp"
CONFIG = ROOT / "configs/p255_aces2065_openexr_scanline_streaming_v1.json"
EVIDENCE = ROOT / "docs/evidence/P255_ACES2065_OPENEXR_SCANLINE_STREAMING_RESULT.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_config_binds_parent_sources_and_execution_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    for prefix in (
        "contract",
        "native_source",
        "runner",
        "p248_config",
        "p248_evidence",
        "p249_config",
        "p249_evidence",
    ):
        path = ROOT / bindings[f"{prefix}_path"]
        assert path.stat().st_size == bindings[f"{prefix}_bytes"]
        assert _sha256(path) == bindings[f"{prefix}_sha256"]
    assert _sha256(ROOT / bindings["p248_native_source_path"]) == bindings[
        "p248_native_source_sha256"
    ]
    assert config["transform"]["ap1_to_ap0_matrix"] == [
        [0.6954522414, 0.1406786965, 0.1638690622],
        [0.0447945634, 0.8596711185, 0.0955343182],
        [-0.0055258826, 0.0040252103, 1.0015006723],
    ]
    assert config["metadata"]["aces_image_container_flag"] == 1
    assert config["metadata"]["color_interop_id"] == "lin_ap0_scene"
    assert config["probe"]["logical_input_bytes"] == 288_000_000
    assert config["gates"]["maximum_worker_process_tree_rss_bytes"] == 2**31
    assert config["gates"]["maximum_worker_wall_seconds"] == 120.0
    assert config["gates"]["require_decoded_pixel_maximum_absolute_error"] == 0.0
    assert config["status"] == "FROZEN_BEFORE_NATIVE_BUILD_OR_PIXEL_EXECUTION"


def test_native_source_binds_ap0_transform_and_identity() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for token in (
        "0.6954522414",
        "0.1406786965",
        "0.1638690622",
        "0.7347F",
        "-0.077F",
        '"acesImageContainerFlag"',
        '"colorInteropID"',
        '"lin_ap0_scene"',
        "ZIP_COMPRESSION",
        "P255 row block must equal 16",
        "MOVEFILE_WRITE_THROUGH",
    ):
        assert token in text


def test_p248_parent_source_remains_separate() -> None:
    parent = ROOT / "src/eval/p248_acescg_openexr_scanline_writer.cpp"
    text = parent.read_text(encoding="utf-8")
    assert "acesImageContainerFlag" not in text
    assert "colorInteropID" not in text
    assert "P248 row block must equal 16" in text


def test_formal_evidence_binds_reports_and_preserves_claim_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_ACES2065_OPENEXR_24MP_SCANLINE_STREAMING"
    assert all(evidence["gates"].values())
    identities = {row["stable_identity"] for row in evidence["formal_reports"]}
    assert identities == {
        "0fe90cd9a5a2218300a61f0e1ceb166846c69609a7a8a170e43c99616322dbb2"
    }
    for report in evidence["formal_reports"]:
        path = ROOT / report["path"]
        assert path.stat().st_size == report["bytes"]
        assert _sha256(path) == report["sha256"]
    result = evidence["result"]
    assert result["two_complete_controllers_stable_exact"]
    assert result["decoded_maximum_absolute_error"] == 0.0
    assert result["small_probe_native_vs_p249_maximum_absolute_error"] == 0.0
    assert result["maximum_worker_peak_process_tree_rss_bytes"] <= 2**31
    assert result["maximum_worker_wall_seconds"] <= 120.0
    assert "No natural-image quality" in evidence["claim_ceiling"]
    assert "Candidate 3 remains closed at 2/3" in evidence["claim_ceiling"]
