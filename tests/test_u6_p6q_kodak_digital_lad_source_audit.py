from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.run_u6_p6q_kodak_digital_lad_source_audit import build_report
from src.eval.kodak_digital_lad_source_audit import (
    KodakDigitalLadAuditError,
    audit_zip_inventory,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6q_kodak_digital_lad_source_audit_v1.json"


def test_formal_source_audit_stops_before_archive_decompression() -> None:
    report = build_report(CONFIG)
    assert report["decision"] == (
        "FAIL_CLOSED_ARCHIVE_MEMBER_EXCEEDS_FROZEN_LIMIT"
    )
    assert not report["all_gates_passed"]
    assert not report["gates"]["all_archive_members_within_frozen_byte_limit"]
    for name in ("dpx_archive", "cineon_archive"):
        archive = report["source_access"][name]
        assert archive["member_count"] == 2
        assert archive["largest_member_bytes"] == 51_019_264
        assert archive["maximum_member_bytes"] == 50_000_000
        assert not archive["largest_member_within_limit"]
    assert report["execution"]["archive_member_decompress_count"] == 0
    assert report["execution"]["pixel_decode_count"] == 0
    assert report["execution"]["operator_fit_count"] == 0


def test_source_archive_hash_drift_fails_closed(tmp_path: Path) -> None:
    source = ROOT / "data/physical_scanner/kodak_digital_lad_v1/Digital-LAD-DPX.zip"
    bad = tmp_path / "bad.zip"
    bad.write_bytes(source.read_bytes() + b"drift")
    with pytest.raises(KodakDigitalLadAuditError, match="byte count drift"):
        audit_zip_inventory(
            bad,
            expected_bytes=source.stat().st_size,
            expected_sha256=sha256_file(source),
            maximum_member_bytes=50_000_000,
            maximum_expansion_ratio=4.0,
        )


def test_unsafe_archive_path_is_reported_without_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("../escape.dpx", b"payload")
    report = audit_zip_inventory(
        archive_path,
        expected_bytes=archive_path.stat().st_size,
        expected_sha256=sha256_file(archive_path),
        maximum_member_bytes=50_000_000,
        maximum_expansion_ratio=100.0,
    )
    assert not report["all_member_paths_safe"]
    assert not report["members"][0]["safe_relative_non_symlink_path"]


def test_config_preserves_documentation_only_failure_branch() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["source_gate"]["no_archive_member_above_bytes"] == 50_000_000
    assert "documentation-only" in config["branch_rule"]["format_or_pixel_mismatch"]
    assert any("display-sRGB" in item for item in config["forbidden"])
