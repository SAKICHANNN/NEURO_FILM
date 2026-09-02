from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_u7_20g_exif_oriented_jpeg_preview import build_report


def test_audit_rejects_noncanonical_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="order"):
        build_report(scratch=tmp_path / "scratch", order="random")


def test_audit_requires_absent_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(FileExistsError, match="scratch"):
        build_report(scratch=scratch, order="forward")
