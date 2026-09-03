from __future__ import annotations

from pathlib import Path

from scripts.audit_u7_21b_desktop_runtime_scope_session_binding import build_report


def test_forward_reverse_reports_are_exact_and_pass(tmp_path: Path) -> None:
    forward = build_report(scratch=tmp_path / "forward", order="forward")
    reverse = build_report(scratch=tmp_path / "reverse", order="reverse")
    assert forward == reverse
    assert forward["status"] == (
        "PASS_PRIVATE_U7_21B_DESKTOP_RUNTIME_SCOPE_SESSION_BINDING"
    )
    assert all(forward["gates"].values())
    assert not (tmp_path / "forward").exists()
    assert not (tmp_path / "reverse").exists()
