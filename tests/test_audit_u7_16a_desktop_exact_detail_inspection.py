from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.audit_u7_16a_desktop_exact_detail_inspection import (
    CASES,
    UI_PATH,
    _case_order,
    _git_oid,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_16a_desktop_exact_detail_inspection.py"


def test_formal_case_order_is_reversible_and_complete() -> None:
    forward = _case_order("forward")
    reverse = _case_order("reverse")
    assert reverse == tuple(reversed(forward))
    assert set(forward) == set(CASES)
    assert len(forward) == 10


def test_formal_controller_has_a_bounded_cli() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--order {forward,reverse}" in completed.stdout
    assert "--report REPORT" in completed.stdout


def test_parent_ui_identity_uses_git_object_identity() -> None:
    assert len(_git_oid(UI_PATH, "e76cd5501a267e8ec63c9404b1d5d96ed60038c1")) == 40
