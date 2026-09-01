from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.audit_u7_17a_desktop_bounded_finishing_effects import (
    CASES,
    _case_order,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_17a_desktop_bounded_finishing_effects.py"


def test_formal_case_order_is_reversible_and_complete() -> None:
    forward = _case_order("forward")
    reverse = _case_order("reverse")
    assert reverse == tuple(reversed(forward))
    assert set(forward) == set(CASES)
    assert len(forward) == 13


def test_formal_controller_has_bounded_cli() -> None:
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


def test_formal_controller_runs_parent_files_in_isolated_processes() -> None:
    source = SCRIPT.read_text("utf-8")
    assert '[sys.executable, "-m", "pytest", "-q", CASES[name]]' in source
    assert 'line.startswith(("FAILED ", "ERROR "))' in source


def test_formal_controller_binds_zero_parent_and_nonzero_effect_semantics() -> None:
    source = SCRIPT.read_text("utf-8")
    assert '"zero_effect_parent_output_exact"' in source
    assert '"maximum_effect_recipe_exact"' in source
    assert '"maximum_effect_detail_and_final_exact"' in source
    assert '"maximum_effect_no_code_boundary"' in source


def test_formal_report_excludes_execution_order() -> None:
    source = SCRIPT.read_text("utf-8")
    assert '"order": args.order' not in source
    assert '"case_results": {key: cases[key] for key in sorted(cases)}' in source
