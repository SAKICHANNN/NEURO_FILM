from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_10a_product_desktop_input_workflow.py"


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("audit_u7_10a", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u7_10a_formal_report_passes_all_frozen_mechanical_gates() -> None:
    report = _module().build_report("forward")
    assert report["status"] == "PASS_PRIVATE_U7_10A_PRODUCT_DESKTOP_INPUT_WORKFLOW"
    assert all(report["scientific"]["gates"].values())
    assert len(report["scientific"]["cases"]) == 2
    assert sum(len(row["exports"]) for row in report["scientific"]["cases"]) == 6


def test_u7_10a_implementation_commit_is_a_resolvable_full_commit() -> None:
    module = _module()
    assert len(module.IMPLEMENTATION_COMMIT) == 40
    completed = subprocess.run(
        ["git", "cat-file", "-t", module.IMPLEMENTATION_COMMIT],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "commit"


def test_u7_10a_formal_report_is_order_invariant() -> None:
    forward = _module().build_report("forward")
    reverse = _module().build_report("reverse")
    assert forward == reverse
