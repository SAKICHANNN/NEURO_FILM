from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_11a_desktop_single_look_batch.py"


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("audit_u7_11a", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u7_11a_formal_report_passes_all_frozen_gates() -> None:
    report = _module().build_report("forward")
    assert report["status"] == "PASS_PRIVATE_U7_11A_DESKTOP_SINGLE_LOOK_BATCH"
    assert all(report["scientific"]["gates"].values())
    assert report["scientific"]["selection_order_confirmation_exact"] is True
    assert report["scientific"]["targeted_tests"] == {
        "all_pass": True,
        "runs": [
            {"passed_count": 21, "returncode": 0, "stderr_empty": True},
            {"passed_count": 21, "returncode": 0, "stderr_empty": True},
        ],
    }
    case = report["scientific"]["canonical_case"]
    assert case["job_count"] == 2
    assert len(case["children"]) == 2
    assert case["style_id"] == "ektar_100"
    assert case["look_amount"] == 0.625


def test_u7_11a_implementation_commit_is_resolvable() -> None:
    module = _module()
    completed = subprocess.run(
        ["git", "cat-file", "-t", module.IMPLEMENTATION_COMMIT],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "commit"


def test_u7_11a_invalid_order_rejects_before_execution() -> None:
    with pytest.raises(ValueError, match="forward or reverse"):
        _module().build_report("random")
