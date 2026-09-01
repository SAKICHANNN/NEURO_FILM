from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/audit_u7_3m_research_recipe_product_export_exclusion.py"


def test_audit_reports_are_exact_and_all_gates_pass(tmp_path: Path) -> None:
    reports: list[bytes] = []
    for order in ("forward", "reverse"):
        report = tmp_path / f"{order}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--report",
                str(report),
                "--scratch",
                str(tmp_path / "scratch"),
                "--order",
                order,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = report.read_bytes()
        reports.append(payload)
        value = json.loads(payload)
        assert value["status"] == (
            "PASS_PRIVATE_U7_3M_RESEARCH_RECIPE_PRODUCT_EXPORT_EXCLUSION"
        )
        assert all(value["scientific"]["gates"].values())
        assert not (tmp_path / "scratch").exists()
    assert reports[0] == reports[1]

