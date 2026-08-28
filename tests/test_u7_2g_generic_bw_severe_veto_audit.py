from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/audit_u7_2g_generic_bw_severe_veto.py"


def test_audit_reports_are_scientifically_exact(tmp_path: Path) -> None:
    reports = []
    for order in ("forward", "reverse"):
        output = tmp_path / f"{order}.json"
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "--order", order, "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        reports.append(json.loads(output.read_text(encoding="utf-8")))
    assert reports[0] == reports[1]
    assert reports[0]["status"] == "PASS"
    assert all(reports[0]["gates"].values())
