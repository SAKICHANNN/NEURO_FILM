from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_15b_desktop_visible_strength_quantization.py"


def test_forward_reverse_audit_is_exact(tmp_path: Path) -> None:
    reports = []
    for order in ("forward", "reverse"):
        report = tmp_path / f"{order}.json"
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--order", order, "--report", str(report)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        reports.append(report.read_bytes())
    assert reports[0] == reports[1]
    payload = json.loads(reports[0])
    assert payload["status"] == (
        "PASS_PRIVATE_U7_15B_DESKTOP_VISIBLE_STRENGTH_QUANTIZATION"
    )
    assert all(payload["gates"].values())
    assert payload["ui_probe"]["parent"] == {
        "visible": "65%",
        "variable": 0.654321,
        "workflow_amount": 0.654321,
    }
    assert payload["ui_probe"]["current_render_entry"] == {
        "visible": "65%",
        "variable": 0.65,
        "workflow_amount": 0.65,
    }
