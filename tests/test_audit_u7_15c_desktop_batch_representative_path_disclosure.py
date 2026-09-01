from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts/audit_u7_15c_desktop_batch_representative_path_disclosure.py"


def test_forward_reverse_audit_is_exact(tmp_path: Path) -> None:
    reports: list[bytes] = []
    for order in ("forward", "reverse"):
        report = tmp_path / f"{order}.json"
        subprocess.run(
            (
                sys.executable,
                str(AUDIT),
                "--order",
                order,
                "--report",
                str(report),
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        reports.append(report.read_bytes())
    assert reports[0] == reports[1]
    payload = json.loads(reports[0])
    assert payload["status"] == (
        "PASS_PRIVATE_U7_15C_DESKTOP_BATCH_REPRESENTATIVE_PATH_DISCLOSURE"
    )
    assert all(payload["scientific"]["gates"].values())
    assert payload["scientific"]["result"]["ui"]["values"] == [
        "Automatic · canonical first",
        "001 · same-name.png",
        "002 · same-name.png",
    ]
