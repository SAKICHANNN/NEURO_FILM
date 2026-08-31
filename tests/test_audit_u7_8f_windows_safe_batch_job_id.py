from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/audit_u7_8f_windows_safe_batch_job_id.py"
CONFIG = ROOT / "configs/u7_8f_windows_safe_batch_job_id_v1.json"


def test_frozen_config_matches_implementation_controls() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["node_id"] == "U7.8F"
    assert config["job_id"]["normalization"] == "none-reject-only"
    assert config["job_id"]["trailing_full_stop_forbidden"] is True
    assert set(config["job_id"]["reserved_device_basenames"]) == {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }


def test_formal_runner_is_forward_reverse_exact(tmp_path: Path) -> None:
    forward = tmp_path / "forward.json"
    reverse = tmp_path / "reverse.json"
    for order, output in (("forward", forward), ("reverse", reverse)):
        subprocess.run(
            [
                str(ROOT / ".venv/Scripts/python.exe"),
                str(RUNNER),
                "--order",
                order,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
    assert forward.read_bytes() == reverse.read_bytes()
    report = json.loads(forward.read_text(encoding="utf-8"))
    assert report["decision"] == "PASS_PRIVATE_U7_8F_WINDOWS_SAFE_BATCH_JOB_ID"
    assert all(report["gate_results"].values())
    assert report["metrics"]["forbidden_operation_call_count"] == 0
    assert not (ROOT / "tmp/u7_8f_formal_scratch").exists()
    assert sha256_file(forward) == sha256_file(reverse)
