from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_2K_PRODUCT_LOOK_CLI_DISCOVERY_RESULT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_2k_evidence_binds_cli_and_exact_catalog_replay() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    for binding in evidence["bindings"].values():
        assert _sha(ROOT / binding["path"]) == binding["sha256"]

    outputs: list[bytes] = []
    for _ in range(evidence["result"]["fresh_process_replay_count"]):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/render_film.py"), "--list-product-looks"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
    assert len(outputs[0]) == evidence["result"]["stdout_bytes_each"]
    assert hashlib.sha256(outputs[0]).hexdigest() == evidence["result"]["stdout_sha256"]
    payload = json.loads(outputs[0])
    assert [row["look_id"] for row in payload["looks"]] == evidence["result"][
        "look_order"
    ]
