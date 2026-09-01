from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U7_13A_LINUX_PRODUCT_CLI_RUNTIME_RESULT.json"
EVIDENCE_SHA256 = "217c2fff4f54e60b23a90db35c5f279cd753205120505bbed201cee3d540c3a0"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def test_u7_13a_evidence_is_exact_pre_report_failure() -> None:
    payload = EVIDENCE.read_bytes()
    assert _sha256_bytes(payload) == EVIDENCE_SHA256
    report = json.loads(payload)
    assert report["status"] == (
        "FAIL_CLOSED_U7_13A_P_BACKED_EXFAT_VENV_UNAVAILABLE_PRE_REPORT"
    )
    assert report["formal_attempt"]["attempt_count"] == 1
    assert report["formal_attempt"]["stage"] == (
        "fresh-linux-venv-create-before-pip-install-render-or-report"
    )
    assert report["formal_attempt"]["pip_install_count"] == 0
    assert report["formal_attempt"]["render_count"] == 0
    assert report["formal_attempt"]["replay_count"] == 0
    assert report["formal_attempt"]["pixel_decode_count"] == 0
    assert report["formal_attempt"]["report_count"] == 0
    assert report["post_attempt"]["owned_residue_count"] == 0
    assert report["decision"]["exact_leaf_closed"] is True
    assert report["decision"]["allowed_rescue"] is False
    scientific = json.dumps(
        report["scientific"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert _sha256_bytes(scientific) == report["scientific_identity"]


def test_u7_13a_evidence_binds_formal_git_objects_and_source() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    commit = report["source_commit"]
    for path, expected in report["bindings"].items():
        assert _sha256_bytes(_git_blob(commit, path)) == expected
    config = json.loads(
        (ROOT / "configs/u7_13a_linux_product_cli_runtime_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = ROOT / config["source"]["path"]
    assert source.stat().st_size == config["source"]["bytes"]
    assert _sha256_file(source) == report["post_attempt"]["source_sha256"]


def test_u7_13a_evidence_binds_exact_p_backed_wheel_archive() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    config = json.loads(
        (ROOT / "configs/u7_13a_linux_product_cli_runtime_v1.json").read_text(
            encoding="utf-8"
        )
    )
    root = ROOT / config["runtime"]["wheel_root"]
    rows = config["wheels"]
    assert len(rows) == report["acquisition"]["wheel_count"] == 12
    assert sum(int(row[3]) for row in rows) == report["acquisition"]["total_bytes"]
    for row in rows:
        wheel = root / row[2]
        assert wheel.stat().st_size == int(row[3])
        assert _sha256_file(wheel) == row[4]
