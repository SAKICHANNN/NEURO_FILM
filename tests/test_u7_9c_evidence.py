from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION_RESULT.json"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_u7_9c_evidence_is_internally_exact() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    report = evidence["formal_report"]
    assert evidence["status"] == report["status"]
    assert evidence["accepted_execution"]["reports_byte_exact"] is True
    assert evidence["accepted_execution"]["forward_report_sha256"] == (
        evidence["accepted_execution"]["reverse_report_sha256"]
    )
    assert report["status"] == "PASS_PRIVATE_U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION"
    assert all(report["gates"].values())
    scientific = {
        key: value
        for key, value in report.items()
        if key not in {"execution", "stable_identity"}
    }
    stable = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert stable == report["stable_identity"]


def test_u7_9c_source_identities_resolve() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    accepted = evidence["accepted_execution"]
    report = evidence["formal_report"]
    assert _git("cat-file", "-t", accepted["commit"]) == "commit"
    assert _git(
        "rev-parse",
        f'{accepted["commit"]}:scripts/audit_u7_9c_product_runtime_startup_isolation.py',
    ) == accepted["audit_git_blob"]
    paths = {
        "config_sha256": ROOT / "configs" / "u7_9c_product_runtime_startup_isolation_v1.json",
        "contract_sha256": ROOT
        / "docs"
        / "planning"
        / "U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION_CONTRACT.md",
        "installer_sha256": ROOT / "scripts" / "install_product_runtime.py",
        "renderer_sha256": ROOT / "scripts" / "render_film.py",
        "requirements_sha256": ROOT / "requirements-product-v2.txt",
    }
    assert {key: _sha256(path) for key, path in paths.items()} == report["source"]


def test_u7_9c_excluded_attempts_are_not_relabelled() -> None:
    evidence = json.loads(EVIDENCE.read_text("utf-8"))
    attempts = evidence["excluded_attempts"]
    assert len(attempts) == 2
    assert all(attempt["accepted"] is False for attempt in attempts)
    assert all(_git("cat-file", "-t", attempt["commit"]) == "commit" for attempt in attempts)
    assert attempts[0]["reports_written"] == 0
    assert attempts[0]["product_render_executions"] == 12
    assert attempts[1]["reports_written"] == 2
    assert attempts[1]["product_render_executions"] == 12
    assert attempts[1]["report_sha256"] == (
        "c5d661a4b8fe6d9f470cea5bc3756202977a8d4e5caf36aaf53101cdb7e4c827"
    )
