#!/usr/bin/env python3
"""Committed-head audit for U7.12A desktop foreground close safety."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "tests/test_u7_12a_desktop_foreground_worker_close_safety.py"
BOUND_PATHS = (
    "configs/u7_12a_desktop_foreground_worker_close_safety_v1.json",
    "docs/planning/U7_12A_DESKTOP_FOREGROUND_WORKER_CLOSE_SAFETY_CONTRACT.md",
    "src/inference/product_desktop_ui.py",
    TEST_PATH,
    "scripts/audit_u7_12a_desktop_foreground_worker_close_safety.py",
)
CASES = {
    "preview-close": (
        f"{TEST_PATH}::test_close_waits_asynchronously_for_foreground_operation[preview]"
    ),
    "single-export-close": (
        f"{TEST_PATH}::test_close_waits_asynchronously_for_foreground_operation[single-export]"
    ),
    "success-error-concurrency": (
        f"{TEST_PATH}::test_foreground_success_error_and_concurrency_are_serialized"
    ),
}


def _run(*args: str) -> bytes:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_blob(path: str) -> bytes:
    return _run("git", "show", f"HEAD:{path}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256(encoded)


def _case_order(order: str) -> tuple[str, ...]:
    ordered = ("preview-close", "single-export-close")
    if order == "reverse":
        ordered = tuple(reversed(ordered))
    return (*ordered, "success-error-concurrency")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    commit = _run("git", "rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff_clean = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    bindings = {path: _sha256(_git_blob(path)) for path in BOUND_PATHS}
    results: dict[str, dict[str, Any]] = {}
    for name in _case_order(args.order):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", CASES[name]],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
        }

    gates = {
        "committed_source_bound": all(len(value) == 64 for value in bindings.values()),
        "tracked_diff_clean": tracked_diff_clean,
        "preview_close_safe": results["preview-close"]["passed"],
        "single_export_close_safe": results["single-export-close"]["passed"],
        "success_error_concurrency_serialized": results[
            "success-error-concurrency"
        ]["passed"],
    }
    scientific = {
        "schema": "kmcfm.u7-12a-desktop-foreground-worker-close-safety-report.v1",
        "node_id": "U7.12A",
        "source_commit": commit,
        "bindings": bindings,
        "case_results": {key: results[key] for key in sorted(results)},
        "gates": gates,
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "private_windows_tk_close_safety_only": True,
            "renderer_changed": False,
            "batch_changed": False,
            "installer_changed": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "public_release": False,
        },
    }
    report = {
        **scientific,
        "scientific_identity": _canonical_sha256(scientific),
        "status": (
            "PASS_PRIVATE_U7_12A_DESKTOP_FOREGROUND_CLOSE_SAFETY"
            if all(gates.values())
            else "FAIL_CLOSED_U7_12A_DESKTOP_FOREGROUND_CLOSE_SAFETY"
        ),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = args.report.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

