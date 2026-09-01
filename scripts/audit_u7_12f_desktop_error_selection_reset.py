#!/usr/bin/env python3
"""Committed-head formal audit for U7.12F desktop selection reset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/u7_12f_desktop_error_selection_reset_v1.json"
TEST_PATH = "tests/test_u7_12f_desktop_error_selection_reset.py"
UI_PATH = "src/inference/product_desktop_ui.py"
BOUND_PATHS = (
    CONFIG_PATH,
    "docs/planning/U7_12F_DESKTOP_ERROR_SELECTION_RESET_CONTRACT.md",
    UI_PATH,
    TEST_PATH,
    "scripts/audit_u7_12f_desktop_error_selection_reset.py",
)
CASES = {
    "single-export-error-reset": (
        f"{TEST_PATH}::test_error_invalidates_stale_selection_before_new_preview"
        "[single-export-error]"
    ),
    "batch-error-reset": (
        f"{TEST_PATH}::test_error_invalidates_stale_selection_before_new_preview"
        "[batch-error]"
    ),
    "successful-export-retains-selection": (
        f"{TEST_PATH}::test_success_completion_does_not_clear_valid_selection"
    ),
    "u7-10a-explicit-selection-parent": (
        "tests/test_u7_10a_product_desktop_input_workflow.py::"
        "test_native_export_freezes_explicit_selection_while_busy"
    ),
    "u7-11a-batch-error-parent": (
        "tests/test_u7_11a_desktop_single_look_batch.py::"
        "test_native_batch_success_and_error_clear_worker_state"
    ),
    "u7-12a-close-safety-parent": (
        "tests/test_u7_12a_desktop_foreground_worker_close_safety.py::"
        "test_foreground_success_error_and_concurrency_are_serialized"
    ),
    "u7-12b-format-parent": (
        "tests/test_u7_12b_desktop_single_photo_output_format.py::"
        "test_desktop_format_controls_route_single_photo_and_fix_batch_png16"
    ),
    "u7-12c-preview-parent": (
        "tests/test_u7_12c_desktop_display_native_preview.py::"
        "test_product_workflow_requests_the_shared_visible_box"
    ),
}


def _run(*args: str) -> bytes:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _git_blob(path: str, commit: str = "HEAD") -> bytes:
    return _run("git", "show", f"{commit}:{path}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _windows_checkout_bytes(value: bytes) -> bytes:
    """Return Git LF content in the CRLF form frozen from the Windows checkout."""

    return value.replace(b"\n", b"\r\n")


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _error_handler(payload: bytes) -> str:
    text = payload.decode("utf-8")
    start = text.index("    def _show_error(")
    end = text.index("    def close(", start)
    return text[start:end]


def _case_order(order: str) -> tuple[str, ...]:
    primary = ("single-export-error-reset", "batch-error-reset")
    if order == "reverse":
        primary = tuple(reversed(primary))
    parents = tuple(sorted(set(CASES) - set(primary)))
    return (*primary, *parents)


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
    config = json.loads(_git_blob(CONFIG_PATH))
    parent_commit = str(config["source_parent_commit"])
    parent_blob = _git_blob(UI_PATH, parent_commit)
    parent_handler = _error_handler(parent_blob)
    current_handler = _error_handler(_git_blob(UI_PATH))
    trigger_binding = {
        "parent_commit": parent_commit,
        "parent_ui_git_blob_sha256": _sha256(parent_blob),
        "parent_ui_windows_checkout_sha256": _sha256(
            _windows_checkout_bytes(parent_blob)
        ),
        "expected_parent_ui_windows_checkout_sha256": config["source_identities"][
            "product_desktop_ui_sha256"
        ],
        "parent_clears_style": 'self.style.set("")' in parent_handler,
        "current_clear_count": current_handler.count('self.style.set("")'),
        "clear_precedes_preview_invalidation": (
            current_handler.index('self.style.set("")')
            < current_handler.index("self.preview_ready = False")
        ),
    }

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
        "frozen_parent_defect_bound": (
            trigger_binding["parent_ui_windows_checkout_sha256"]
            == trigger_binding["expected_parent_ui_windows_checkout_sha256"]
            and trigger_binding["parent_clears_style"] is False
        ),
        "current_error_reset_is_exact": (
            trigger_binding["current_clear_count"] == 1
            and trigger_binding["clear_precedes_preview_invalidation"] is True
        ),
        "single_export_error_requires_fresh_selection": results[
            "single-export-error-reset"
        ]["passed"],
        "batch_error_requires_fresh_selection": results["batch-error-reset"]["passed"],
        "successful_export_retains_selection": results[
            "successful-export-retains-selection"
        ]["passed"],
        "focused_parent_behavior_pass": all(
            result["passed"]
            for name, result in results.items()
            if name.startswith("u7-")
        ),
    }
    scientific = {
        "schema": "kmcfm.u7-12f-desktop-error-selection-reset-report.v1",
        "node_id": "U7.12F",
        "source_commit": commit,
        "bindings": bindings,
        "trigger_binding": trigger_binding,
        "case_results": {key: results[key] for key in sorted(results)},
        "gates": gates,
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "private_windows_tk_selection_safety_only": True,
            "renderer_changed": False,
            "product_desktop_core_changed": False,
            "output_or_recipe_changed": False,
            "recovery_or_cache_changed": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
            "public_release": False,
        },
    }
    report = {
        **scientific,
        "scientific_identity": _canonical_sha256(scientific),
        "status": (
            "PASS_PRIVATE_U7_12F_DESKTOP_ERROR_SELECTION_RESET"
            if all(gates.values())
            else "FAIL_CLOSED_U7_12F_DESKTOP_ERROR_SELECTION_RESET"
        ),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = args.report.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
