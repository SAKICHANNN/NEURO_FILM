#!/usr/bin/env python3
"""Committed-head audit for U7.12G desktop batch output formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "tests/test_u7_12g_desktop_batch_output_format.py"
BOUND_PATHS = (
    "configs/u7_12g_desktop_batch_output_format_v1.json",
    "docs/planning/U7_12G_DESKTOP_BATCH_OUTPUT_FORMAT_CONTRACT.md",
    "src/inference/product_desktop.py",
    "src/inference/product_desktop_ui.py",
    TEST_PATH,
    "scripts/audit_u7_12g_desktop_batch_output_format.py",
)
FORMAT_CASES = {
    "png16": f"{TEST_PATH}::test_png16_default_keeps_legacy_receipt_shape",
    "tiff16": (
        f"{TEST_PATH}::test_non_png_batch_matches_direct_cli_and_strict_replay"
        "[tiff16-.tiff-TIFF-16]"
    ),
    "jpeg8": (
        f"{TEST_PATH}::test_non_png_batch_matches_direct_cli_and_strict_replay"
        "[jpeg8-.jpg-JPEG-8]"
    ),
}
FIXED_CASES = {
    "contract": (
        f"{TEST_PATH}::test_contract_freezes_versioned_formats_and_claim_ceiling"
    ),
    "invalid-format": (
        f"{TEST_PATH}::test_unknown_batch_format_rejects_before_stage_or_renderer"
    ),
    "desktop-ui-routing": (
        "tests/test_u7_12b_desktop_single_photo_output_format.py::"
        "test_desktop_format_controls_route_single_photo_and_batch"
    ),
    "png16-recovery-ceiling": (
        "tests/test_u7_12d_desktop_single_look_batch_recovery.py::"
        "test_contract_freezes_restart_and_flat_publication_boundaries"
    ),
    "late-foreign-preserved": (
        "tests/test_u7_11a_desktop_single_look_batch.py::"
        "test_late_foreign_destination_is_preserved"
    ),
    "parent-direct-replay": (
        "tests/test_u7_11a_desktop_single_look_batch.py::"
        "test_two_input_batch_matches_direct_cli_and_strict_replay"
    ),
}


def _run(*args: str) -> bytes:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


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


def _ordered_cases(order: str) -> tuple[tuple[str, str], ...]:
    format_ids = ["png16", "tiff16", "jpeg8"]
    if order == "reverse":
        format_ids.reverse()
    rows = [("contract", FIXED_CASES["contract"])]
    rows.extend((name, FORMAT_CASES[name]) for name in format_ids)
    rows.extend((name, FIXED_CASES[name]) for name in FIXED_CASES if name != "contract")
    return tuple(rows)


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
    for name, node in _ordered_cases(args.order):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", node],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
        }

    formats_pass = all(results[name]["passed"] for name in FORMAT_CASES)
    gates = {
        "committed_source_bound": all(len(value) == 64 for value in bindings.values()),
        "tracked_diff_clean": tracked_diff_clean,
        "contract_and_claim_ceiling_exact": results["contract"]["passed"],
        "png16_legacy_receipt_shape_exact": results["png16"]["passed"],
        "tiff16_and_jpeg8_direct_cli_exact": (
            results["tiff16"]["passed"] and results["jpeg8"]["passed"]
        ),
        "strict_replay_all_formats_exact": formats_pass,
        "recipe_format_and_bit_depth_exact": formats_pass,
        "invalid_format_pre_stage_and_pre_renderer": results["invalid-format"][
            "passed"
        ],
        "desktop_ui_routes_selected_batch_format": results["desktop-ui-routing"][
            "passed"
        ],
        "recovery_remains_png16_only": results["png16-recovery-ceiling"]["passed"],
        "late_foreign_destination_preserved": results["late-foreign-preserved"][
            "passed"
        ],
        "parent_png16_direct_replay_exact": results["parent-direct-replay"]["passed"],
    }
    scientific = {
        "schema": "kmcfm.u7-12g-desktop-batch-output-format-report.v1",
        "node_id": "U7.12G",
        "source_commit": commit,
        "bindings": bindings,
        "case_results": {key: results[key] for key in sorted(results)},
        "gates": gates,
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "private_windows_desktop_batch_export_only": True,
            "formats": ["png16", "tiff16", "jpeg8"],
            "one_format_per_atomic_batch": True,
            "png16_receipt_schema": "kmcfm.desktop-single-look-batch.v1",
            "non_png_receipt_schema": "kmcfm.desktop-single-look-batch.v2",
            "recovery_format": "png16-only",
            "renderer_changed": False,
            "look_math_changed": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
            "hdr_or_wide_gamut_publication": False,
            "public_release": False,
        },
    }
    report = {
        **scientific,
        "scientific_identity": _canonical_sha256(scientific),
        "status": (
            "PASS_PRIVATE_U7_12G_DESKTOP_BATCH_OUTPUT_FORMATS"
            if all(gates.values())
            else "FAIL_CLOSED_U7_12G_DESKTOP_BATCH_OUTPUT_FORMATS"
        ),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = args.report.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
