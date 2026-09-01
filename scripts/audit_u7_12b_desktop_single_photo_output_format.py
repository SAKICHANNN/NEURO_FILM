#!/usr/bin/env python3
"""Committed-head audit for U7.12B desktop single-photo output formats."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "tests/test_u7_12b_desktop_single_photo_output_format.py"
BOUND_PATHS = (
    "configs/u7_12b_desktop_single_photo_output_format_v1.json",
    "docs/planning/U7_12B_DESKTOP_SINGLE_PHOTO_OUTPUT_FORMAT_CONTRACT.md",
    "src/inference/product_desktop.py",
    "src/inference/product_desktop_ui.py",
    TEST_PATH,
    "scripts/audit_u7_12b_desktop_single_photo_output_format.py",
)
FORMAT_CASES = {
    "png16": (
        f"{TEST_PATH}::test_single_export_matches_direct_cli_and_strict_replay"
        "[png16-.png-PNG-16]"
    ),
    "tiff16": (
        f"{TEST_PATH}::test_single_export_matches_direct_cli_and_strict_replay"
        "[tiff16-.tiff-TIFF-16]"
    ),
    "jpeg8": (
        f"{TEST_PATH}::test_single_export_matches_direct_cli_and_strict_replay"
        "[jpeg8-.jpg-JPEG-8]"
    ),
}
CLOSE_CASES = {
    "close-png16": (
        f"{TEST_PATH}::test_close_waits_safely_for_each_single_photo_format[png16-.png]"
    ),
    "close-tiff16": (
        f"{TEST_PATH}::test_close_waits_safely_for_each_single_photo_format"
        "[tiff16-.tiff]"
    ),
    "close-jpeg8": (
        f"{TEST_PATH}::test_close_waits_safely_for_each_single_photo_format[jpeg8-.jpg]"
    ),
}
FIXED_CASES = {
    "contract-dependency": f"{TEST_PATH}::test_contract_and_dependency_lock_are_exact",
    "invalid-controls": (
        f"{TEST_PATH}::test_invalid_format_or_suffix_rejects_before_command_runner"
    ),
    "png16-batch-compatibility": (
        f"{TEST_PATH}::test_png16_default_command_and_batch_builder_remain_exact"
    ),
    "desktop-ui": (
        f"{TEST_PATH}::test_desktop_format_controls_route_single_photo_and_fix_batch_png16"
    ),
    "parent-u7-12a-close": (
        "tests/test_u7_12a_desktop_foreground_worker_close_safety.py"
    ),
    "parent-u7-11a-batch": (
        "tests/test_u7_11a_desktop_single_look_batch.py::"
        "test_two_input_batch_matches_direct_cli_and_strict_replay"
    ),
    "parent-u7-11a-single": (
        "tests/test_u7_11a_desktop_single_look_batch.py::"
        "test_one_photo_ui_keeps_single_export_route"
    ),
    "parent-u7-10a-direct": (
        "tests/test_u7_10a_product_desktop_input_workflow.py::"
        "test_real_export_matches_same_path_direct_cli_exactly"
    ),
    "parent-u7-10a-existing-pair": (
        "tests/test_u7_10a_product_desktop_input_workflow.py::"
        "test_export_requires_current_preview_and_preserves_existing_pair"
    ),
    "parent-u7-3l-create-only": (
        "tests/test_u7_3l_recipe_replay_create_only_repair.py"
    ),
    "parent-u7-2t-extension": (
        "tests/test_u7_2t_product_output_extension_preflight.py"
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


def _ordered_cases(order: str) -> tuple[tuple[str, str], ...]:
    format_ids = ["png16", "tiff16", "jpeg8"]
    if order == "reverse":
        format_ids.reverse()
    rows: list[tuple[str, str]] = [
        ("contract-dependency", FIXED_CASES["contract-dependency"])
    ]
    rows.extend((name, FORMAT_CASES[name]) for name in format_ids)
    rows.extend((f"close-{name}", CLOSE_CASES[f"close-{name}"]) for name in format_ids)
    rows.extend(
        (name, FIXED_CASES[name])
        for name in FIXED_CASES
        if name != "contract-dependency"
    )
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

    all_formats = all(results[name]["passed"] for name in FORMAT_CASES)
    all_close = all(results[f"close-{name}"]["passed"] for name in FORMAT_CASES)
    all_parents = all(
        results[name]["passed"] for name in results if name.startswith("parent-")
    )
    gates = {
        "committed_source_bound": all(len(value) == 64 for value in bindings.values()),
        "tracked_diff_clean": tracked_diff_clean,
        "u7_12a_dependency_exact": results["contract-dependency"]["passed"],
        "direct_cli_image_and_recipe_exact_all_formats": all_formats,
        "strict_replay_exact_all_formats": all_formats,
        "recipe_semantics_exact_all_formats": all_formats,
        "invalid_format_and_suffix_pre_render_reject": results["invalid-controls"][
            "passed"
        ],
        "png16_default_and_batch_command_exact": results["png16-batch-compatibility"][
            "passed"
        ],
        "desktop_format_state_and_routing_exact": results["desktop-ui"]["passed"],
        "foreground_close_safety_all_formats": all_close,
        "create_only_and_late_foreign_preserved": (
            results["parent-u7-10a-existing-pair"]["passed"]
            and results["parent-u7-3l-create-only"]["passed"]
        ),
        "owned_residue_zero_all_formats": all_formats and all_close,
        "behavioral_parent_chain_passed": all_parents,
    }
    scientific = {
        "schema": "kmcfm.u7-12b-desktop-single-photo-output-format-report.v1",
        "node_id": "U7.12B",
        "source_commit": commit,
        "bindings": bindings,
        "case_results": {key: results[key] for key in sorted(results)},
        "gates": gates,
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "private_native_single_photo_sdr_export_only": True,
            "formats": ["png16", "tiff16", "jpeg8"],
            "batch_format": "png16",
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
            "PASS_PRIVATE_U7_12B_DESKTOP_SINGLE_PHOTO_OUTPUT_FORMATS"
            if all(gates.values())
            else "FAIL_CLOSED_U7_12B_DESKTOP_SINGLE_PHOTO_OUTPUT_FORMATS"
        ),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = args.report.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
