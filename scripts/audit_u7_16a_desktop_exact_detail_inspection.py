#!/usr/bin/env python3
"""Committed-head formal audit for U7.16A exact desktop detail inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_desktop import (
    ProductDesktopWorkflow,
    sha256_file,
)
from src.inference.product_detail_inspection import (
    render_exact_export_detail,
)

CONFIG_PATH = "configs/u7_16a_desktop_exact_detail_inspection_v1.json"
CONTRACT_PATH = "docs/planning/U7_16A_DESKTOP_EXACT_DETAIL_INSPECTION_CONTRACT.md"
CORE_PATH = "src/inference/product_desktop.py"
UI_PATH = "src/inference/product_desktop_ui.py"
HELPER_PATH = "src/inference/product_detail_inspection.py"
FOCUSED_TEST_PATH = "tests/test_u7_16a_desktop_exact_detail_inspection.py"
SCRIPT_PATH = "scripts/audit_u7_16a_desktop_exact_detail_inspection.py"
AUDIT_TEST_PATH = "tests/test_audit_u7_16a_desktop_exact_detail_inspection.py"
BOUND_PATHS = (
    CONFIG_PATH,
    CONTRACT_PATH,
    CORE_PATH,
    UI_PATH,
    HELPER_PATH,
    FOCUSED_TEST_PATH,
    SCRIPT_PATH,
    AUDIT_TEST_PATH,
)
CASES = {
    "u7-10a-product-desktop": "tests/test_u7_10a_product_desktop_input_workflow.py",
    "u7-11a-single-look-batch": "tests/test_u7_11a_desktop_single_look_batch.py",
    "u7-12b-output-format": "tests/test_u7_12b_desktop_single_photo_output_format.py",
    "u7-12c-display-native": "tests/test_u7_12c_desktop_display_native_preview.py",
    "u7-12g-batch-format": "tests/test_u7_12g_desktop_batch_output_format.py",
    "u7-14a-input-basis": "tests/test_u7_14a_desktop_input_basis_preview.py",
    "u7-15a-representative": "tests/test_u7_15a_desktop_batch_representative_selection.py",
    "u7-15b-visible-strength": "tests/test_u7_15b_desktop_visible_strength_quantization.py",
    "u7-15c-path-disclosure": "tests/test_u7_15c_desktop_batch_representative_path_disclosure.py",
    "u7-16a-exact-detail": FOCUSED_TEST_PATH,
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


def _case_order(order: str) -> tuple[str, ...]:
    names = tuple(sorted(CASES))
    return tuple(reversed(names)) if order == "reverse" else names


def _parent_has_helper(parent_commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{parent_commit}:{HELPER_PATH}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _run_primary(config: dict[str, Any]) -> dict[str, Any]:
    primary = config["primary"]
    detail_config = config["detail"]
    source = (ROOT / primary["input_relative_path"]).resolve(strict=True)
    source_before = sha256_file(source)
    scratch_parent = (ROOT / "tmp").resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix="u7_16a_formal_", dir=scratch_parent
    ) as raw:
        scratch = Path(raw)
        workflow = ProductDesktopWorkflow(
            root=ROOT,
            scratch_root=scratch,
            python_executable=Path(sys.executable),
            tile_size=int(primary["tile_size"]),
            tile_workers=int(primary["tile_workers"]),
            png_compression=6,
        )
        final_path = scratch / "ordinary-final.png"
        try:
            state = workflow.render_previews(source, float(primary["look_amount"]))
            detail = render_exact_export_detail(
                workflow,
                str(primary["style_id"]),
                output_format_id=str(primary["output_format_id"]),
                point=tuple(detail_config["primary_point"]),
                crop_limit=int(detail_config["crop_limit"]),
            )
            receipt = workflow.export(
                str(primary["style_id"]),
                final_path,
                output_format_id=str(primary["output_format_id"]),
            )
            with Image.open(final_path) as opened:
                final_size = tuple(int(value) for value in opened.size)
                oracle = opened.convert("RGB").crop(detail.crop_box)
                oracle_bytes = np.asarray(oracle, dtype=np.uint8).tobytes()
            with Image.open(BytesIO(detail.crop_png)) as opened:
                detail_size = tuple(int(value) for value in opened.size)
                detail_bytes = np.asarray(opened, dtype=np.uint8).tobytes()
            observation = {
                "input_sha256": source_before,
                "preview_input_sha256": state.input_sha256,
                "style_id": detail.style_id,
                "look_amount": detail.look_amount,
                "output_format_id": detail.output_format_id,
                "full_dimensions": list(final_size),
                "detail_dimensions": list(detail_size),
                "crop_box": list(detail.crop_box),
                "detail_crop_sha256": detail.crop_sha256,
                "oracle_rgb_sha256": _sha256(oracle_bytes),
                "detail_rgb_sha256": _sha256(detail_bytes),
                "temporary_full_output_sha256": detail.full_output_sha256,
                "ordinary_full_output_sha256": receipt.output_sha256,
                "temporary_recipe_sha256": detail.strict_recipe_sha256,
                "ordinary_recipe_sha256": receipt.recipe_sha256,
                "temporary_workspace_residue": len(
                    tuple(scratch.glob("u7-16a-detail-*"))
                ),
            }
            final_path.unlink()
            receipt.recipe_path.unlink()
        finally:
            closed = workflow.close()
        observation["workflow_close_clean"] = bool(closed)
        observation["scratch_residue"] = len(tuple(scratch.iterdir()))
    observation["source_sha256_after"] = sha256_file(source)
    return observation


def _run_cases(order: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in _case_order(order):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", CASES[name]],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    commit = _run("git", "rev-parse", "HEAD").decode("ascii").strip()
    config = json.loads(_git_blob(CONFIG_PATH))
    parent_commit = str(config["parent_head"])
    bindings = {path: _sha256(_git_blob(path)) for path in BOUND_PATHS}
    parent_ui = _git_blob(UI_PATH, parent_commit)
    parent_bindings = config["parent_bindings"]
    helper = _git_blob(HELPER_PATH)
    tracked_diff_clean = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    primary = _run_primary(config)
    cases = _run_cases(args.order)

    expected_crop = [2687, 1619, 3199, 2131]
    gates = {
        "committed_source_bound": all(len(value) == 64 for value in bindings.values()),
        "tracked_diff_clean": tracked_diff_clean,
        "parent_product_core_exact": (
            _sha256(_git_blob(CORE_PATH, parent_commit))
            == parent_bindings["product_desktop"]["sha256"]
        ),
        "parent_ui_exact": (
            _sha256(parent_ui) == parent_bindings["product_desktop_ui"]["sha256"]
        ),
        "parent_u7_15c_evidence_exact": (
            _sha256(
                _git_blob(parent_bindings["u7_15c_evidence"]["path"], parent_commit)
            )
            == parent_bindings["u7_15c_evidence"]["sha256"]
        ),
        "parent_detail_absent": (
            b"Inspect selected at 1:1" not in parent_ui
            and not _parent_has_helper(parent_commit)
        ),
        "exact_export_not_preview_source": (
            b"workflow.export(" in helper
            and b"preview_bytes" not in helper
            and b"render_previews" not in helper
        ),
        "primary_source_exact": (
            primary["input_sha256"] == config["primary"]["input_sha256"]
            and primary["source_sha256_after"] == config["primary"]["input_sha256"]
            and primary["preview_input_sha256"] == config["primary"]["input_sha256"]
        ),
        "primary_role_exact": (
            primary["style_id"] == config["primary"]["style_id"]
            and primary["look_amount"] == config["primary"]["look_amount"]
            and primary["output_format_id"] == config["primary"]["output_format_id"]
        ),
        "primary_full_dimensions_exact": (
            primary["full_dimensions"]
            == [config["primary"]["width"], config["primary"]["height"]]
        ),
        "primary_crop_geometry_exact": (
            primary["crop_box"] == expected_crop
            and primary["detail_dimensions"]
            == [config["detail"]["crop_limit"], config["detail"]["crop_limit"]]
        ),
        "primary_full_output_byte_exact": (
            primary["temporary_full_output_sha256"]
            == primary["ordinary_full_output_sha256"]
        ),
        "primary_crop_rgb_exact": (
            primary["oracle_rgb_sha256"] == primary["detail_rgb_sha256"]
        ),
        "successful_cleanup_exact": (
            primary["temporary_workspace_residue"] == 0
            and primary["workflow_close_clean"]
            and primary["scratch_residue"] == 0
        ),
        "all_isolated_behavior_suites_pass": all(
            result["passed"] for result in cases.values()
        ),
    }
    scientific = {
        "schema": "kmcfm.u7-16a-desktop-exact-detail-inspection-report.v1",
        "node_id": "U7.16A",
        "source_commit": commit,
        "bindings": bindings,
        "primary_observation": primary,
        "case_results": {key: cases[key] for key in sorted(cases)},
        "gates": gates,
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "private_windows_python_spatial_detail_only": True,
            "one_exported_pixel_per_viewer_pixel": True,
            "sample_code_or_display_calibrated": False,
            "renderer_or_look_math_changed": False,
            "recipe_or_receipt_schema_changed": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
            "hdr_or_wide_gamut": False,
            "public_or_cross_platform_release": False,
        },
    }
    report = {
        **scientific,
        "scientific_identity": _canonical_sha256(scientific),
        "status": (
            "PASS_PRIVATE_U7_16A_DESKTOP_EXACT_DETAIL_INSPECTION"
            if all(gates.values())
            else "FAIL_CLOSED_U7_16A_DESKTOP_EXACT_DETAIL_INSPECTION"
        ),
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = args.report.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise SystemExit("report destination must be absent") from None
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
