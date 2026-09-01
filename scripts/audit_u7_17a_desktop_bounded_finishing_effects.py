#!/usr/bin/env python3
"""Committed-head formal audit for U7.17A desktop finishing effects."""

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
    DESKTOP_EFFECT_CAPS,
    DesktopFinishingEffects,
    ProductDesktopWorkflow,
    sha256_file,
)
from src.inference.product_detail_inspection import render_exact_export_detail

CONFIG_PATH = "configs/u7_17a_desktop_bounded_finishing_effects_v1.json"
CONTRACT_PATH = "docs/planning/U7_17A_DESKTOP_BOUNDED_FINISHING_EFFECTS_CONTRACT.md"
CORE_PATH = "src/inference/product_desktop.py"
UI_PATH = "src/inference/product_desktop_ui.py"
DETAIL_PATH = "src/inference/product_detail_inspection.py"
FOCUSED_TEST_PATH = "tests/test_u7_17a_desktop_bounded_finishing_effects.py"
SCRIPT_PATH = "scripts/audit_u7_17a_desktop_bounded_finishing_effects.py"
AUDIT_TEST_PATH = "tests/test_audit_u7_17a_desktop_bounded_finishing_effects.py"
BOUND_PATHS = (
    CONFIG_PATH,
    CONTRACT_PATH,
    CORE_PATH,
    UI_PATH,
    DETAIL_PATH,
    FOCUSED_TEST_PATH,
    SCRIPT_PATH,
    AUDIT_TEST_PATH,
)
CASES = {
    "u7-10a-input-workflow": "tests/test_u7_10a_product_desktop_input_workflow.py",
    "u7-11a-batch": "tests/test_u7_11a_desktop_single_look_batch.py",
    "u7-12a-close": "tests/test_u7_12a_desktop_foreground_worker_close_safety.py",
    "u7-12b-format": "tests/test_u7_12b_desktop_single_photo_output_format.py",
    "u7-12c-preview": "tests/test_u7_12c_desktop_display_native_preview.py",
    "u7-12g-batch-format": "tests/test_u7_12g_desktop_batch_output_format.py",
    "u7-14a-input-basis": "tests/test_u7_14a_desktop_input_basis_preview.py",
    "u7-14b-scratch": "tests/test_u7_14b_desktop_canonical_scratch_boundary.py",
    "u7-15a-representative": "tests/test_u7_15a_desktop_batch_representative_selection.py",
    "u7-15b-strength": "tests/test_u7_15b_desktop_visible_strength_quantization.py",
    "u7-15c-disclosure": "tests/test_u7_15c_desktop_batch_representative_path_disclosure.py",
    "u7-16a-detail": "tests/test_u7_16a_desktop_exact_detail_inspection.py",
    "u7-17a-effects": FOCUSED_TEST_PATH,
}


def _run(*args: str) -> bytes:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True).stdout


def _git_blob(path: str, commit: str = "HEAD") -> bytes:
    return _run("git", "show", f"{commit}:{path}")


def _git_oid(path: str, commit: str = "HEAD") -> str:
    return _run("git", "rev-parse", f"{commit}:{path}").decode("ascii").strip()


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


def _rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def _run_primary(config: dict[str, Any]) -> dict[str, Any]:
    primary = config["primary"]
    source = (ROOT / primary["input_relative_path"]).resolve(strict=True)
    source_before = sha256_file(source)
    effects = DesktopFinishingEffects(**config["effects"]["caps"])
    parent_evidence = json.loads(
        _git_blob(config["parent_bindings"]["u7_16a_evidence"]["path"])
    )
    expected_zero_sha = parent_evidence["primary_observation"][
        "ordinary_full_output_sha256"
    ]
    scratch_parent = (ROOT / "tmp").resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix="u7_17a_formal_", dir=scratch_parent
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
        zero_path = scratch / "zero.png"
        effect_path = scratch / "effect.png"
        try:
            state = workflow.render_previews(source, float(primary["look_amount"]))
            zero_command = workflow.export_command(str(primary["style_id"]), zero_path)
            zero = workflow.export(str(primary["style_id"]), zero_path)
            detail = render_exact_export_detail(
                workflow,
                str(primary["style_id"]),
                output_format_id=str(primary["output_format_id"]),
                point=(0.73, 0.31),
                crop_limit=512,
                effects=effects,
            )
            effect = workflow.export(
                str(primary["style_id"]),
                effect_path,
                output_format_id=str(primary["output_format_id"]),
                effects=effects,
            )
            zero_rgb = _rgb8(zero_path)
            effect_rgb = _rgb8(effect_path)
            with Image.open(BytesIO(detail.crop_png)) as opened:
                detail_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            with Image.open(effect_path) as opened:
                oracle_rgb = np.asarray(
                    opened.convert("RGB").crop(detail.crop_box), dtype=np.uint8
                )
            zero_boundary = (zero_rgb == 0) | (zero_rgb == 255)
            effect_boundary = (effect_rgb == 0) | (effect_rgb == 255)
            expected_effect_recipe = {
                "grain": {"strength": 0.05, "seed": 7, "color": True},
                "halation": {
                    "strength": 0.15,
                    "model": "simple",
                    "preset": None,
                    "control_mode": "locked",
                    "resolved_parameters": None,
                },
                "dust": {"strength": 0.02, "seed": 24},
            }
            observation = {
                "input_sha256": source_before,
                "preview_input_sha256": state.input_sha256,
                "style_id": effect.style_id,
                "look_amount": effect.look_amount,
                "output_format_id": effect.output_format_id,
                "dimensions": list(reversed(effect_rgb.shape[:2])),
                "effects": effect.effects.receipt_identity(),
                "effect_recipe_exact": (
                    effect.recipe["render"]["effects"] == expected_effect_recipe
                ),
                "zero_command_effect_options_absent": not bool(
                    {"--grain", "--halation", "--dust", "--seed"} & set(zero_command)
                ),
                "expected_parent_zero_sha256": expected_zero_sha,
                "zero_output_sha256": zero.output_sha256,
                "temporary_effect_output_sha256": detail.full_output_sha256,
                "ordinary_effect_output_sha256": effect.output_sha256,
                "effect_rgb_min": int(effect_rgb.min()),
                "effect_rgb_max": int(effect_rgb.max()),
                "effect_boundary_fraction": float(effect_boundary.mean()),
                "new_boundary_fraction": float(
                    np.mean(effect_boundary & ~zero_boundary, dtype=np.float64)
                ),
                "changed_component_fraction": float(
                    np.mean(effect_rgb != zero_rgb, dtype=np.float64)
                ),
                "detail_rgb_sha256": _sha256(detail_rgb.tobytes()),
                "oracle_rgb_sha256": _sha256(oracle_rgb.tobytes()),
                "temporary_workspace_residue": len(
                    tuple(scratch.glob("u7-16a-detail-*"))
                ),
            }
            for receipt in (zero, effect):
                receipt.output_path.unlink()
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
        failure_lines = sorted(
            line.strip()
            for line in (completed.stdout + "\n" + completed.stderr).splitlines()
            if line.startswith(("FAILED ", "ERROR "))
        )
        results[name] = {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "failure_lines": failure_lines,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    commit = _run("git", "rev-parse", "HEAD").decode("ascii").strip()
    config = json.loads(_git_blob(CONFIG_PATH))
    parent = str(config["parent_head"])
    bindings = {
        path: {
            "git_blob": _git_oid(path),
            "bytes": len(_git_blob(path)),
            "sha256": _sha256(_git_blob(path)),
        }
        for path in BOUND_PATHS
    }
    tracked_clean = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    primary = _run_primary(config)
    cases = _run_cases(args.order)
    parent_exact = all(
        _git_oid(binding["path"], parent) == binding["git_blob"]
        and _sha256(_git_blob(binding["path"], parent)) == binding["sha256"]
        for binding in config["parent_bindings"].values()
    )
    effects = config["effects"]
    gates = {
        "committed_source_bound": all(
            len(row["git_blob"]) == 40 and len(row["sha256"]) == 64
            for row in bindings.values()
        ),
        "tracked_diff_clean": tracked_clean,
        "all_parent_git_objects_exact": parent_exact,
        "frozen_caps_exact": (
            effects["caps"] == DESKTOP_EFFECT_CAPS
            and effects["seed"] == 7
            and effects["dust_seed"] == 24
            and effects["halation_model"] == "simple"
        ),
        "primary_source_immutable": (
            primary["input_sha256"] == config["primary"]["input_sha256"]
            and primary["preview_input_sha256"] == config["primary"]["input_sha256"]
            and primary["source_sha256_after"] == config["primary"]["input_sha256"]
        ),
        "primary_role_exact": (
            primary["style_id"] == config["primary"]["style_id"]
            and primary["look_amount"] == config["primary"]["look_amount"]
            and primary["output_format_id"] == config["primary"]["output_format_id"]
            and primary["dimensions"]
            == [config["primary"]["width"], config["primary"]["height"]]
        ),
        "zero_effect_parent_output_exact": (
            primary["zero_command_effect_options_absent"]
            and primary["zero_output_sha256"] == primary["expected_parent_zero_sha256"]
        ),
        "maximum_effect_recipe_exact": (
            primary["effects"]
            == {
                **config["effects"]["caps"],
                "seed": 7,
                "halation_model": "simple",
            }
            and primary["effect_recipe_exact"]
        ),
        "maximum_effect_detail_and_final_exact": (
            primary["temporary_effect_output_sha256"]
            == primary["ordinary_effect_output_sha256"]
            and primary["detail_rgb_sha256"] == primary["oracle_rgb_sha256"]
        ),
        "maximum_effect_visible": primary["changed_component_fraction"] > 0.0,
        "maximum_effect_no_code_boundary": (
            primary["effect_rgb_min"] >= 4
            and primary["effect_rgb_max"] <= 251
            and primary["effect_boundary_fraction"] == 0.0
            and primary["new_boundary_fraction"] == 0.0
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
        "schema": "kmcfm.u7-17a-desktop-bounded-finishing-effects-report.v1",
        "node_id": "U7.17A",
        "source_commit": commit,
        "bindings": bindings,
        "primary_observation": primary,
        "case_results": {key: cases[key] for key in sorted(cases)},
        "gates": gates,
        "claim": {
            "output_label": "film-inspired / Look Approximation",
            "evidence_grade": "look-approximation",
            "private_desktop_finishing_controls": True,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
            "arbitrary_media_or_population_safety": False,
            "public_or_cross_platform_release": False,
        },
    }
    report = {
        "scientific": scientific,
        "scientific_sha256": _canonical_sha256(scientific),
        "status": (
            "PASS_PRIVATE_U7_17A_DESKTOP_BOUNDED_FINISHING_EFFECTS"
            if all(gates.values())
            else "FAIL_CLOSED_U7_17A_DESKTOP_BOUNDED_FINISHING_EFFECTS"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
