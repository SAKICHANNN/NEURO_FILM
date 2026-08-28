#!/usr/bin/env python3
"""Audit product-catalog enforcement across recipe build, verify and replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    build_render_recipe,
    load_render_profile,
    verify_render_recipe_inputs,
)
from src.inference.style_safe_engine import (
    replay_style_safe_color_recipe,
)

WORK_ROOT = ROOT / "tmp/u7_2j_product_recipe_catalog_enforcement_v1"
SCRIPT = ROOT / "scripts/render_film.py"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
U7_2I_EVIDENCE = (
    ROOT / "docs/evidence/U7_2I_PRODUCT_LOOK_CATALOG_ENFORCEMENT_RESULT.json"
)
CORE = ROOT / "src/inference/render_contract.py"
TEST = ROOT / "tests/test_u7_2j_product_recipe_catalog_enforcement.py"
ERROR = "available product-catalog look"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _run_recipe(
    source: Path, output: Path, *, profile: Path, style: str
) -> tuple[int, dict[str, Any] | None]:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--style",
            style,
            "--use-render-profile",
            "--render-profile",
            str(profile),
            "--write-recipe",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    recipe_path = output.with_suffix(".recipe.json")
    recipe = (
        json.loads(recipe_path.read_text(encoding="utf-8"))
        if recipe_path.is_file()
        else None
    )
    return completed.returncode, recipe


def _dummy_render(style: str) -> dict[str, Any]:
    return {
        "engine_id": "safe_lab_v1",
        "preset": "safe-rich",
        "style": style,
        "seed": 7,
        "color_parameters": {},
        "effects": {
            "grain": {"strength": 0.0, "seed": 7, "color": True},
            "halation": {
                "strength": 0.0,
                "model": "simple",
                "preset": None,
                "control_mode": "locked",
                "resolved_parameters": None,
            },
            "dust": {"strength": 0.0, "seed": 24},
        },
    }


def _capture_error(call: Callable[[], object]) -> dict[str, str]:
    try:
        call()
    except ValueError as error:
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "", "message": ""}


def _build_rejection(style: str, profile: dict[str, Any]) -> dict[str, str]:
    return _capture_error(
        lambda: build_render_recipe(
            profile_path=PRODUCT_PROFILE,
            profile=profile,
            input_path=WORK_ROOT / "must_not_hash_input.png",
            input_metadata={},
            render_metadata=_dummy_render(style),
            output_path=WORK_ROOT / "must_not_hash_output.png",
            output_format="PNG",
            output_bit_depth=8,
            output_icc_fingerprint_sha256="1" * 64,
            output_claim={},
            software_commit="2" * 40,
        )
    )


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent = json.loads(U7_2I_EVIDENCE.read_text(encoding="utf-8"))
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    WORK_ROOT.mkdir(parents=True)
    source = WORK_ROOT / "source.png"
    _source(source)
    source_before = _sha(source)
    product = load_render_profile(PRODUCT_PROFILE, root=ROOT)
    product_results: dict[str, dict[str, Any]] = {}
    rejection_results: dict[str, dict[str, Any]] = {}
    try:
        for style in order:
            output = WORK_ROOT / f"product-{style}.png"
            returncode, recipe = _run_recipe(
                source, output, profile=PRODUCT_PROFILE, style=style
            )
            if recipe is None:
                raise RuntimeError(f"available product recipe missing for {style}")
            verify_render_recipe_inputs(recipe, profile_path=PRODUCT_PROFILE, root=ROOT)
            replay_a = replay_style_safe_color_recipe(
                recipe, profile_path=PRODUCT_PROFILE, root=ROOT
            )
            replay_b = replay_style_safe_color_recipe(
                recipe, profile_path=PRODUCT_PROFILE, root=ROOT
            )
            product_results[style] = {
                "returncode": returncode,
                "output_sha256": _sha(output),
                "expected_u7_2i_output_sha256": parent["output_sha256"][style],
                "replay_float32_sha256": _array_sha(replay_a),
                "repeat_replay_exact": _array_sha(replay_a) == _array_sha(replay_b),
            }

        legacy_output = WORK_ROOT / "legacy-hp5.png"
        legacy_returncode, legacy_recipe = _run_recipe(
            source, legacy_output, profile=LEGACY_PROFILE, style="hp5"
        )
        if legacy_recipe is None:
            raise RuntimeError("legacy HP5 recipe missing")
        verify_render_recipe_inputs(legacy_recipe, profile_path=LEGACY_PROFILE, root=ROOT)
        legacy_a = replay_style_safe_color_recipe(
            legacy_recipe, profile_path=LEGACY_PROFILE, root=ROOT
        )
        legacy_b = replay_style_safe_color_recipe(
            legacy_recipe, profile_path=LEGACY_PROFILE, root=ROOT
        )

        rejected_order = list(config["rejected_product_recipe_styles"])
        if order != tuple(config["available_product_looks"]):
            rejected_order.reverse()
        for style in rejected_order:
            forged = copy.deepcopy(legacy_recipe)
            forged["profile"] = {
                "profile_id": product["profile_id"],
                "profile_version": product["profile_version"],
                "sha256": _sha(PRODUCT_PROFILE),
            }
            forged["assets"] = copy.deepcopy(product["assets"])
            forged["render"]["style"] = style
            forged["render"]["color_parameters"] = copy.deepcopy(
                product["style_parameters"].get(style, {})
            )
            forged["claim"]["claim_ceiling"] = product["evidence"]["claim_ceiling"]
            forged["input"]["path"] = str(WORK_ROOT / "must_not_hash_or_decode.png")
            build_error = _build_rejection(style, product)
            verify_error = _capture_error(
                lambda forged=forged: verify_render_recipe_inputs(
                    forged, profile_path=PRODUCT_PROFILE, root=ROOT
                )
            )
            replay_error = _capture_error(
                lambda forged=forged: replay_style_safe_color_recipe(
                    forged, profile_path=PRODUCT_PROFILE, root=ROOT
                )
            )
            rejection_results[style or "<empty>"] = {
                "build_error_type": build_error["type"],
                "build_error": build_error["message"],
                "verify_error_type": verify_error["type"],
                "verify_error": verify_error["message"],
                "replay_error_type": replay_error["type"],
                "replay_error": replay_error["message"],
            }

        product_exact = all(
            row["returncode"] == 0
            and row["output_sha256"] == row["expected_u7_2i_output_sha256"]
            and row["repeat_replay_exact"]
            for row in product_results.values()
        )
        rejection_exact = all(
            ERROR in row["build_error"]
            and ERROR in row["verify_error"]
            and bool(row["replay_error"])
            and "must_not_hash_or_decode" not in row["verify_error"]
            and "must_not_hash_or_decode" not in row["replay_error"]
            for row in rejection_results.values()
        )
        forbidden_residue = any(
            (WORK_ROOT / name).exists()
            for name in (
                "must_not_hash_input.png",
                "must_not_hash_output.png",
                "must_not_hash_or_decode.png",
            )
        )
        gates = {
            "source_immutable": _sha(source) == source_before,
            "parent_u7_2i_evidence_exact": _sha(U7_2I_EVIDENCE)
            == config["source_locks"]["parent_u7_2i_evidence_sha256"],
            "product_profile_exact": _sha(PRODUCT_PROFILE)
            == config["source_locks"]["product_profile_sha256"],
            "render_contract_changed_from_prechange_lock": _sha(CORE)
            != config["source_locks"]["render_contract_sha256"],
            "non_catalog_build_verify_replay_reject_before_io": rejection_exact,
            "forbidden_io_residue_zero": not forbidden_residue,
            "three_available_product_outputs_and_replays_exact": product_exact,
            "legacy_hp5_output_exact": legacy_returncode == 0
            and _sha(legacy_output)
            == parent["selection_result"]["legacy_hp5_output_sha256"],
            "legacy_hp5_repeat_replay_exact": _array_sha(legacy_a)
            == _array_sha(legacy_b),
        }
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
        report = {
            "schema_id": "neuro-film.u7-2j-product-recipe-catalog-enforcement-result.v1",
            "experiment_id": "U7.2J",
            "implementation_commit": head,
            "bindings": {
                "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "config_sha256": _sha(config_path),
                "render_contract_sha256": _sha(CORE),
                "focused_test_sha256": _sha(TEST),
                "parent_u7_2i_evidence_sha256": _sha(U7_2I_EVIDENCE),
            },
            "product_results": dict(sorted(product_results.items())),
            "rejection_results": dict(sorted(rejection_results.items())),
            "legacy_hp5_result": {
                "returncode": legacy_returncode,
                "output_sha256": _sha(legacy_output),
                "replay_float32_sha256": _array_sha(legacy_a),
                "repeat_replay_exact": _array_sha(legacy_a) == _array_sha(legacy_b),
            },
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "decision": (
                "Enforce the authoritative available product-look catalog in the "
                "shared recipe builder and verifier before file hashing or decode."
            ),
            "claim_ceiling": config["claim_ceiling"],
        }
    finally:
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
    report["owned_runtime_residue_zero"] = not WORK_ROOT.exists()
    if not report["owned_runtime_residue_zero"]:
        report["status"] = "FAIL_CLOSED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_2j_product_recipe_catalog_enforcement_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    order = tuple(config["available_product_looks"])
    if args.order == "reverse":
        order = tuple(reversed(order))
    report = build_report(config_path=args.config, order=order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
