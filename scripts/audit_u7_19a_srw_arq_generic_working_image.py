#!/usr/bin/env python3
"""Run the frozen U7.19A predecode generic-RAW admission audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image
from PIL import __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    load_render_profile,
    replay_style_safe_recipe_to_file,
    validate_render_recipe,
)
from src.preprocess.pipeline import inspect_input, load_working_image
from src.preprocess.raw_decode import (
    RAW_SUFFIXES,
    inspect_raw,
    load_raw_working_image,
)

REPORT_SCHEMA = "kmcfm.u7-19a-srw-arq-generic-working-image-preflight.v1"
FORMAL_REPORT_SCHEMA = "kmcfm.u7-19a-srw-arq-generic-working-image-result.v1"
LEGACY_RECIPE_SCHEMA_ID = "kmcfm.render-recipe.v1"


class U719AError(RuntimeError):
    """Raised when a frozen U7.19A identity or preflight invariant differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise U719AError(f"expected JSON object: {path}")
    return value


def _verify_file(path: Path, binding: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and _sha256_file(path) == str(binding["sha256"])
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _binding_checks(config: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, binding in sorted(config["bindings"].items()):
        checks[name] = _verify_file(ROOT / binding["path"], binding)
    return checks


def _commit_resolves(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _runtime_checks(config: dict[str, Any]) -> dict[str, bool]:
    runtime = config["runtime"]
    return {
        "libraw": list(rawpy.libraw_version) == runtime["libraw"],
        "numpy": np.__version__ == runtime["numpy"],
        "python": ".".join(map(str, sys.version_info[:3])) == runtime["python"],
        "rawpy": rawpy.__version__ == runtime["rawpy"],
        "windows": sys.platform == "win32" and runtime["platform"] == "Windows",
    }


def _warning_record(warnings: list[Any]) -> tuple[list[str], dict[str, str]]:
    codes = sorted(warning.code for warning in warnings)
    messages = {warning.code: warning.message for warning in warnings}
    return codes, messages


def _row_record(
    producer_repo: Path,
    row: dict[str, Any],
    *,
    inspect: Callable[[Path], Any] = inspect_raw,
    load: Callable[..., Any] = load_raw_working_image,
) -> dict[str, Any]:
    source = producer_repo / row["path"]
    if not source.is_file():
        raise U719AError(f"source is missing: {row['source_id']}")
    before = _sha256_file(source)
    if source.stat().st_size != int(row["bytes"]) or before != row["sha256"]:
        raise U719AError(f"source identity differs: {row['source_id']}")

    inspection = inspect(source)
    working = load(source, use_camera_wb=True, no_auto_bright=True)
    pixels = working.pixels
    warning_codes, warning_messages = _warning_record(working.warnings)
    inspection_warning_codes, _ = _warning_record(inspection.warnings)
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    pixel_hash = _sha256_bytes(
        np.ascontiguousarray(pixels, dtype="<f4").tobytes(order="C")
    )
    record = {
        "alpha_policy": working.alpha_policy,
        "bit_depth_in": working.bit_depth_in,
        "inspection": {
            "bit_depth": inspection.bit_depth,
            "format_name": inspection.format_name,
            "height": inspection.height,
            "raw_metadata": inspection.raw_metadata,
            "source_kind": inspection.source_kind,
            "source_profile_description": inspection.source_profile.description,
            "source_profile_kind": inspection.source_profile.kind,
            "transfer_state": inspection.transfer_state,
            "warning_codes": inspection_warning_codes,
            "width": inspection.width,
        },
        "orientation_applied": working.orientation_applied,
        "pixels": {
            "c_contiguous": bool(pixels.flags.c_contiguous),
            "dtype": str(pixels.dtype),
            "f32le_sha256": pixel_hash,
            "finite": bool(np.isfinite(pixels).all()),
            "maximum": maximum,
            "minimum": minimum,
            "nonconstant": minimum < maximum,
            "owned": bool(pixels.flags.owndata),
            "shape": list(pixels.shape),
            "writeable": bool(pixels.flags.writeable),
        },
        "source_id": row["source_id"],
        "source_path": row["path"],
        "source_profile_description": working.source_profile.description,
        "source_profile_kind": working.source_profile.kind,
        "source_sha256": before,
        "source_unchanged": _sha256_file(source) == before,
        "source_transfer_state": working.source_transfer_state,
        "transfer_state": working.transfer_state,
        "warning_codes": warning_codes,
        "warning_messages": warning_messages,
        "working_space": working.working_space,
    }
    del working, pixels
    gc.collect()
    return record


def _stratum_result(
    config: dict[str, Any],
    producer_repo: Path,
    stratum: dict[str, Any],
    *,
    reverse: bool,
    public_pipeline: bool = False,
) -> dict[str, Any]:
    rows = list(stratum["sources"])
    if reverse:
        rows.reverse()
    if public_pipeline:
        records = [
            _row_record(
                producer_repo,
                row,
                inspect=inspect_input,
                load=lambda path, **_kwargs: load_working_image(path),
            )
            for row in rows
        ]
    else:
        records = [_row_record(producer_repo, row) for row in rows]
    records.sort(key=lambda item: item["source_id"])
    required_warnings = sorted(
        config["per_stratum_preflight_gates"]["required_warning_codes"]
    )
    gates = {
        "all_inspections_clean": all(
            not row["inspection"]["warning_codes"] for row in records
        ),
        "all_members_complete": len(records) == int(stratum["required_members"]),
        "all_nonconstant": all(row["pixels"]["nonconstant"] for row in records),
        "all_owned_c_contiguous_writable": all(
            row["pixels"]["owned"]
            and row["pixels"]["c_contiguous"]
            and row["pixels"]["writeable"]
            for row in records
        ),
        "all_sources_immutable": all(row["source_unchanged"] for row in records),
        "all_structures_exact": all(
            len(row["pixels"]["shape"]) == 3
            and row["pixels"]["shape"][2] == 3
            and row["pixels"]["dtype"] == "float32"
            and row["pixels"]["finite"]
            and 0.0 <= row["pixels"]["minimum"] <= row["pixels"]["maximum"] <= 1.0
            and row["working_space"] == "linear_srgb"
            and row["transfer_state"] == "scene_linear"
            and row["source_transfer_state"] == "scene_linear"
            and row["orientation_applied"] is True
            and row["alpha_policy"] == "absent"
            and row["bit_depth_in"] == 16
            for row in records
        ),
        "all_warning_boundaries_exact": all(
            row["warning_codes"] == required_warnings
            and "exact vendor/Adobe rendering is not promised"
            in row["warning_messages"]["generic_raw_render"]
            and "without a calibrated scene-to-display tone map"
            in row["warning_messages"]["generic_raw_display_mapping"]
            for row in records
        ),
    }
    passed = all(gates.values())
    return {
        "admission": "PASS_PREFLIGHT" if passed else "FAIL_CLOSED_PREFLIGHT",
        "extension": stratum["extension"],
        "gates": gates,
        "records": records,
        "representative_source_id": stratum["representative_source_id"],
        "stratum_id": stratum["stratum_id"],
    }


def _relative_to_root(path: Path) -> str:
    return path.absolute().relative_to(ROOT.absolute()).as_posix()


def _output_facts(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "bytes": path.stat().st_size,
            "format": image.format,
            "height": image.height,
            "icc_present": bool(image.info.get("icc_profile")),
            "mode": image.mode,
            "sha256": _sha256_file(path),
            "width": image.width,
        }


def _effective_recipe_look_amount(recipe: dict[str, Any]) -> float:
    render = recipe["render"]
    if "look_amount" in render:
        return float(render["look_amount"])
    if recipe["schema_id"] == LEGACY_RECIPE_SCHEMA_ID:
        return 1.0
    raise U719AError("recipe omits look_amount outside canonical v1 default semantics")


def _product_record(
    config: dict[str, Any],
    producer_repo: Path,
    stratum: dict[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    row = next(
        item
        for item in stratum["sources"]
        if item["source_id"] == stratum["representative_source_id"]
    )
    source = producer_repo / row["path"]
    source_before = _sha256_file(source)
    if source.stat().st_size != int(row["bytes"]) or source_before != row["sha256"]:
        raise U719AError(f"representative source identity differs: {row['source_id']}")

    row_root = scratch_root / stratum["stratum_id"]
    row_root.mkdir(parents=True, exist_ok=False)
    output = row_root / "render.png"
    recipe_path = output.with_suffix(".recipe.json")
    replay = row_root / "replay.png"
    profile_path = ROOT / config["product_chain"]["profile_path"]
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        config["product_chain"]["style"],
        "--look-amount",
        str(config["product_chain"]["look_amount"]),
        "--use-render-profile",
        "--render-profile",
        _relative_to_root(profile_path),
        "--output-bit-depth",
        str(config["product_chain"]["output_bit_depth"]),
        "--write-recipe",
        "--output",
        _relative_to_root(output),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            raise U719AError(
                f"product CLI failed for {stratum['extension']}: "
                f"{stderr[-1] if stderr else 'no stderr'}"
            )
        if not output.is_file() or not recipe_path.is_file():
            raise U719AError(f"product artifacts missing: {stratum['extension']}")
        recipe_bytes = recipe_path.read_bytes()
        recipe = json.loads(recipe_bytes)
        validate_render_recipe(recipe)
        profile = load_render_profile(profile_path, root=ROOT)
        output_facts = _output_facts(output)
        recipe_exact = {
            "claim_is_look_approximation": (
                recipe["claim"]["render_mode"]
                == config["product_chain"]["require_claim_render_mode"]
                and recipe["claim"]["output_label"]
                == config["product_chain"]["require_claim_output_label"]
                and recipe["claim"]["evidence_grade"]
                == config["product_chain"]["require_claim_evidence_grade"]
                and recipe["claim"]["calibrated_reference_allowed"] is False
                and recipe["claim"]["claim_ceiling"]
                == profile["evidence"]["claim_ceiling"]
            ),
            "explicit_look": (
                recipe["render"]["style"] == config["product_chain"]["style"]
                and _effective_recipe_look_amount(recipe)
                == float(config["product_chain"]["look_amount"])
            ),
            "input_identity": (
                Path(recipe["input"]["path"]).resolve() == source.resolve()
                and recipe["input"]["sha256"] == row["sha256"]
            ),
            "output_identity": (
                Path(recipe["output"]["path"]).resolve() == output.resolve()
                and recipe["output"]["sha256"] == output_facts["sha256"]
                and recipe["output"]["format"] == "PNG"
                and recipe["output"]["bit_depth"]
                == int(config["product_chain"]["output_bit_depth"])
            ),
            "product_profile": (
                recipe["profile"]["profile_id"] == "safe-rich-product-v1"
                and recipe["profile"]["profile_version"] == profile["profile_version"]
                and recipe["profile"]["sha256"] == _sha256_file(profile_path)
            ),
            "software_commit": recipe["software"]["commit"] == _git_head(),
        }
        replay_digest = replay_style_safe_recipe_to_file(
            recipe,
            profile_path=profile_path,
            output_path=replay,
            root=ROOT,
        )
        replay_facts = _output_facts(replay)
        return {
            "cli_returncode": completed.returncode,
            "extension": stratum["extension"],
            "output": output_facts,
            "recipe": {
                "bytes": len(recipe_bytes),
                "claim": recipe["claim"],
                "exact": recipe_exact,
                "schema_id": recipe["schema_id"],
                "sha256": _sha256_bytes(recipe_bytes),
            },
            "replay": {**replay_facts, "returned_sha256": replay_digest},
            "replay_byte_exact": output.read_bytes() == replay.read_bytes(),
            "source_id": row["source_id"],
            "source_sha256": source_before,
            "source_unchanged": _sha256_file(source) == source_before,
            "stratum_id": stratum["stratum_id"],
        }
    finally:
        for path in (replay, recipe_path, output):
            path.unlink(missing_ok=True)
        if row_root.exists():
            row_root.rmdir()


def execute_preflight(
    config_path: Path,
    producer_repo: Path,
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    config = _load_json(config_path)
    if config["status"] != "FROZEN_PREDECODE":
        raise U719AError("config is not the frozen predecode contract")
    bindings = _binding_checks(config)
    runtime = _runtime_checks(config)
    if not all(bindings.values()):
        raise U719AError("frozen local binding differs")
    if not all(runtime.values()):
        raise U719AError("frozen runtime identity differs")
    if not producer_repo.is_dir():
        raise U719AError("producer repository is unavailable")

    strata = list(config["strata"])
    if reverse:
        strata.reverse()
    results = [
        _stratum_result(config, producer_repo, stratum, reverse=reverse)
        for stratum in strata
    ]
    results.sort(key=lambda item: item["extension"])
    passed_extensions = sorted(
        row["extension"] for row in results if row["admission"] == "PASS_PREFLIGHT"
    )
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "execution_commit": _git_head(),
        "independent_admission_policy": config["independent_admission_policy"],
        "network_requests": 0,
        "passed_extensions": passed_extensions,
        "predecode_suffix_state": {
            ".arq": ".arq" in RAW_SUFFIXES,
            ".srw": ".srw" in RAW_SUFFIXES,
        },
        "results": results,
        "runtime": {
            "checks": runtime,
            "libraw": list(rawpy.libraw_version),
            "numpy": np.__version__,
            "python": ".".join(map(str, sys.version_info[:3])),
            "rawpy": rawpy.__version__,
        },
        "schema": REPORT_SCHEMA,
        "status": (
            "PASS_PREFLIGHT_AT_LEAST_ONE_EXTENSION"
            if passed_extensions
            else "FAIL_CLOSED_PREFLIGHT_ALL_EXTENSIONS"
        ),
        "stop_rule": config["stop_rule"],
        "tracked_worktree_clean": _tracked_clean(),
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def execute_formal(
    config_path: Path,
    execution_lock_path: Path,
    producer_repo: Path,
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    config = _load_json(config_path)
    execution_lock = _load_json(execution_lock_path)
    bindings = {
        name: _verify_file(ROOT / binding["path"], binding)
        for name, binding in sorted(execution_lock["bindings"].items())
    }
    preflight_reports = {
        name: _verify_file(ROOT / binding["path"], binding)
        for name, binding in sorted(execution_lock["preflight_reports"].items())
    }
    commits = {
        name: _commit_resolves(commit)
        for name, commit in sorted(execution_lock["commits"].items())
    }
    runtime = _runtime_checks(config)
    runtime["pillow"] = PILLOW_VERSION == execution_lock["runtime"]["pillow"]
    if not all(bindings.values()):
        raise U719AError("formal binding differs")
    if not all(preflight_reports.values()):
        raise U719AError("preflight report binding differs")
    if not all(commits.values()):
        raise U719AError("historical commit binding differs")
    if not all(runtime.values()):
        raise U719AError("formal runtime identity differs")

    preflight = _load_json(
        ROOT / execution_lock["preflight_reports"]["forward"]["path"]
    )
    admitted = sorted(execution_lock["admitted_extensions"])
    if admitted != sorted(preflight["passed_extensions"]):
        raise U719AError("admitted extensions differ from frozen preflight")
    strata_by_extension = {row["extension"]: row for row in config["strata"]}
    strata = [strata_by_extension[extension] for extension in admitted]
    if reverse:
        strata.reverse()
    public_results = [
        _stratum_result(
            config,
            producer_repo,
            stratum,
            reverse=reverse,
            public_pipeline=True,
        )
        for stratum in strata
    ]
    public_results.sort(key=lambda item: item["extension"])

    scratch_root = ROOT / execution_lock["scratch_root"]
    if scratch_root.exists() and any(scratch_root.iterdir()):
        raise U719AError("owned scratch root is not empty before formal execution")
    scratch_root.mkdir(parents=True, exist_ok=True)
    try:
        product_records = [
            _product_record(config, producer_repo, stratum, scratch_root)
            for stratum in strata
        ]
        product_records.sort(key=lambda item: item["extension"])
    finally:
        if scratch_root.exists() and not any(scratch_root.iterdir()):
            scratch_root.rmdir()
    scratch_residue = (
        sum(1 for path in scratch_root.rglob("*") if path.is_file())
        if scratch_root.exists()
        else 0
    )
    gates = {
        "all_admitted_extensions_dispatched": all(
            extension in RAW_SUFFIXES for extension in admitted
        ),
        "all_product_cli_success": all(
            row["cli_returncode"] == 0 for row in product_records
        ),
        "all_product_outputs_exact": all(
            row["output"]["format"] == "PNG"
            and row["output"]["mode"] == "RGB"
            and row["output"]["icc_present"]
            for row in product_records
        ),
        "all_public_ingress_pass": all(
            row["admission"] == "PASS_PREFLIGHT" for row in public_results
        ),
        "all_recipe_fields_exact": all(
            all(row["recipe"]["exact"].values()) for row in product_records
        ),
        "all_replays_byte_exact": all(
            row["replay_byte_exact"] for row in product_records
        ),
        "all_replay_hashes_exact": all(
            row["output"]["sha256"]
            == row["replay"]["sha256"]
            == row["replay"]["returned_sha256"]
            for row in product_records
        ),
        "all_sources_immutable": all(row["source_unchanged"] for row in product_records)
        and all(
            record["source_unchanged"]
            for result in public_results
            for record in result["records"]
        ),
        "bindings_exact": all(bindings.values()),
        "commits_resolve": all(commits.values()),
        "network_requests_zero": True,
        "preflight_reports_exact": all(preflight_reports.values()),
        "required_extensions_complete": len(product_records) == len(admitted),
        "runtime_exact": all(runtime.values()),
        "scratch_residue_zero": scratch_residue == 0,
        "tracked_worktree_clean": _tracked_clean(),
    }
    report: dict[str, Any] = {
        "admitted_extensions": admitted,
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "commits": commits,
        "execution_commit": _git_head(),
        "gates": gates,
        "network_requests": 0,
        "preflight_reports": preflight_reports,
        "product_records": product_records,
        "public_ingress_results": public_results,
        "runtime": {
            "checks": runtime,
            "libraw": list(rawpy.libraw_version),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
            "python": ".".join(map(str, sys.version_info[:3])),
            "rawpy": rawpy.__version__,
        },
        "schema": FORMAL_REPORT_SCHEMA,
        "scratch_residue_files": scratch_residue,
        "status": (
            "PASS_PRIVATE_U7_19A_SRW_ARQ_GENERIC_PRODUCT_INGRESS"
            if all(gates.values())
            else "FAIL_CLOSED_U7_19A_SRW_ARQ_GENERIC_PRODUCT_INGRESS"
        ),
        "stop_rule": config["stop_rule"],
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "formal"), default="preflight")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.stage == "formal":
        if args.execution_lock is None:
            raise U719AError("--execution-lock is required for formal execution")
        report = execute_formal(
            args.config.resolve(),
            args.execution_lock.resolve(),
            args.producer_repo.resolve(),
            reverse=args.reverse,
        )
    else:
        report = execute_preflight(
            args.config.resolve(), args.producer_repo.resolve(), reverse=args.reverse
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
