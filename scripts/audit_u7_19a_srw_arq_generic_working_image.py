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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.raw_decode import (
    RAW_SUFFIXES,
    inspect_raw,
    load_raw_working_image,
)

REPORT_SCHEMA = "kmcfm.u7-19a-srw-arq-generic-working-image-preflight.v1"


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
) -> dict[str, Any]:
    rows = list(stratum["sources"])
    if reverse:
        rows.reverse()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute_preflight(
        args.config.resolve(), args.producer_repo.resolve(), reverse=args.reverse
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
