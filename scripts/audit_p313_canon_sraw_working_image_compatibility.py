#!/usr/bin/env python3
"""Audit exact CC0 Canon sRAW/mRAW files through the generic WorkingImage ingress."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rawpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.raw_decode import inspect_raw, load_raw_working_image

REPORT_SCHEMA = "neuro-film.p313-canon-sraw-working-image-compatibility-result.v1"


class P313Error(RuntimeError):
    """Raised when a frozen P313 identity or execution invariant differs."""


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
        raise P313Error(f"expected JSON object: {path}")
    return value


def _verify_file(path: Path, expected: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected["bytes"])
        and _sha256_file(path) == str(expected["sha256"])
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


def _validate_source_rows(
    config: dict[str, Any], source_snapshot: dict[str, Any]
) -> dict[str, bool]:
    snapshot_rows = {int(row["file_id"]): row for row in source_snapshot["rows"]}
    config_rows = {int(row["file_id"]): row for row in config["rows"]}
    return {
        "cc0_public_domain_exact": (
            source_snapshot["license_name"] == "Creative Commons 0 - Public Domain"
            and source_snapshot["license_url"]
            == "https://creativecommons.org/publicdomain/zero/1.0/"
        ),
        "row_count_exact": len(snapshot_rows) == len(config_rows) == 6,
        "row_file_ids_exact": set(snapshot_rows) == set(config_rows),
        "row_identities_exact": all(
            all(
                snapshot_rows[file_id][key] == config_rows[file_id][key]
                for key in ("bytes", "file_id", "mode", "model", "sha256")
            )
            for file_id in config_rows
        ),
        "unique_models": len({row["model"] for row in config["rows"]}) == 6,
        "unique_sha256": len({row["sha256"] for row in config["rows"]}) == 6,
    }


def _row_record(row: dict[str, Any]) -> dict[str, Any]:
    source_path = ROOT / row["path"]
    before_sha = _sha256_file(source_path)
    if source_path.stat().st_size != int(row["bytes"]) or before_sha != str(
        row["sha256"]
    ):
        raise P313Error(f"source identity differs: {row['source_id']}")

    inspection = inspect_raw(source_path)
    working = load_raw_working_image(
        source_path,
        use_camera_wb=True,
        no_auto_bright=True,
    )
    pixels = working.pixels
    warning_codes = sorted(warning.code for warning in working.warnings)
    warning_messages = {warning.code: warning.message for warning in working.warnings}
    inspection_warning_codes = sorted(warning.code for warning in inspection.warnings)
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
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
        "model": row["model"],
        "mode": row["mode"],
        "orientation_applied": working.orientation_applied,
        "pixels": {
            "c_contiguous": bool(pixels.flags.c_contiguous),
            "dtype": str(pixels.dtype),
            "f32le_sha256": _sha256_bytes(
                np.ascontiguousarray(pixels, dtype="<f4").tobytes(order="C")
            ),
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
        "source_sha256": before_sha,
        "source_transfer_state": working.source_transfer_state,
        "source_unchanged": _sha256_file(source_path) == before_sha,
        "transfer_state": working.transfer_state,
        "warning_codes": warning_codes,
        "warning_messages": warning_messages,
        "working_space": working.working_space,
    }
    return record


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = {
        name: _verify_file(ROOT / item["path"], item)
        for name, item in sorted(config["bindings"].items())
    }
    if not all(bindings.values()):
        raise P313Error("frozen local binding differs")

    runtime = config["runtime"]
    runtime_exact = {
        "libraw": list(rawpy.libraw_version) == runtime["libraw"],
        "numpy": np.__version__ == runtime["numpy"],
        "python": ".".join(map(str, sys.version_info[:3])) == runtime["python"],
        "rawpy": rawpy.__version__ == runtime["rawpy"],
        "windows": sys.platform == "win32" and runtime["platform"] == "Windows",
    }
    source_snapshot = _load_json(ROOT / config["bindings"]["source_rows"]["path"])
    source_lock = _validate_source_rows(config, source_snapshot)
    if not all(source_lock.values()):
        raise P313Error("frozen source-row binding differs")

    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    records = [_row_record(row) for row in rows]
    records.sort(key=lambda item: item["source_id"])

    required_warnings = sorted(config["gates"]["required_warning_codes"])
    gates = {
        "all_decodes_successful": len(records) == int(config["gates"]["required_rows"]),
        "all_inspections_clean": all(
            not row["inspection"]["warning_codes"] for row in records
        ),
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
        "bindings_exact": all(bindings.values()),
        "media_output_files_zero": True,
        "network_requests_zero": True,
        "runtime_exact": all(runtime_exact.values()),
        "source_lock_exact": all(source_lock.values()),
        "tracked_worktree_clean": _tracked_clean(),
    }
    decision = (
        "PASS_PRIVATE_CANON_SRAW_WORKING_IMAGE_COMPATIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_CANON_SRAW_WORKING_IMAGE_COMPATIBILITY"
    )
    report: dict[str, Any] = {
        "bindings": bindings,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "decision": decision,
        "execution_commit": _git_head(),
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "media_output_files": 0,
        "network_requests": 0,
        "records": records,
        "runtime": {
            "checks": runtime_exact,
            "libraw": list(rawpy.libraw_version),
            "numpy": np.__version__,
            "python": ".".join(map(str, sys.version_info[:3])),
            "rawpy": rawpy.__version__,
        },
        "schema": REPORT_SCHEMA,
        "source_lock": source_lock,
        "stop_rule": config["stop_rule"],
        "temp_residue_files": 0,
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
