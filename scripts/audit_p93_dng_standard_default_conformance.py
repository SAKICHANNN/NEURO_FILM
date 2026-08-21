#!/usr/bin/env python3
"""Audit DNG standard-default receipts against LibRaw metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rawpy

from src.preprocess.dng_libraw_conformance import compare_dng_receipt_to_libraw
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.dng_metadata_v2 import (
    STANDARD_LOGICAL_PATH,
    STANDARD_SHA256,
    build_dng_capture_metadata_receipt_v2,
)
from src.preprocess.raw_decode import inspect_raw

REPORT_SCHEMA = "neuro_film.p93_dng_standard_default_conformance_result.v1"
_ALLOWED_DEFAULTS = {
    "cfa_layout": {
        "count": 1,
        "ifd_paths": [],
        "tiff_type": "SHORT",
        "value": [1],
        "value_origin": "dng_standard_default_1_7_1_0",
    },
    "cfa_plane_color": {
        "count": 3,
        "ifd_paths": [],
        "tiff_type": "BYTE",
        "value": [0, 1, 2],
        "value_origin": "dng_standard_default_1_7_1_0",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_bindings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    bindings = config["bindings"]
    checked: dict[str, str] = {}
    for key in (
        "contract",
        "standard_specification",
        "v1_implementation",
        "v2_implementation",
        "raw_decode",
        "conformance_implementation",
        "runner",
    ):
        logical_path = bindings[f"{key}_path"]
        path = ROOT / logical_path
        actual = _sha256_file(path)
        if actual != bindings[f"{key}_sha256"]:
            raise ValueError(f"binding mismatch for {key}")
        checked[f"{key}_path"] = logical_path
        checked[f"{key}_sha256"] = actual

    if bindings["standard_specification_path"] != STANDARD_LOGICAL_PATH:
        raise ValueError("standard specification logical path mismatch")
    if bindings["standard_specification_sha256"] != STANDARD_SHA256:
        raise ValueError("standard specification implementation mismatch")

    manifest_path = ROOT / bindings["source_manifest_path"]
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != bindings["source_manifest_sha256"]:
        raise ValueError("source manifest binding mismatch")
    checked["source_manifest_path"] = bindings["source_manifest_path"]
    checked["source_manifest_sha256"] = manifest_sha
    return _load_json(manifest_path), checked


def _manifest_row(manifest: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [row for row in manifest["candidates"] if row["id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected one source manifest row for {source_id}")
    return matches[0]


def _validate_fact_origins(receipt: dict[str, Any]) -> tuple[bool, list[str]]:
    defaults = receipt["standard_defaults_used"]
    if not isinstance(defaults, list) or len(defaults) != len(set(defaults)):
        return False, []
    if any(name not in _ALLOWED_DEFAULTS for name in defaults):
        return False, []
    for name, record in receipt["facts"].items():
        if name in defaults:
            if record != _ALLOWED_DEFAULTS[name]:
                return False, []
        elif record.get("value_origin") != "explicit_ifd_tag":
            return False, []
    return True, sorted(defaults)


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    manifest, bindings = _validate_bindings(config)
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    output_rows: list[dict[str, Any]] = []

    for row in rows:
        source = ROOT / row["logical_path"]
        source_fact = _manifest_row(manifest, row["source_id"])
        manifest_match = (
            source_fact["path"] == row["logical_path"]
            and source_fact["sha256"] == row["source_sha256"]
            and manifest["source"]["declared_license"]
            == "CC0/Public Domain per each exact repository row"
        )
        receipt = build_dng_capture_metadata_receipt_v2(
            source,
            source_id=row["source_id"],
            logical_path=row["logical_path"],
        )
        source_match = (
            receipt["source_sha256"] == row["source_sha256"]
            and receipt["source_bytes"] == row["source_bytes"]
        )
        origin_valid, defaults = _validate_fact_origins(receipt)
        inspection = inspect_raw(source)
        comparison = compare_dng_receipt_to_libraw(receipt, inspection)
        output_rows.append(
            {
                "comparison": comparison,
                "defaulted_facts": defaults,
                "manifest_match": manifest_match,
                "origin_valid": origin_valid,
                "raw_kind": receipt["raw_kind"],
                "receipt_body_sha256": receipt["receipt_body_sha256"],
                "source_id": row["source_id"],
                "source_match": source_match,
            }
        )

    output_rows.sort(key=lambda value: value["source_id"])
    gates_config = config["gates"]
    per_row_gates: dict[str, dict[str, bool]] = {}
    for row in output_rows:
        comparison = row["comparison"]
        per_row_gates[row["source_id"]] = {
            "black_level": comparison["black_level"]["maximum_error_codes"]
            <= gates_config["maximum_black_level_error_codes"],
            "camera_white_balance": comparison["camera_white_balance"][
                "maximum_relative_error"
            ]
            <= gates_config["maximum_camera_wb_relative_error"],
            "fact_origins": row["origin_valid"],
            "manifest_binding": row["manifest_match"],
            "raw_geometry": comparison["raw_geometry"]["match"],
            "source_hash": row["source_match"],
            "visible_geometry_explained": comparison["visible_geometry"][
                "classification"
            ]
            != "unexplained",
            "warnings": comparison["warning_count"]
            <= gates_config["maximum_metadata_warnings"],
            "white_level": comparison["white_level"]["error_codes"]
            <= gates_config["maximum_white_level_error_codes"],
        }

    default_rows = {row["source_id"] for row in output_rows if row["defaulted_facts"]}
    metrics = {
        "default_fact_counts": {
            name: sum(name in row["defaulted_facts"] for row in output_rows)
            for name in sorted(_ALLOWED_DEFAULTS)
        },
        "default_row_count": len(default_rows),
        "maximum_black_level_error_codes": max(
            row["comparison"]["black_level"]["maximum_error_codes"]
            for row in output_rows
        ),
        "maximum_camera_wb_relative_error": max(
            row["comparison"]["camera_white_balance"]["maximum_relative_error"]
            for row in output_rows
        ),
        "maximum_white_level_error_codes": max(
            row["comparison"]["white_level"]["error_codes"] for row in output_rows
        ),
        "pixel_or_rgb_decode_calls": 0,
        "raw_geometry_match_rate": sum(
            row["comparison"]["raw_geometry"]["match"] for row in output_rows
        )
        / len(output_rows),
        "row_count": len(output_rows),
        "visible_geometry_class_counts": {
            name: sum(
                row["comparison"]["visible_geometry"]["classification"] == name
                for row in output_rows
            )
            for name in (
                "raw_and_default_crop",
                "raw_ifd",
                "default_crop",
                "unexplained",
            )
        },
        "warning_count": sum(row["comparison"]["warning_count"] for row in output_rows),
    }
    for key in (
        "maximum_black_level_error_codes",
        "maximum_camera_wb_relative_error",
        "maximum_white_level_error_codes",
        "raw_geometry_match_rate",
    ):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f"non-finite aggregate metric: {key}")

    gates = {
        "all_per_row": all(all(row.values()) for row in per_row_gates.values()),
        "known_default_rows": set(gates_config["required_known_default_rows"])
        <= default_rows,
        "minimum_default_rows": len(default_rows)
        >= gates_config["minimum_standard_default_rows"],
        "pixel_or_rgb_decode": metrics["pixel_or_rgb_decode_calls"]
        <= gates_config["maximum_pixel_or_rgb_decode_calls"],
        "row_count": metrics["row_count"] == gates_config["required_row_count"],
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "metrics": metrics,
        "node": config["node"],
        "per_row_gates": per_row_gates,
        "rows": output_rows,
        "runtime": {
            "libraw_version": list(rawpy.libraw_version),
            "rawpy_version": rawpy.__version__,
        },
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_DNG_STANDARD_DEFAULT_CONFORMANCE"
        if all(gates.values())
        else "FAIL_CLOSED_DNG_STANDARD_DEFAULT_CONFORMANCE",
    }
    scientific["stable_evidence_id"] = hashlib.sha256(
        canonical_json_bytes(scientific)
    ).hexdigest()
    return scientific


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p93_dng_standard_default_conformance_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    report = run(config_path, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
                "stable_evidence_id": report["stable_evidence_id"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
