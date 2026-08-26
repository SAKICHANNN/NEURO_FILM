#!/usr/bin/env python3
"""Audit the frozen P100 P98-to-official-ACES-2 numerical composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_forward_raster import load_dng_forward_working_image
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.ocio_aces2_output import (
    CONFIG_CACHE_ID,
    SOURCE_SPACE,
    apply_working_image_aces2_output,
    load_aces2_config,
    target_display_view,
)
from src.preprocess.types import WorkingImage

SCHEMA = "neuro_film.p100_dng_forward_aces2_output_result.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_bindings(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    checked: dict[str, dict[str, str]] = {}
    bindings = config["bindings"]
    for name in (
        "contract",
        "p98_config",
        "p98_evidence",
        "p98_implementation",
        "aces_adapter",
        "aces_evidence",
        "runner",
        "test",
    ):
        path = ROOT / bindings[f"{name}_path"]
        actual = _sha256_file(path)
        expected = bindings[f"{name}_sha256"]
        if actual != expected:
            raise ValueError(f"P100 binding mismatch: {name}")
        checked[name] = {"path": bindings[f"{name}_path"], "sha256": actual}
    return checked


def _direct_official_output(working: WorkingImage, target: str) -> np.ndarray:
    """Independently invoke the two official processors used by the adapter."""

    import PyOpenColorIO as ocio

    if working.working_space != "linear_rec2020":
        raise ValueError("P100 direct path requires linear_rec2020")
    if working.transfer_state != "scene_linear":
        raise ValueError("P100 direct path requires scene_linear")
    config = load_aces2_config()
    pixels = np.ascontiguousarray(working.pixels.reshape(-1, 3).copy())
    source_processor = config.getProcessor(
        "Linear Rec.2020", SOURCE_SPACE
    ).getDefaultCPUProcessor()
    source_processor.apply(ocio.PackedImageDesc(pixels, pixels.shape[0], 1, 3))
    display, view = target_display_view(target)  # type: ignore[arg-type]
    transform = ocio.DisplayViewTransform(
        src=SOURCE_SPACE,
        display=display,
        view=view,
    )
    output_processor = config.getProcessor(transform).getDefaultCPUProcessor()
    output_processor.apply(ocio.PackedImageDesc(pixels, pixels.shape[0], 1, 3))
    if not np.isfinite(pixels).all():
        raise ValueError("P100 direct official output is non-finite")
    return np.ascontiguousarray(pixels.reshape(working.pixels.shape))


def _row(row: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    source = ROOT / row["logical_path"]
    before = _sha256_file(source)
    if source.stat().st_size != row["source_bytes"] or before != row["source_sha256"]:
        raise ValueError(f"P100 source identity mismatch: {row['source_id']}")
    working = load_dng_forward_working_image(
        source,
        expected_source_bytes=row["source_bytes"],
        expected_source_sha256=row["source_sha256"],
    )
    input_sha = _array_sha256(working.pixels)
    output_rows: dict[str, dict[str, Any]] = {}
    for target in targets:
        adapter = apply_working_image_aces2_output(working, target)  # type: ignore[arg-type]
        direct = _direct_official_output(working, target)
        difference = adapter.astype(np.float64) - direct.astype(np.float64)
        output_rows[target] = {
            "adapter_sha256": _array_sha256(adapter),
            "direct_bytes_exact": bool(np.array_equal(adapter, direct)),
            "direct_max_abs_error": float(np.max(np.abs(difference))),
            "dtype": str(adapter.dtype),
            "finite": bool(np.isfinite(adapter).all()),
            "in_unit": bool(np.all((adapter >= 0.0) & (adapter <= 1.0))),
            "maximum": float(np.max(adapter)),
            "minimum": float(np.min(adapter)),
            "shape": list(adapter.shape),
        }
    return {
        "camera_make": row["camera_make"],
        "input_pixels_unchanged": _array_sha256(working.pixels) == input_sha,
        "input_sha256": input_sha,
        "outputs": output_rows,
        "source_bytes": row["source_bytes"],
        "source_id": row["source_id"],
        "source_sha256": before,
        "source_unchanged": _sha256_file(source) == before,
        "transfer_state": working.transfer_state,
        "warning_codes": [warning.code for warning in working.warnings],
        "working_dtype": str(working.pixels.dtype),
        "working_finite": bool(np.isfinite(working.pixels).all()),
        "working_shape": list(working.pixels.shape),
        "working_space": working.working_space,
    }


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _validate_bindings(config)
    p98 = _load_json(ROOT / config["bindings"]["p98_config_path"])
    rows = list(p98["rows"])
    if reverse:
        rows.reverse()
    targets = list(config["targets"])
    records = sorted(
        (_row(row, targets) for row in rows), key=lambda item: item["source_id"]
    )
    all_outputs = [
        output for row in records for output in row["outputs"].values()
    ]
    gates = config["gates"]
    gate_results = {
        "finite_in_unit_outputs": all(
            output["dtype"] == "float32" and output["finite"] and output["in_unit"]
            for output in all_outputs
        ),
        "input_pixels_unchanged": all(
            row["input_pixels_unchanged"] for row in records
        ),
        "official_direct_bytes_exact": all(
            output["direct_bytes_exact"] for output in all_outputs
        ),
        "required_rows": len(records) == gates["required_rows"],
        "required_targets": len(targets) == gates["required_targets"],
        "source_hashes_unchanged": all(row["source_unchanged"] for row in records),
        "target_hashes_distinct": all(
            row["outputs"][targets[0]]["adapter_sha256"]
            != row["outputs"][targets[1]]["adapter_sha256"]
            for row in records
        ),
        "working_contract": all(
            row["working_dtype"] == "float32"
            and row["working_finite"]
            and row["working_space"] == "linear_rec2020"
            and row["transfer_state"] == "scene_linear"
            and all(
                output["shape"] == row["working_shape"]
                for output in row["outputs"].values()
            )
            for row in records
        ),
    }
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": {
            "maximum_direct_abs_error": max(
                output["direct_max_abs_error"] for output in all_outputs
            ),
            "records": records,
            "row_count": len(records),
            "target_count": len(targets),
        },
        "node": config["node"],
        "runtime": {
            "aces_config_cache_id": load_aces2_config().getCacheID(),
            "aces_config_cache_identity_exact": load_aces2_config().getCacheID()
            == CONFIG_CACHE_ID,
        },
        "schema": SCHEMA,
        "status": "PASS_PRIVATE_DNG_FORWARD_ACES2_OUTPUT_CONFORMANCE"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_FORWARD_ACES2_OUTPUT_CONFORMANCE",
    }
    report["stable_evidence_id"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p100_dng_forward_aces2_output_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config, reverse=args.reverse)
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
