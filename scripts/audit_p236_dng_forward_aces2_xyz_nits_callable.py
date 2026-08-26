#!/usr/bin/env python3
"""Audit exact-five P98 DNG to official ACES 2 XYZ-D65-nits callable."""

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

from src.preprocess.dng_forward_aces2_xyz import (
    DISPLAY_REFERENCE_SCALE_NITS,
    OUTPUT_BUILTIN,
    SCENE_TO_REFERENCE_BUILTIN,
    load_dng_forward_aces2_xyz_d65_nits,
    render_acescg_to_xyz_d65_nits,
)
from src.preprocess.dng_forward_raster import load_dng_forward_working_image
from src.preprocess.dng_metadata import canonical_json_bytes

SCHEMA = "neuro-film.p236-dng-forward-aces2-xyz-nits-callable-result.v1"


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


def _validate_bindings(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name in (
        "p98_config",
        "p98_evidence",
        "p98_source",
        "p224_config",
        "p224_evidence",
        "ocio_source",
    ):
        path = ROOT / config["bindings"][f"{name}_path"]
        actual = _sha256_file(path)
        expected = config["bindings"][f"{name}_sha256"]
        if actual != expected:
            raise ValueError(f"P236 binding mismatch: {name}")
        observed[name] = {
            "path": config["bindings"][f"{name}_path"],
            "sha256": actual,
        }
    return observed


def _direct_official(working_pixels: np.ndarray, working_space: str) -> np.ndarray:
    import PyOpenColorIO as ocio

    if working_space != "linear_rec2020":
        raise ValueError("P236 direct chain requires linear_rec2020")
    output = np.ascontiguousarray(working_pixels.reshape(-1, 3).copy())
    config = ocio.Config.CreateFromFile("ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5")
    source = config.getProcessor("Linear Rec.2020", "ACEScg").getDefaultCPUProcessor()
    source.apply(ocio.PackedImageDesc(output, output.shape[0], 1, 3))
    group = ocio.GroupTransform()
    group.appendTransform(ocio.BuiltinTransform(SCENE_TO_REFERENCE_BUILTIN))
    group.appendTransform(ocio.BuiltinTransform(OUTPUT_BUILTIN))
    ocio.Config.CreateRaw().getProcessor(group).getDefaultCPUProcessor().apply(
        ocio.PackedImageDesc(output, output.shape[0], 1, 3)
    )
    output *= np.float32(DISPLAY_REFERENCE_SCALE_NITS)
    return np.ascontiguousarray(output.reshape(working_pixels.shape))


def _row(row: dict[str, Any]) -> dict[str, Any]:
    source = ROOT / row["logical_path"]
    source_before = _sha256_file(source)
    working = load_dng_forward_working_image(
        source,
        expected_source_bytes=row["source_bytes"],
        expected_source_sha256=row["source_sha256"],
    )
    input_before = _array_sha256(working.pixels)
    candidate = load_dng_forward_aces2_xyz_d65_nits(
        source,
        expected_source_bytes=row["source_bytes"],
        expected_source_sha256=row["source_sha256"],
    )
    direct = _direct_official(working.pixels, working.working_space)
    difference = candidate.astype(np.float64) - direct.astype(np.float64)
    return {
        "camera_make": row["camera_make"],
        "candidate_sha256": _array_sha256(candidate),
        "direct_bytes_exact": bool(np.array_equal(candidate, direct)),
        "direct_maximum_absolute_error_nits": float(np.max(np.abs(difference))),
        "dtype": str(candidate.dtype),
        "finite": bool(np.isfinite(candidate).all()),
        "input_pixels_unchanged": _array_sha256(working.pixels) == input_before,
        "maximum_nits": float(np.max(candidate)),
        "minimum_nits": float(np.min(candidate)),
        "owned": not np.shares_memory(candidate, working.pixels),
        "shape": list(candidate.shape),
        "source_id": row["source_id"],
        "source_unchanged": _sha256_file(source) == source_before,
        "working_shape": list(working.pixels.shape),
    }


def execute(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _validate_bindings(config)
    p98 = _load_json(ROOT / config["bindings"]["p98_config_path"])
    rows = list(p98["rows"])
    if reverse:
        rows.reverse()
    records = sorted((_row(row) for row in rows), key=lambda row: row["source_id"])
    probes = np.ascontiguousarray(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.001, 0.001, 0.001],
                [0.18, 0.18, 0.18],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    probe_output = render_acescg_to_xyz_d65_nits(probes)
    neutral_spreads = np.max(probe_output, axis=1) - np.min(probe_output, axis=1)
    gates = config["gates"]
    gate_results = {
        "required_rows": len(records) == gates["required_rows"],
        "direct_official_bytes_exact": all(
            row["direct_bytes_exact"] for row in records
        ),
        "owned_c_contiguous_float32_output": all(
            row["dtype"] == "float32"
            and row["owned"]
            and row["shape"] == row["working_shape"]
            for row in records
        ),
        "finite_nonnegative_output": all(
            row["finite"]
            and row["minimum_nits"] >= -gates["maximum_negative_output_nits"]
            for row in records
        ),
        "input_and_source_unchanged": all(
            row["input_pixels_unchanged"] and row["source_unchanged"] for row in records
        ),
        "black_anchor": float(np.max(np.abs(probe_output[0])))
        <= gates["maximum_black_absolute_nits"],
        "neutral_probes": float(np.max(neutral_spreads))
        <= gates["maximum_neutral_spread_nits"],
    }
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": {
            "black_output_nits": [float(value) for value in probe_output[0]],
            "maximum_direct_absolute_error_nits": max(
                row["direct_maximum_absolute_error_nits"] for row in records
            ),
            "maximum_neutral_spread_nits": float(np.max(neutral_spreads)),
            "records": records,
            "row_count": len(records),
        },
        "node": config["node"],
        "schema": SCHEMA,
        "status": "PASS_PRIVATE_DNG_FORWARD_ACES2_XYZ_NITS_CALLABLE"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_FORWARD_ACES2_XYZ_NITS_CALLABLE",
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
        default=Path("configs/p236_dng_forward_aces2_xyz_nits_callable_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = execute(config, reverse=args.reverse)
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
