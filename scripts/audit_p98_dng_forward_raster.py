#!/usr/bin/env python3
"""Audit the frozen P98 opt-in DNG ForwardMatrix raster path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rawpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_forward_matrix import build_dual_illuminant_camera_to_pcs
from src.preprocess.dng_forward_raster import (
    DngForwardRasterError,
    _apply_camera_to_rec2020,
    _decode_camera_linear_dng,
    _read_profile_tags,
    load_dng_forward_working_image,
)
from src.preprocess.prophoto_icc import d50_xyz_to_linear_rec2020

SCHEMA = "neuro_film.p98_dng_forward_raster_result.v1"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _verify_file(
    path: Path, expected_sha256: str, expected_bytes: int | None = None
) -> None:
    if not path.is_file():
        raise DngForwardRasterError(f"bound file is absent: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise DngForwardRasterError(f"bound file size mismatch: {path}")
    if _sha256(path) != expected_sha256:
        raise DngForwardRasterError(f"bound file hash mismatch: {path}")


def _decode_generic_rec2020(path: Path) -> np.ndarray:
    try:
        with rawpy.imread(str(path)) as raw:
            value = raw.postprocess(
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_bright=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.Rec2020,
                gamma=(1.0, 1.0),
                user_flip=None,
            )
    except Exception as exc:
        raise DngForwardRasterError(
            f"LibRaw Rec.2020 diagnostic decode failed: {exc}"
        ) from exc
    if value.dtype != np.uint16 or value.ndim != 3 or value.shape[2] != 3:
        raise DngForwardRasterError("LibRaw Rec.2020 diagnostic is not uint16 HxWx3")
    return value.astype(np.float32) / np.float32(65535.0)


def _oracle_metrics(
    camera: np.ndarray,
    camera_to_pcs: np.ndarray,
    working_pixels: np.ndarray,
) -> dict[str, Any]:
    camera_frozen = camera.copy()
    direct_matrix = (
        d50_xyz_to_linear_rec2020(np.eye(3, dtype=np.float64)).T @ camera_to_pcs
    )
    oracle_digest = hashlib.sha256()
    composition_max = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    outside = 0
    component_count = 0
    for start in range(0, camera.shape[0], 128):
        stop = min(start + 128, camera.shape[0])
        normalized = camera[start:stop].astype(np.float64) / 65535.0
        staged = d50_xyz_to_linear_rec2020(normalized @ camera_to_pcs.T)
        direct = normalized @ direct_matrix.T
        composition_max = max(composition_max, float(np.max(np.abs(staged - direct))))
        minimum = min(minimum, float(np.min(staged)))
        maximum = max(maximum, float(np.max(staged)))
        outside += int(np.count_nonzero((staged < 0.0) | (staged > 1.0)))
        component_count += int(staged.size)
        oracle_digest.update(staged.astype(np.float32).tobytes(order="C"))
    if not np.array_equal(camera, camera_frozen):
        raise DngForwardRasterError("camera raster changed during oracle conversion")
    oracle_sha = oracle_digest.hexdigest()
    return {
        "camera_input_unchanged": True,
        "composition_max_abs_error": composition_max,
        "maximum": maximum,
        "minimum": minimum,
        "oracle_float32_sha256": oracle_sha,
        "outside_unit_fraction": outside / component_count,
        "working_float32_matches_oracle": _array_sha256(working_pixels) == oracle_sha,
    }


def _row(row: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / row["logical_path"]
    _verify_file(path, row["source_sha256"], row["source_bytes"])
    before_sha = _sha256(path)
    camera = _decode_camera_linear_dng(path)
    camera_sha = _array_sha256(camera)
    values = _read_profile_tags(path)
    camera_to_pcs = build_dual_illuminant_camera_to_pcs(**values).camera_to_pcs
    working = load_dng_forward_working_image(
        path,
        expected_source_bytes=row["source_bytes"],
        expected_source_sha256=row["source_sha256"],
    )
    oracle = _oracle_metrics(camera, camera_to_pcs, working.pixels)
    generic = _decode_generic_rec2020(path)
    if generic.shape != working.pixels.shape:
        raise DngForwardRasterError("generic diagnostic shape differs from P98")
    difference = working.pixels.astype(np.float64) - generic.astype(np.float64)
    if _sha256(path) != before_sha:
        raise DngForwardRasterError("source bytes changed during P98 audit")
    return {
        "camera_make": row["camera_make"],
        "camera_raster": {
            "dtype": str(camera.dtype),
            "maximum": int(np.max(camera)),
            "minimum": int(np.min(camera)),
            "shape": list(camera.shape),
            "sha256": camera_sha,
            "unit_range": int(np.min(camera)) >= 0 and int(np.max(camera)) <= 65535,
        },
        "generic_rec2020_diagnostic": {
            "median_abs_difference": float(np.median(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "sha256": _array_sha256(generic),
        },
        "oracle": oracle,
        "source_bytes": row["source_bytes"],
        "source_id": row["source_id"],
        "source_sha256": row["source_sha256"],
        "source_unchanged": True,
        "working": {
            "dtype": str(working.pixels.dtype),
            "finite": bool(np.all(np.isfinite(working.pixels))),
            "shape": list(working.pixels.shape),
            "sha256": _array_sha256(working.pixels),
            "transfer_state": working.transfer_state,
            "warning_codes": [warning.code for warning in working.warnings],
            "working_space": working.working_space,
        },
    }


def _invalid_rejection_count() -> int:
    cases = (
        (np.zeros((1, 1, 3), np.float32), np.eye(3), 1),
        (np.zeros((1, 1, 3), np.uint16), np.eye(2), 1),
        (np.zeros((1, 1, 3), np.uint16), np.full((3, 3), np.nan), 1),
        (np.zeros((1, 1, 3), np.uint16), np.eye(3), 0),
    )
    rejected = 0
    for camera, matrix, row_block in cases:
        try:
            _apply_camera_to_rec2020(camera, matrix, row_block=row_block)
        except DngForwardRasterError:
            rejected += 1
    return rejected


def run(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    for path_key, digest_key in (
        ("contract_path", "contract_sha256"),
        ("p94_evidence_path", "p94_evidence_sha256"),
        ("p97_evidence_path", "p97_evidence_sha256"),
        ("implementation_path", "implementation_sha256"),
        ("runner_path", "runner_sha256"),
    ):
        _verify_file(ROOT / bindings[path_key], bindings[digest_key])
    runtime = config["runtime"]
    runtime_exact = (
        rawpy.__version__ == runtime["rawpy_version"]
        and list(rawpy.libraw_version) == runtime["libraw_version"]
    )
    rows = list(config["rows"])
    if order == "reverse":
        rows.reverse()
    scientific_rows = sorted(
        (_row(row) for row in rows), key=lambda item: item["source_id"]
    )
    invalid_rejections = _invalid_rejection_count()
    metrics = {
        "distinct_camera_makes": len({row["camera_make"] for row in scientific_rows}),
        "invalid_rejections": invalid_rejections,
        "maximum_absolute_output": max(
            max(abs(row["oracle"]["minimum"]), abs(row["oracle"]["maximum"]))
            for row in scientific_rows
        ),
        "maximum_composition_error": max(
            row["oracle"]["composition_max_abs_error"] for row in scientific_rows
        ),
        "maximum_outside_unit_fraction": max(
            row["oracle"]["outside_unit_fraction"] for row in scientific_rows
        ),
        "row_count": len(scientific_rows),
    }
    gates = config["gates"]
    gate_results = {
        "camera_inputs_unchanged": all(
            row["oracle"]["camera_input_unchanged"] for row in scientific_rows
        ),
        "camera_make_count": metrics["distinct_camera_makes"]
        == gates["required_camera_makes"],
        "camera_raster_unit_range": all(
            row["camera_raster"]["unit_range"] for row in scientific_rows
        ),
        "composition_error": metrics["maximum_composition_error"]
        <= gates["maximum_composition_error"],
        "finite_working_images": all(
            row["working"]["finite"] for row in scientific_rows
        ),
        "float32_oracle_bytes": all(
            row["oracle"]["working_float32_matches_oracle"] for row in scientific_rows
        ),
        "invalid_inputs": invalid_rejections == 4,
        "output_absolute_range": metrics["maximum_absolute_output"]
        <= gates["maximum_absolute_output"],
        "output_outside_unit_fraction": metrics["maximum_outside_unit_fraction"]
        <= gates["maximum_outside_unit_fraction"],
        "row_count": metrics["row_count"] == gates["required_rows"],
        "runtime_exact": runtime_exact,
        "source_hashes_unchanged": all(
            row["source_unchanged"] for row in scientific_rows
        ),
        "working_contract": all(
            row["working"]["dtype"] == "float32"
            and row["working"]["working_space"] == "linear_rec2020"
            and row["working"]["transfer_state"] == "scene_linear"
            and row["working"]["shape"] == row["camera_raster"]["shape"]
            for row in scientific_rows
        ),
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_DNG_FORWARD_RASTER"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_FORWARD_RASTER",
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": metrics,
        "rows": scientific_rows,
        "runtime": {
            "libraw_version": list(rawpy.libraw_version),
            "rawpy_version": rawpy.__version__,
        },
        "schema": SCHEMA,
    }
    return {
        **scientific,
        "stable_identity_sha256": hashlib.sha256(
            _canonical_bytes(scientific)
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    report = run(config, args.order)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(report)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "output": str(output),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
                "stable_identity_sha256": report["stable_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
