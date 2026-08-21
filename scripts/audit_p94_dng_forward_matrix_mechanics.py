#!/usr/bin/env python3
"""Run the frozen P94 metadata-only DNG ForwardMatrix mechanics audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_forward_matrix import (
    PCS_XYZ,
    DngForwardMatrixError,
    build_dual_illuminant_camera_to_pcs,
    normalize_forward_matrix,
)

SCHEMA = "neuro_film.p94_dng_forward_matrix_mechanics_result.v1"
_TAGS = {
    "color_matrix1": 50721,
    "color_matrix2": 50722,
    "camera_calibration1": 50723,
    "camera_calibration2": 50724,
    "analog_balance": 50727,
    "as_shot_neutral": 50728,
    "calibration_illuminant1": 50778,
    "calibration_illuminant2": 50779,
    "camera_calibration_signature": 50931,
    "profile_calibration_signature": 50932,
    "forward_matrix1": 50964,
    "forward_matrix2": 50965,
}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bindings(bindings: dict[str, Any]) -> None:
    path_pairs = (
        ("contract_path", "contract_sha256"),
        ("implementation_path", "implementation_sha256"),
        ("runner_path", "runner_sha256"),
        ("standard_specification_path", "standard_specification_sha256"),
        ("sdk_archive_path", "sdk_archive_sha256"),
    )
    for path_key, digest_key in path_pairs:
        path = Path(bindings[path_key])
        if not path.is_file() or _sha256(path) != bindings[digest_key]:
            raise DngForwardMatrixError(f"binding mismatch: {path_key}")
    with zipfile.ZipFile(bindings["sdk_archive_path"]) as archive:
        for name, expected in bindings["sdk_member_sha256"].items():
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise DngForwardMatrixError(f"SDK member binding mismatch: {name}")


def _rational_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size % 2:
        raise DngForwardMatrixError(f"{name} is not an interleaved rational array")
    numerator = raw[0::2].astype(np.float64)
    denominator = raw[1::2].astype(np.float64)
    if np.any(denominator == 0.0):
        raise DngForwardMatrixError(f"{name} has a zero denominator")
    result = numerator / denominator
    if not np.all(np.isfinite(result)):
        raise DngForwardMatrixError(f"{name} is non-finite")
    return result


def _read_tags(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with tifffile.TiffFile(path) as document:
        page = document.pages[0]
        tags = page.tags
        missing = [
            name
            for name in (
                "color_matrix1",
                "color_matrix2",
                "as_shot_neutral",
                "calibration_illuminant1",
                "calibration_illuminant2",
                "forward_matrix1",
                "forward_matrix2",
            )
            if _TAGS[name] not in tags
        ]
        if missing:
            raise DngForwardMatrixError(f"missing required tags: {', '.join(missing)}")
        signatures = {}
        for name in ("camera_calibration_signature", "profile_calibration_signature"):
            code = _TAGS[name]
            signatures[name] = "" if code not in tags else str(tags[code].value)
        if (
            signatures["camera_calibration_signature"]
            != signatures["profile_calibration_signature"]
        ):
            raise DngForwardMatrixError("calibration signatures do not match")

        values: dict[str, Any] = {}
        provenance: dict[str, str] = {}
        for name in (
            "color_matrix1",
            "color_matrix2",
            "as_shot_neutral",
            "forward_matrix1",
            "forward_matrix2",
        ):
            values[name] = _rational_array(tags[_TAGS[name]].value, name=name)
            provenance[name] = "explicit_ifd_tag"
        for name in ("calibration_illuminant1", "calibration_illuminant2"):
            values[name] = int(tags[_TAGS[name]].value)
            provenance[name] = "explicit_ifd_tag"
        for name in ("camera_calibration1", "camera_calibration2"):
            code = _TAGS[name]
            if code in tags:
                values[name] = _rational_array(tags[code].value, name=name)
                provenance[name] = "explicit_ifd_tag"
            else:
                values[name] = None
                provenance[name] = "dng_standard_identity_default"
        if _TAGS["analog_balance"] in tags:
            values["analog_balance"] = _rational_array(
                tags[_TAGS["analog_balance"]].value, name="analog_balance"
            )
            provenance["analog_balance"] = "explicit_ifd_tag"
        else:
            values["analog_balance"] = None
            provenance["analog_balance"] = "dng_standard_ones_default"

    for name in (
        "color_matrix1",
        "color_matrix2",
        "camera_calibration1",
        "camera_calibration2",
        "forward_matrix1",
        "forward_matrix2",
    ):
        if values[name] is not None:
            values[name] = values[name].reshape(3, 3)
    return values, provenance


def _row(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["logical_path"])
    if (
        path.stat().st_size != row["source_bytes"]
        or _sha256(path) != row["source_sha256"]
    ):
        raise DngForwardMatrixError(f"source identity mismatch: {row['source_id']}")
    values, provenance = _read_tags(path)
    result = build_dual_illuminant_camera_to_pcs(**values)
    normalized = (
        normalize_forward_matrix(values["forward_matrix1"]),
        normalize_forward_matrix(values["forward_matrix2"]),
    )
    forward_white_error = max(
        float(np.max(np.abs(matrix @ np.ones(3) - PCS_XYZ))) for matrix in normalized
    )
    camera_white_error = float(
        np.max(np.abs(result.camera_to_pcs @ result.camera_white - PCS_XYZ))
    )
    probes = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.25, 0.5, 0.5],
            [0.5, 0.75, 0.5],
            [0.5, 0.25, 0.5],
            [0.5, 0.5, 0.75],
            [0.5, 0.5, 0.25],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(result.camera_to_pcs)
    roundtrip = (inverse @ (result.camera_to_pcs @ probes.T)).T
    return {
        "camera_make": row["camera_make"],
        "camera_to_pcs": result.camera_to_pcs.tolist(),
        "camera_to_pcs_determinant": float(np.linalg.det(result.camera_to_pcs)),
        "camera_white": result.camera_white.tolist(),
        "camera_white_to_pcs_max_abs_error": camera_white_error,
        "forward_one_to_pcs_max_abs_error": forward_white_error,
        "interpolation_weight_first": result.interpolation_weight_first,
        "inverse_roundtrip_max_abs_error": float(np.max(np.abs(roundtrip - probes))),
        "logical_path": row["logical_path"],
        "metadata_provenance": provenance,
        "neutral_iterations": result.neutral_iterations,
        "neutral_xy": list(result.neutral_xy),
        "source_bytes": row["source_bytes"],
        "source_id": row["source_id"],
        "source_sha256": row["source_sha256"],
    }


def run(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_bindings(config["bindings"])
    rows = list(config["rows"])
    if order == "reverse":
        rows.reverse()
    scientific_rows = sorted(
        (_row(row) for row in rows), key=lambda item: item["source_id"]
    )
    gates = config["gates"]
    metrics = {
        "distinct_camera_makes": len({row["camera_make"] for row in scientific_rows}),
        "maximum_camera_white_to_pcs_error": max(
            row["camera_white_to_pcs_max_abs_error"] for row in scientific_rows
        ),
        "maximum_forward_one_to_pcs_error": max(
            row["forward_one_to_pcs_max_abs_error"] for row in scientific_rows
        ),
        "maximum_inverse_roundtrip_error": max(
            row["inverse_roundtrip_max_abs_error"] for row in scientific_rows
        ),
        "maximum_neutral_iterations": max(
            row["neutral_iterations"] for row in scientific_rows
        ),
        "raster_sample_rgb_decode_calls": 0,
        "row_count": len(scientific_rows),
    }
    gate_results = {
        "camera_make_count": metrics["distinct_camera_makes"]
        == gates["required_camera_makes"],
        "camera_white_to_pcs": metrics["maximum_camera_white_to_pcs_error"]
        <= gates["maximum_camera_white_to_pcs_error"],
        "finite_nonsingular": all(
            np.isfinite(row["camera_to_pcs_determinant"])
            and row["camera_to_pcs_determinant"] != 0.0
            for row in scientific_rows
        ),
        "forward_one_to_pcs": metrics["maximum_forward_one_to_pcs_error"]
        <= gates["maximum_forward_one_to_pcs_error"],
        "inverse_roundtrip": metrics["maximum_inverse_roundtrip_error"]
        <= gates["maximum_inverse_roundtrip_error"],
        "neutral_iterations": metrics["maximum_neutral_iterations"]
        <= gates["maximum_neutral_iterations"],
        "row_count": metrics["row_count"] == gates["required_row_count"],
        "zero_raster_decode": metrics["raster_sample_rgb_decode_calls"] == 0,
    }
    scientific = {
        "bindings": config["bindings"],
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_DNG_FORWARD_MATRIX_MECHANICS"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_FORWARD_MATRIX_MECHANICS",
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": metrics,
        "rows": scientific_rows,
        "schema": SCHEMA,
    }
    stable = hashlib.sha256(_canonical_bytes(scientific)).hexdigest()
    return {**scientific, "stable_identity_sha256": stable}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    report = run(args.config, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    print(report["stable_identity_sha256"])
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
