#!/usr/bin/env python3
"""Run the frozen P97 DNG PCS-to-Rec.2020 composition audit."""

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

from src.eval.dng_camera_to_pcs_native_conformance import build_probes
from src.preprocess.prophoto_icc import (
    ProPhotoICCError,
    d50_xyz_to_linear_rec2020,
    decode_prophoto_rgb16_to_linear_rec2020,
)

SCHEMA = "neuro_film.p97_dng_pcs_rec2020_composition_result.v1"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bindings(bindings: dict[str, Any]) -> None:
    for binding in bindings.values():
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise RuntimeError(f"P97 binding mismatch: {path}")


def run(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _verify_bindings(config["bindings"])
    profile = Path(config["bindings"]["romm_profile"]["path"]).read_bytes()
    codes = np.arange(65536, dtype=np.uint16)
    encoded = np.stack(
        (codes, codes[::-1], np.bitwise_xor(codes, 0x5A5A)), axis=1
    ).reshape(1, -1, 3)
    regression = decode_prophoto_rgb16_to_linear_rec2020(encoded, profile)
    regression_sha256 = hashlib.sha256(regression.tobytes()).hexdigest()

    parent = json.loads(
        Path(config["bindings"]["p94_report"]["path"]).read_text(encoding="utf-8")
    )
    rows = list(parent["rows"])
    rows.sort(key=lambda item: item["source_id"], reverse=order == "reverse")
    probes = build_probes()
    basis = d50_xyz_to_linear_rec2020(np.eye(3, dtype=np.float64))
    scientific_rows = []
    for row in rows:
        matrix = np.asarray(row["camera_to_pcs"], dtype=np.float64)
        staged = d50_xyz_to_linear_rec2020(probes @ matrix.T)
        direct = probes @ (basis.T @ matrix).T
        white = d50_xyz_to_linear_rec2020(
            matrix @ np.asarray(row["camera_white"], dtype=np.float64)
        )
        scientific_rows.append(
            {
                "camera_make": row["camera_make"],
                "composition_max_abs_error": float(np.max(np.abs(staged - direct))),
                "output_sha256": hashlib.sha256(staged.tobytes()).hexdigest(),
                "source_id": row["source_id"],
                "white_to_unit_max_abs_error": float(np.max(np.abs(white - 1.0))),
            }
        )
    scientific_rows.sort(key=lambda item: item["source_id"])
    invalid_rejections = 0
    for invalid in (
        np.zeros((2, 4), dtype=np.float64),
        np.asarray([[0.0, 0.0, np.nan]], dtype=np.float64),
    ):
        try:
            d50_xyz_to_linear_rec2020(invalid)
        except (ValueError, ProPhotoICCError):
            invalid_rejections += 1
    metrics = {
        "composition_max_abs_error": max(
            row["composition_max_abs_error"] for row in scientific_rows
        ),
        "invalid_rejections": invalid_rejections,
        "profile_count": len(scientific_rows),
        "raster_sample_rgb_reads": 0,
        "regression_output_sha256": regression_sha256,
        "transformed_triplet_count": len(scientific_rows) * len(probes),
        "white_to_unit_max_abs_error": max(
            row["white_to_unit_max_abs_error"] for row in scientific_rows
        ),
    }
    gates = config["gates"]
    gate_results = {
        "composition_error": metrics["composition_max_abs_error"]
        <= gates["maximum_composition_absolute_error"],
        "invalid_inputs": metrics["invalid_rejections"]
        == gates["required_invalid_rejections"],
        "profile_count": metrics["profile_count"] == gates["required_profiles"],
        "regression_hash": regression_sha256 == gates["required_regression_sha256"],
        "triplet_count": metrics["transformed_triplet_count"]
        == gates["required_transformed_triplets"],
        "white_mechanics": metrics["white_to_unit_max_abs_error"]
        <= gates["maximum_white_to_unit_absolute_error"],
        "zero_raster_reads": metrics["raster_sample_rgb_reads"] == 0,
    }
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_DNG_PCS_REC2020_COMPOSITION"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_PCS_REC2020_COMPOSITION",
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": metrics,
        "rows": scientific_rows,
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError("P97 formal report already exists")
    report = run(args.config, args.order)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(_canonical_bytes(report))
    print(report["stable_identity_sha256"])
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
