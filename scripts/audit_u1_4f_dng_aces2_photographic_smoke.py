#!/usr/bin/env python3
"""Formal U1.4F four-DNG ACES 2 photographic runtime smoke."""

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

from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.ocio_aces2_output import apply_working_image_aces2_output
from src.preprocess.raw_decode import load_raw_working_image

REPORT_SCHEMA = "neuro_film.u1_4f_dng_aces2_photographic_smoke_result.v1"
TARGETS = ("sdr_rec709", "hdr_rec2020_pq")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_bindings(config: dict[str, Any]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for key in ("contract", "adapter", "runner", "source_config", "adapter_evidence"):
        relative = config["bindings"][f"{key}_path"]
        actual = _sha256_file(ROOT / relative)
        if actual != config["bindings"][f"{key}_sha256"]:
            raise ValueError(f"binding mismatch for {key}")
        checked[f"{key}_path"] = relative
        checked[f"{key}_sha256"] = actual
    return checked


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _validate_bindings(config)
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    results: list[dict[str, Any]] = []
    for row in rows:
        source = ROOT / row["logical_path"]
        source_bytes = source.stat().st_size
        source_sha256 = _sha256_file(source)
        if source_bytes != row["source_bytes"] or source_sha256 != row["source_sha256"]:
            raise ValueError(f"source identity drift: {row['source_id']}")
        working = load_raw_working_image(source)
        input_sha256 = _array_sha256(working.pixels)
        targets: dict[str, dict[str, Any]] = {}
        for target in TARGETS:
            output = apply_working_image_aces2_output(working, target)
            targets[target] = {
                "finite": bool(np.isfinite(output).all()),
                "maximum": float(np.max(output)),
                "minimum": float(np.min(output)),
                "output_sha256": _array_sha256(output),
                "outside_unit_fraction": float(
                    np.count_nonzero((output < 0.0) | (output > 1.0)) / output.size
                ),
                "shape": list(output.shape),
            }
            del output
        results.append(
            {
                "decoded_input_sha256": input_sha256,
                "decoded_shape": list(working.pixels.shape),
                "input_unchanged": _array_sha256(working.pixels) == input_sha256,
                "source_bytes": source_bytes,
                "source_id": row["source_id"],
                "source_sha256": source_sha256,
                "targets": targets,
                "warning_codes": [warning.code for warning in working.warnings],
                "working_space": working.working_space,
                "transfer_state": working.transfer_state,
            }
        )
        del working
    results.sort(key=lambda item: item["source_id"])
    gates = {
        "all_finite": all(
            target["finite"]
            for row in results
            for target in row["targets"].values()
        ),
        "all_input_unchanged": all(row["input_unchanged"] for row in results),
        "all_output_shapes_match": all(
            target["shape"] == row["decoded_shape"]
            for row in results
            for target in row["targets"].values()
        ),
        "all_source_identities_match": len(results) == config["gates"]["required_rows"],
        "all_targets_distinct": all(
            row["targets"]["sdr_rec709"]["output_sha256"]
            != row["targets"]["hdr_rec2020_pq"]["output_sha256"]
            for row in results
        ),
        "all_working_boundaries_exact": all(
            row["working_space"] == "linear_srgb"
            and row["transfer_state"] == "scene_linear"
            for row in results
        ),
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "metrics": {
            "rows": results,
            "source_count": len(results),
        },
        "node": config["node"],
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_FOUR_DNG_ACES2_PHOTOGRAPHIC_SMOKE"
        if all(gates.values())
        else "FAIL_CLOSED_FOUR_DNG_ACES2_PHOTOGRAPHIC_SMOKE",
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
        default=Path("configs/u1_4f_dng_aces2_photographic_smoke_v1.json"),
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
