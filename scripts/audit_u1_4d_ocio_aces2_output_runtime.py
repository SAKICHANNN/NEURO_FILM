#!/usr/bin/env python3
"""Formal U1.4D official OCIO ACES 2 CPU runtime audit."""

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
from src.preprocess.ocio_aces2_output import (
    CONFIG_CACHE_ID,
    CONFIG_URI,
    OCIO_VERSION,
    SOURCE_SPACE,
    apply_aces2_output_packed,
    apply_aces2_output_scalar,
    build_aces2_numeric_fixture,
    load_aces2_config,
    target_display_view,
)

REPORT_SCHEMA = "neuro_film.u1_4d_ocio_aces2_output_runtime_result.v1"
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
    for key in ("contract", "implementation", "runner"):
        path = ROOT / config["bindings"][f"{key}_path"]
        actual = _sha256_file(path)
        if actual != config["bindings"][f"{key}_sha256"]:
            raise ValueError(f"binding mismatch for {key}")
        checked[f"{key}_path"] = config["bindings"][f"{key}_path"]
        checked[f"{key}_sha256"] = actual
    return checked


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _validate_bindings(config)
    ocio_config = load_aces2_config()
    fixture = build_aces2_numeric_fixture()
    fixture_sha = _array_sha256(fixture)
    ordered = np.ascontiguousarray(fixture[::-1] if reverse else fixture)
    ordered_before = _array_sha256(ordered)

    outputs: dict[str, np.ndarray] = {}
    target_rows: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        scalar = apply_aces2_output_scalar(ordered, target)
        packed = apply_aces2_output_packed(ordered, target)
        if reverse:
            scalar = np.ascontiguousarray(scalar[::-1])
            packed = np.ascontiguousarray(packed[::-1])
        outputs[target] = packed
        neutral = packed[-257:]
        neutral_mean = neutral.mean(axis=1, dtype=np.float64)
        display, view = target_display_view(target)
        target_rows[target] = {
            "display": display,
            "finite": bool(np.isfinite(scalar).all() and np.isfinite(packed).all()),
            "maximum_neutral_channel_spread": float(
                np.max(np.max(neutral, axis=1) - np.min(neutral, axis=1))
            ),
            "maximum_scalar_packed_absolute_error": float(np.max(np.abs(scalar - packed))),
            "minimum_neutral_mean_step": float(np.min(np.diff(neutral_mean))),
            "packed_output_sha256": _array_sha256(packed),
            "scalar_output_sha256": _array_sha256(scalar),
            "view": view,
        }

    gates_config = config["gates"]
    gates = {
        "config_cache_id": ocio_config.getCacheID() == CONFIG_CACHE_ID,
        "finite": all(row["finite"] for row in target_rows.values()),
        "input_unchanged": _array_sha256(ordered) == ordered_before,
        "materially_distinct_outputs": float(
            np.max(np.abs(outputs["sdr_rec709"] - outputs["hdr_rec2020_pq"]))
        )
        >= gates_config["minimum_sdr_hdr_maximum_difference"],
        "neutral_monotonic": all(
            row["minimum_neutral_mean_step"] >= -gates_config["neutral_monotonic_tolerance"]
            for row in target_rows.values()
        ),
        "neutral_spread": all(
            row["maximum_neutral_channel_spread"]
            <= gates_config["maximum_neutral_channel_spread"]
            for row in target_rows.values()
        ),
        "scalar_packed": all(
            row["maximum_scalar_packed_absolute_error"]
            <= gates_config["maximum_scalar_packed_absolute_error"]
            for row in target_rows.values()
        ),
    }
    metrics = {
        "fixture_rows": int(fixture.shape[0]),
        "fixture_sha256": fixture_sha,
        "maximum_sdr_hdr_absolute_difference": float(
            np.max(np.abs(outputs["sdr_rec709"] - outputs["hdr_rec2020_pq"]))
        ),
        "targets": target_rows,
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "metrics": metrics,
        "node": config["node"],
        "runtime": {
            "config_cache_id": CONFIG_CACHE_ID,
            "config_uri": CONFIG_URI,
            "ocio_version": OCIO_VERSION,
            "source_space": SOURCE_SPACE,
        },
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_OCIO_ACES2_OUTPUT_RUNTIME"
        if all(gates.values())
        else "FAIL_CLOSED_OCIO_ACES2_OUTPUT_RUNTIME",
    }
    scientific["stable_evidence_id"] = hashlib.sha256(canonical_json_bytes(scientific)).hexdigest()
    return scientific


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/u1_4d_ocio_aces2_output_runtime_v1.json"),
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
