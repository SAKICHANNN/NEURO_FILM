#!/usr/bin/env python3
"""Formal U1.6G4K staged physical-colour halation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import composite_layers, physical_halation_layer
from src.filmfx.staged_colour import (
    execute_staged_physical_colour_halation_default,
    materialized_physical_colour_halation_v2_reference,
)

CONFIG = ROOT / "configs/u1_6g4k_staged_physical_colour_halation_v1.json"
CORE = ROOT / "src/filmfx/staged_colour.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _synthetic(case: dict) -> np.ndarray:
    shape = tuple(case["shape"])
    rng = np.random.default_rng(case["seed"])
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = rng.uniform(0.01, 0.30, size=shape).astype(np.float32)
    first = np.exp(
        -((y - shape[0] * 0.35) ** 2 + (x - shape[1] * 0.31) ** 2)
        / max(120.0, shape[0] * 1.15)
    )
    second = np.exp(
        -((y - shape[0] * 0.64) ** 2 + (x - shape[1] * 0.70) ** 2)
        / max(170.0, shape[0] * 1.55)
    )
    if case["kind"] == "skin_guard_and_edge_visibility":
        first_colour = np.array([0.74, 0.36, 0.24], np.float32)
        second_colour = np.array([0.93, 0.87, 0.76], np.float32)
    else:
        first_colour = np.array([0.88, 0.85, 0.82], np.float32)
        second_colour = np.array([0.92, 0.58, 0.25], np.float32)
    base += first[..., None].astype(np.float32) * first_colour
    base += second[..., None].astype(np.float32) * second_colour
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _real(case: dict) -> np.ndarray:
    path = ROOT / case["path"]
    if not path.is_file() or _sha256(path) != case["sha256"]:
        raise RuntimeError(f"source lock failed: {case['case_id']}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)


def _seam_max(difference: np.ndarray, tile_size: int) -> float:
    values = []
    for y in range(tile_size, difference.shape[0], tile_size):
        values.append(float(difference[max(0, y - 1) : min(difference.shape[0], y + 1)].max()))
    for x in range(tile_size, difference.shape[1], tile_size):
        values.append(float(difference[:, max(0, x - 1) : min(difference.shape[1], x + 1)].max()))
    return max(values, default=0.0)


def _metrics(case: dict, base: np.ndarray, config: dict) -> dict:
    gates = config["gates"]
    reference = materialized_physical_colour_halation_v2_reference(
        base, source_row_chunk=config["source_row_chunk"]
    )
    legacy = physical_halation_layer(base)
    reference_output = composite_layers(base, [reference])
    legacy_output = composite_layers(base, [legacy])
    legacy_alpha = np.abs(legacy.alpha - reference.alpha)
    legacy_rgb = np.abs(legacy.rgb - reference.rgb)
    legacy_output_delta = np.abs(legacy_output - reference_output)
    legacy_code_delta = np.abs(
        np.rint(legacy_output * 255).astype(np.int16)
        - np.rint(reference_output * 255).astype(np.int16)
    )
    policies = []
    for tile_size in case["tile_sizes"]:
        staged, metadata = execute_staged_physical_colour_halation_default(
            base,
            tile_size=tile_size,
            source_row_chunk=config["source_row_chunk"],
            coarse_row_chunk=config["coarse_row_chunk"],
        )
        repeated, repeated_metadata = execute_staged_physical_colour_halation_default(
            base,
            tile_size=tile_size,
            source_row_chunk=config["source_row_chunk"],
            coarse_row_chunk=config["coarse_row_chunk"],
        )
        staged_output = composite_layers(base, [staged])
        alpha_delta = np.abs(staged.alpha - reference.alpha)
        rgb_delta = np.abs(staged.rgb - reference.rgb)
        output_delta = np.abs(staged_output - reference_output)
        row = {
            "tile_size": tile_size,
            "alpha_max_error": float(alpha_delta.max()),
            "rgb_max_error": float(rgb_delta.max()),
            "composite_max_error": float(output_delta.max()),
            "seam_max_error": _seam_max(output_delta, tile_size),
            "srgb8_byte_identical": bool(
                np.array_equal(
                    np.rint(staged_output * 255).astype(np.uint8),
                    np.rint(reference_output * 255).astype(np.uint8),
                )
            ),
            "repeat_exact": bool(
                staged.rgb.tobytes() == repeated.rgb.tobytes()
                and staged.alpha.tobytes() == repeated.alpha.tobytes()
                and metadata == repeated_metadata
            ),
            "resource": {
                "input_bytes": metadata.input_bytes,
                "output_bytes": metadata.output_bytes,
                "global_context_bytes": metadata.global_context_bytes,
                "percentile_histogram_bytes": metadata.percentile_histogram_bytes,
                "source_read_calls": metadata.source_read_calls,
                "total_source_read_bytes": metadata.total_source_read_bytes,
                "max_source_window_shape": list(metadata.max_source_window_shape),
                "max_source_window_bytes": metadata.max_source_window_bytes,
                "declared_max_tile_workspace_bytes": metadata.declared_max_tile_workspace_bytes,
                "persistent_derived_full_scalar_bytes": metadata.persistent_derived_full_scalar_bytes,
                "scratch_disk_bytes": metadata.scratch_disk_bytes,
                "context_bytes": {
                    item.name: item.coarse_bytes for item in metadata.contexts
                },
                "plan_fingerprint": metadata.static_plan_fingerprint,
            },
        }
        row["passes"] = bool(
            row["alpha_max_error"] <= gates["staged_v2_alpha_max_error"]
            and row["rgb_max_error"] <= gates["staged_v2_rgb_max_error"]
            and row["composite_max_error"] <= gates["staged_v2_composite_max_error"]
            and row["seam_max_error"] <= gates["staged_v2_seam_max_error"]
            and row["srgb8_byte_identical"]
            and row["repeat_exact"]
            and row["resource"]["max_source_window_shape"][0] < base.shape[0]
            and row["resource"]["persistent_derived_full_scalar_bytes"] == 0
            and row["resource"]["scratch_disk_bytes"] == 0
        )
        policies.append(row)
    legacy_metrics = {
        "alpha_max_drift": float(legacy_alpha.max()),
        "alpha_mean_drift": float(legacy_alpha.mean()),
        "rgb_max_drift": float(legacy_rgb.max()),
        "composite_max_drift": float(legacy_output_delta.max()),
        "composite_mean_drift": float(legacy_output_delta.mean()),
        "uint8_changed_fraction": float(np.count_nonzero(legacy_code_delta) / legacy_code_delta.size),
        "uint8_max_code_delta": int(legacy_code_delta.max()),
    }
    legacy_metrics["passes"] = bool(
        legacy_metrics["alpha_max_drift"] <= gates["legacy_alpha_max_drift"]
        and legacy_metrics["alpha_mean_drift"] <= gates["legacy_alpha_mean_drift"]
        and legacy_metrics["rgb_max_drift"] <= gates["legacy_rgb_max_drift"]
        and legacy_metrics["composite_max_drift"] <= gates["legacy_composite_max_drift"]
        and legacy_metrics["composite_mean_drift"] <= gates["legacy_composite_mean_drift"]
        and legacy_metrics["uint8_changed_fraction"] <= gates["legacy_uint8_changed_fraction"]
        and legacy_metrics["uint8_max_code_delta"] <= gates["legacy_uint8_max_code_delta"]
    )
    return {
        "case_id": case["case_id"],
        "shape": list(base.shape),
        "source_sha256": case.get("sha256", hashlib.sha256(base.tobytes()).hexdigest()),
        "policies": policies,
        "legacy": legacy_metrics,
        "passes": bool(all(item["passes"] for item in policies) and legacy_metrics["passes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cases = [
        *(dict(item, source_kind="synthetic") for item in config["synthetic_cases"]),
        *(dict(item, source_kind="real") for item in config["real_confirmatory_cases"]),
    ]
    if args.reverse:
        cases.reverse()
    rows = []
    for case in cases:
        base = _synthetic(case) if case["source_kind"] == "synthetic" else _real(case)
        rows.append(_metrics(case, base, config))
    rows.sort(key=lambda item: item["case_id"])
    payload = {
        "schema_version": 1,
        "node": config["node"],
        "executor_version": config["executor_version"],
        "config_sha256": _sha256(CONFIG),
        "core_sha256": _sha256(CORE),
        "cases": rows,
        "gates": config["gates"],
        "scientific_identity": _json_sha(rows),
        "status": "PASS_PRIVATE_STAGED_PHYSICAL_COLOUR_EXECUTOR"
        if all(row["passes"] for row in rows)
        else "FAIL_CLOSED_STAGED_PHYSICAL_COLOUR_EXECUTOR",
        "claim_ceiling": config["claim_ceiling"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output), "status": payload["status"]}, sort_keys=True))
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
