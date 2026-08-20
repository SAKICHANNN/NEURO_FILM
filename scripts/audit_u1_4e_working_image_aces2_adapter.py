#!/usr/bin/env python3
"""Formal U1.4E WorkingImage to official ACES 2 adapter audit."""

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
    SOURCE_SPACE,
    OcioAces2RuntimeError,
    apply_working_image_aces2_output,
    build_aces2_numeric_fixture,
    load_aces2_config,
    target_display_view,
)
from src.preprocess.types import SourceProfile, WorkingImage

REPORT_SCHEMA = "neuro_film.u1_4e_working_image_aces2_adapter_result.v1"
TARGETS = ("sdr_rec709", "hdr_rec2020_pq")
SOURCE_SPACES = {
    "linear_srgb": "Linear Rec.709 (sRGB)",
    "linear_rec2020": "Linear Rec.2020",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _working(pixels: np.ndarray, working_space: str) -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space=working_space,
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "formal synthetic fixture"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("u1_4e_synthetic_fixture.exr"),
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_bindings(config: dict[str, Any]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for key in ("contract", "implementation", "runner"):
        relative = config["bindings"][f"{key}_path"]
        actual = _sha256_file(ROOT / relative)
        if actual != config["bindings"][f"{key}_sha256"]:
            raise ValueError(f"binding mismatch for {key}")
        checked[f"{key}_path"] = relative
        checked[f"{key}_sha256"] = actual
    return checked


def _official_outputs(
    ordered: np.ndarray, source_space: str, target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import PyOpenColorIO as ocio

    config = load_aces2_config()
    source_processor = config.getProcessor(source_space, SOURCE_SPACE).getDefaultCPUProcessor()
    scalar_acescg = np.empty_like(ordered)
    for index, row in enumerate(ordered):
        scalar_acescg[index] = source_processor.applyRGB(
            [float(row[0]), float(row[1]), float(row[2])]
        )
    packed_acescg = np.ascontiguousarray(ordered.copy())
    source_processor.apply(ocio.PackedImageDesc(packed_acescg, packed_acescg.shape[0], 1, 3))
    display, view = target_display_view(target)
    direct_transform = ocio.DisplayViewTransform(
        src=source_space, display=display, view=view
    )
    direct = np.ascontiguousarray(ordered.copy())
    config.getProcessor(direct_transform).getDefaultCPUProcessor().apply(
        ocio.PackedImageDesc(direct, direct.shape[0], 1, 3)
    )
    return scalar_acescg, packed_acescg, direct


def _rejection_checks() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    base = np.zeros((1, 1, 3), dtype=np.float32)
    for name, working in (
        ("unsupported_space", _working(base.copy(), "acescg")),
        ("display_linear", _working(base.copy(), "linear_srgb")),
    ):
        if name == "display_linear":
            working.transfer_state = "display_linear"
        try:
            apply_working_image_aces2_output(working, "sdr_rec709")
        except OcioAces2RuntimeError:
            checks[name] = True
        else:
            checks[name] = False
    for name, factory in (
        ("wrong_dtype", lambda: _working(np.zeros((1, 1, 3), np.float64), "linear_srgb")),
        ("wrong_shape", lambda: _working(np.zeros((1, 3), np.float32), "linear_srgb")),
        (
            "nonfinite",
            lambda: _working(np.full((1, 1, 3), np.nan, np.float32), "linear_srgb"),
        ),
    ):
        try:
            factory()
        except (TypeError, ValueError):
            checks[name] = True
        else:
            checks[name] = False
    return checks


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = _validate_bindings(config)
    fixture = build_aces2_numeric_fixture()
    ordered = np.ascontiguousarray(fixture[::-1] if reverse else fixture)
    before = _array_sha256(ordered)
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for working_space, ocio_source in SOURCE_SPACES.items():
        rows[working_space] = {}
        for target in TARGETS:
            scalar_acescg, packed_acescg, direct = _official_outputs(
                ordered, ocio_source, target
            )
            adapter = apply_working_image_aces2_output(
                _working(ordered.reshape(1, -1, 3), working_space), target
            ).reshape(-1, 3)
            if reverse:
                scalar_acescg = np.ascontiguousarray(scalar_acescg[::-1])
                packed_acescg = np.ascontiguousarray(packed_acescg[::-1])
                direct = np.ascontiguousarray(direct[::-1])
                adapter = np.ascontiguousarray(adapter[::-1])
            neutral = adapter[-257:]
            neutral_mean = neutral.mean(axis=1, dtype=np.float64)
            rows[working_space][target] = {
                "adapter_sha256": _array_sha256(adapter),
                "direct_sha256": _array_sha256(direct),
                "finite": bool(np.isfinite(adapter).all()),
                "maximum_adapter_direct_absolute_error": float(
                    np.max(np.abs(adapter - direct))
                ),
                "maximum_neutral_channel_spread": float(
                    np.max(np.max(neutral, axis=1) - np.min(neutral, axis=1))
                ),
                "maximum_source_scalar_packed_absolute_error": float(
                    np.max(np.abs(scalar_acescg - packed_acescg))
                ),
                "minimum_neutral_mean_step": float(np.min(np.diff(neutral_mean))),
            }
    flat_rows = [row for source in rows.values() for row in source.values()]
    rejection_checks = _rejection_checks()
    gate_config = config["gates"]
    gates = {
        "adapter_direct": all(
            row["maximum_adapter_direct_absolute_error"]
            <= gate_config["maximum_adapter_direct_absolute_error"]
            for row in flat_rows
        ),
        "finite": all(row["finite"] for row in flat_rows),
        "input_unchanged": _array_sha256(ordered) == before,
        "neutral_monotonic": all(
            row["minimum_neutral_mean_step"] >= -gate_config["neutral_monotonic_tolerance"]
            for row in flat_rows
        ),
        "neutral_spread": all(
            row["maximum_neutral_channel_spread"]
            <= gate_config["maximum_neutral_channel_spread"]
            for row in flat_rows
        ),
        "rejections": all(rejection_checks.values()),
        "source_scalar_packed": all(
            row["maximum_source_scalar_packed_absolute_error"]
            <= gate_config["maximum_source_scalar_packed_absolute_error"]
            for row in flat_rows
        ),
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "metrics": {
            "fixture_rows_per_source": int(fixture.shape[0]),
            "fixture_sha256": _array_sha256(fixture),
            "rejection_checks": rejection_checks,
            "sources": rows,
        },
        "node": config["node"],
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_WORKING_IMAGE_ACES2_ADAPTER"
        if all(gates.values())
        else "FAIL_CLOSED_WORKING_IMAGE_ACES2_ADAPTER",
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
        default=Path("configs/u1_4e_working_image_aces2_adapter_v1.json"),
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
