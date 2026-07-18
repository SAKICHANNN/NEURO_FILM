"""Run the frozen U1.6G4C numerical and real-field compatibility audit."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import (  # noqa: E402
    build_chunk_invariant_global_stage,
    build_chunk_invariant_global_stage_from_rows,
    build_shape_stable_global_stage,
    plan_chunk_invariant_global_resample,
    plan_shape_stable_global_resample,
    reconstruct_chunk_invariant_full,
    reconstruct_chunk_invariant_tiled,
    reconstruct_shape_stable_full,
)

FROZEN_V1_OUTPUT_SHA256 = "c3a59c15ff785f1b2e64fa146cf6dda4f557fdab2d5e8026720660a9fa7d3c38"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigma(value: object) -> float | tuple[float, float]:
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    return float(value)


def _luma(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32)
    rgb /= np.float32(255.0)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(rgb * weights, axis=2, dtype=np.float32)


def _row_stage(field: np.ndarray, plan, chunk: int):
    return build_chunk_invariant_global_stage_from_rows(
        field.shape,
        plan,
        coarse_row_chunk=chunk,
        reader=lambda y0, y1: field[y0:y1],
    )


def _audit_v2_case(case: dict, field: np.ndarray) -> tuple[dict, np.ndarray]:
    plan = plan_chunk_invariant_global_resample(field.shape[:2], _sigma(case["sigma"]))
    stage = build_chunk_invariant_global_stage(field, plan)
    full = reconstruct_chunk_invariant_full(stage)
    repeated = build_chunk_invariant_global_stage(field, plan)
    rows = []
    for chunk in case["coarse_row_chunks"]:
        row_stage, metadata = _row_stage(field, plan, int(chunk))
        row_output = reconstruct_chunk_invariant_full(row_stage)
        second_stage, second_metadata = _row_stage(field, plan, int(chunk))
        rows.append(
            {
                "chunk": int(chunk),
                "coarse_byte_identical": row_stage.coarse.tobytes() == stage.coarse.tobytes(),
                "output_byte_identical": row_output.tobytes() == full.tobytes(),
                "repeat_coarse_byte_identical": (
                    second_stage.coarse.tobytes() == row_stage.coarse.tobytes()
                ),
                "repeat_metadata_identical": second_metadata == metadata,
                "metadata": dataclasses.asdict(metadata),
            }
        )
    tiles = []
    for tile_size in case["tile_sizes"]:
        tiled, metadata = reconstruct_chunk_invariant_tiled(stage, tile_size=int(tile_size))
        tiles.append(
            {
                "tile_size": int(tile_size),
                "byte_identical": tiled.tobytes() == full.tobytes(),
                "metadata": dataclasses.asdict(metadata),
            }
        )
    result = {
        "case_id": case["case_id"],
        "shape": list(field.shape),
        "coarse_shape": list(plan.coarse_shape),
        "coarse_sha256": _sha256_bytes(stage.coarse.tobytes()),
        "output_sha256": _sha256_bytes(full.tobytes()),
        "repeat_full_stage_byte_identical": repeated.coarse.tobytes() == stage.coarse.tobytes(),
        "row_stages": rows,
        "tiled_outputs": tiles,
    }
    return result, full


def _v1_regression() -> dict:
    field = np.random.default_rng(71).random((257, 389), dtype=np.float32)
    stage = build_shape_stable_global_stage(
        field,
        plan_shape_stable_global_resample(field.shape, 52.0),
    )
    output = reconstruct_shape_stable_full(stage)
    actual = _sha256_bytes(output.tobytes())
    return {
        "expected_output_sha256": FROZEN_V1_OUTPUT_SHA256,
        "actual_output_sha256": actual,
        "passed": actual == FROZEN_V1_OUTPUT_SHA256,
    }


def _compatibility(case: dict, field: np.ndarray, v2_output: np.ndarray) -> dict:
    expected = case["v1_expected"]
    try:
        v1_stage = build_shape_stable_global_stage(
            field,
            plan_shape_stable_global_resample(field.shape, _sigma(case["sigma"])),
        )
        v1_output = reconstruct_shape_stable_full(v1_stage)
    except Exception as error:  # the frozen v1 last-cell failure is evidence
        return {
            "expected": expected,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "expected_failure_matched": (
                expected == "frozen_last_cell_shape_mismatch"
                and isinstance(error, ValueError)
                and "shape-mismatch" in str(error)
            ),
        }
    difference = np.abs(v2_output - v1_output)
    v1_u8 = np.rint(np.clip(v1_output, 0.0, 1.0) * 255.0).astype(np.uint8)
    v2_u8 = np.rint(np.clip(v2_output, 0.0, 1.0) * 255.0).astype(np.uint8)
    code_difference = np.abs(v2_u8.astype(np.int16) - v1_u8.astype(np.int16))
    return {
        "expected": expected,
        "status": "available",
        "max_absolute_drift": float(difference.max()),
        "mean_absolute_drift": float(difference.mean()),
        "uint8_changed_pixels": int(np.count_nonzero(code_difference)),
        "uint8_changed_fraction": float(np.count_nonzero(code_difference) / code_difference.size),
        "uint8_max_code_delta": int(code_difference.max()),
        "v1_output_sha256": _sha256_bytes(v1_output.tobytes()),
    }


def _panel(field: np.ndarray, label: str, *, difference: bool = False) -> Image.Image:
    if difference:
        peak = max(float(np.max(field)), 1e-12)
        display = np.clip(field / peak, 0.0, 1.0)
    else:
        display = np.clip(field, 0.0, 1.0)
    gray = Image.fromarray(np.rint(display * 255.0).astype(np.uint8), mode="L").convert("RGB")
    gray.thumbnail((480, 300), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (500, 340), "white")
    canvas.paste(gray, ((500 - gray.width) // 2, 28))
    ImageDraw.Draw(canvas).text((8, 7), label, fill="black")
    return canvas


def _contact_sheet(real_fields: list[dict], output: Path) -> None:
    rows: list[Image.Image] = []
    for item in real_fields:
        input_field = item["field"]
        v2_output = item["v2_output"]
        panels = [_panel(input_field, f"{item['case_id']} input luma"), _panel(v2_output, "v2 full")]
        row_stage, _ = _row_stage(input_field, item["plan"], int(item["chunks"][0]))
        row_output = reconstruct_chunk_invariant_full(row_stage)
        panels.append(_panel(np.abs(row_output - v2_output), "abs(row-full), normalized", difference=True))
        if item["compatibility"].get("status") == "available":
            v1_stage = build_shape_stable_global_stage(
                input_field,
                plan_shape_stable_global_resample(input_field.shape, item["sigma"]),
            )
            v1_output = reconstruct_shape_stable_full(v1_stage)
            panels.append(_panel(np.abs(v1_output - v2_output), "abs(v1-v2), normalized", difference=True))
        else:
            panels.append(_panel(np.zeros_like(v2_output), "v1 frozen last-cell failure"))
        row = Image.new("RGB", (sum(panel.width for panel in panels), 340), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False)


def run(config: dict, *, contact_sheet: Path | None) -> dict:
    synthetic = []
    for case in config["confirmatory_cases"]:
        shape = tuple(int(item) for item in case["shape"])
        field = np.random.default_rng(int(case["seed"])).normal(size=shape).astype(np.float32)
        result, _ = _audit_v2_case(case, field)
        synthetic.append(result)

    real = []
    sheet_fields = []
    for case in config["real_compatibility_cases"]:
        path = ROOT / case["path"]
        actual_sha = _sha256_path(path)
        if actual_sha != case["sha256"]:
            raise ValueError(f"real compatibility hash mismatch: {case['case_id']}")
        field = _luma(path)
        if list(field.shape) != case["shape"]:
            raise ValueError(f"real compatibility shape mismatch: {case['case_id']}")
        result, v2_output = _audit_v2_case(case, field)
        compatibility = _compatibility(case, field, v2_output)
        result["source_sha256"] = actual_sha
        result["v1_compatibility"] = compatibility
        real.append(result)
        sheet_fields.append(
            {
                "case_id": case["case_id"],
                "field": field,
                "v2_output": v2_output,
                "plan": plan_chunk_invariant_global_resample(field.shape, _sigma(case["sigma"])),
                "chunks": case["coarse_row_chunks"],
                "sigma": _sigma(case["sigma"]),
                "compatibility": compatibility,
            }
        )

    if contact_sheet is not None:
        _contact_sheet(sheet_fields, contact_sheet)

    all_cases = synthetic + real
    byte_gates = all(
        case["repeat_full_stage_byte_identical"]
        and all(
            row["coarse_byte_identical"]
            and row["output_byte_identical"]
            and row["repeat_coarse_byte_identical"]
            and row["repeat_metadata_identical"]
            and row["metadata"]["max_row_span"] < case["shape"][0]
            for row in case["row_stages"]
        )
        and all(tile["byte_identical"] for tile in case["tiled_outputs"])
        for case in all_cases
    )
    available = [
        case["v1_compatibility"]
        for case in real
        if case["v1_compatibility"]["status"] == "available"
    ]
    expected_failures = [
        case["v1_compatibility"]
        for case in real
        if case["v1_compatibility"]["status"] == "failed"
    ]
    gates = config["gates"]
    compatibility_pass = all(
        row["max_absolute_drift"] <= gates["v1_max_absolute_drift"]
        and row["mean_absolute_drift"] <= gates["v1_mean_absolute_drift"]
        and row["uint8_changed_fraction"] <= gates["v1_uint8_changed_fraction"]
        and row["uint8_max_code_delta"] <= gates["v1_uint8_max_code_delta"]
        for row in available
    ) and all(row["expected_failure_matched"] for row in expected_failures)
    v1_regression = _v1_regression()
    automatic_pass = byte_gates and compatibility_pass and v1_regression["passed"]
    return {
        "schema_version": 1,
        "node": config["node"],
        "operator_version": config["operator_version"],
        "synthetic_cases": synthetic,
        "real_compatibility_cases": real,
        "v1_regression": v1_regression,
        "gate_result": {
            "cross_chunk_and_tile_bytes": byte_gates,
            "v1_compatibility": compatibility_pass,
            "v1_regression": v1_regression["passed"],
            "automatic_pass": automatic_pass,
            "visual_gate_status": "pending_manual_autonomous_review",
            "promotion_pass": False,
        },
        "claim_ceiling": "numerical primitive and limited real-field compatibility only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, contact_sheet=args.contact_sheet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
