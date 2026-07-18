"""Run frozen U1.6G4E staged-density numerical, resource and visual gates."""

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
    DENSITY_FAMILY,
    available_halation_integration_capabilities,
    build_halation_resource_plan,
    composite_layers,
    density_halation_layer,
    execute_staged_density_halation_default,
    materialized_density_halation_v2_reference,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _synthetic(case: dict) -> np.ndarray:
    shape = tuple(int(item) for item in case["shape"])
    rng = np.random.default_rng(int(case["seed"]))
    y, x = np.mgrid[: shape[0], : shape[1]]
    if case["kind"] == "highlight_gradient":
        base = np.empty(shape, dtype=np.float32)
        base[..., 0] = 0.04 + 0.22 * x / max(shape[1] - 1, 1)
        base[..., 1] = 0.03 + 0.17 * y / max(shape[0] - 1, 1)
        base[..., 2] = 0.02 + 0.12 * (x + y) / max(sum(shape[:2]) - 2, 1)
        hot = np.exp(
            -((y - shape[0] * 0.43) ** 2 + (x - shape[1] * 0.61) ** 2) / 260.0
        ).astype(np.float32)
        base += hot[..., None] * np.array([0.92, 0.78, 0.48], np.float32)
    elif case["kind"] == "colour_hotspots":
        base = rng.uniform(0.01, 0.28, size=shape).astype(np.float32)
        for center_y, center_x, tint in (
            (0.30, 0.28, [1.0, 0.45, 0.12]),
            (0.68, 0.73, [0.62, 0.82, 1.0]),
        ):
            hot = np.exp(
                -(
                    (y - shape[0] * center_y) ** 2
                    + (x - shape[1] * center_x) ** 2
                )
                / 190.0
            ).astype(np.float32)
            base += hot[..., None] * np.asarray(tint, np.float32)
    else:
        raise ValueError(f"unknown synthetic kind: {case['kind']}")
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _real(case: dict) -> np.ndarray:
    path = ROOT / case["path"]
    if _sha_path(path) != case["sha256"]:
        raise ValueError(f"source hash mismatch: {case['case_id']}")
    with Image.open(path) as source:
        base = np.asarray(source.convert("RGB"), np.float32) / np.float32(255.0)
    if list(base.shape) != case["shape"]:
        raise ValueError(f"source shape mismatch: {case['case_id']}")
    return base


def _seam_max(difference: np.ndarray, tile_size: int) -> float:
    values = []
    for y in range(tile_size, difference.shape[0], tile_size):
        values.append(
            float(difference[max(0, y - 1) : min(difference.shape[0], y + 1)].max())
        )
    for x in range(tile_size, difference.shape[1], tile_size):
        values.append(
            float(difference[:, max(0, x - 1) : min(difference.shape[1], x + 1)].max())
        )
    return max(values, default=0.0)


def _metadata_sha(metadata) -> str:
    payload = json.dumps(
        dataclasses.asdict(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha_bytes(payload)


def _metadata_summary(metadata) -> dict:
    return {
        "sha256": _metadata_sha(metadata),
        "tile_count": metadata.tile_count,
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
        "contexts": [
            {
                "name": context.name,
                "coarse_shape": list(context.coarse_shape),
                "coarse_bytes": context.coarse_bytes,
                "reader_calls": context.row_stage.call_count,
                "max_row_span": context.row_stage.max_row_span,
                "total_read_bytes": context.row_stage.total_read_bytes,
            }
            for context in metadata.contexts
        ],
    }


def _comparison(staged, staged_output, reference, reference_output, legacy, legacy_output, tile):
    alpha_v2 = np.abs(staged.alpha - reference.alpha)
    output_v2 = np.abs(staged_output - reference_output)
    alpha_legacy = np.abs(staged.alpha - legacy.alpha)
    output_legacy = np.abs(staged_output - legacy_output)
    staged_u8 = np.rint(staged_output * 255.0).astype(np.uint8)
    reference_u8 = np.rint(reference_output * 255.0).astype(np.uint8)
    legacy_u8 = np.rint(legacy_output * 255.0).astype(np.uint8)
    legacy_code = np.abs(staged_u8.astype(np.int16) - legacy_u8.astype(np.int16))
    return {
        "tile_size": int(tile),
        "staged_v2_alpha_max_error": float(alpha_v2.max()),
        "staged_v2_composite_max_error": float(output_v2.max()),
        "staged_v2_seam_max_error": _seam_max(output_v2, int(tile)),
        "staged_v2_srgb8_byte_identical": staged_u8.tobytes() == reference_u8.tobytes(),
        "legacy_alpha_max_drift": float(alpha_legacy.max()),
        "legacy_alpha_mean_drift": float(alpha_legacy.mean()),
        "legacy_composite_max_drift": float(output_legacy.max()),
        "legacy_composite_mean_drift": float(output_legacy.mean()),
        "legacy_uint8_changed_fraction": float(
            np.count_nonzero(legacy_code) / legacy_code.size
        ),
        "legacy_uint8_max_code_delta": int(legacy_code.max()),
        "staged_alpha_sha256": _sha_bytes(staged.alpha.tobytes()),
        "staged_output_sha256": _sha_bytes(staged_output.tobytes()),
    }


def _row_pass(row: dict, gates: dict) -> bool:
    return (
        row["staged_v2_alpha_max_error"] <= gates["staged_v2_alpha_max_error"]
        and row["staged_v2_composite_max_error"]
        <= gates["staged_v2_composite_max_error"]
        and row["staged_v2_seam_max_error"] <= gates["staged_v2_seam_max_error"]
        and row["staged_v2_srgb8_byte_identical"]
        and row["legacy_alpha_max_drift"] <= gates["legacy_alpha_max_drift"]
        and row["legacy_alpha_mean_drift"] <= gates["legacy_alpha_mean_drift"]
        and row["legacy_composite_max_drift"]
        <= gates["legacy_composite_max_drift"]
        and row["legacy_composite_mean_drift"]
        <= gates["legacy_composite_mean_drift"]
        and row["legacy_uint8_changed_fraction"]
        <= gates["legacy_uint8_changed_fraction"]
        and row["legacy_uint8_max_code_delta"] <= gates["legacy_uint8_max_code_delta"]
        and row["repeat_float_and_metadata_identical"]
    )


def _run_case(case: dict, base: np.ndarray, config: dict) -> tuple[dict, dict]:
    legacy = density_halation_layer(base)
    reference = materialized_density_halation_v2_reference(
        base, source_row_chunk=int(config["source_row_chunk"])
    )
    legacy_output = composite_layers(base, [legacy])
    reference_output = composite_layers(base, [reference])
    rows = []
    visual = {}
    for tile in case["tile_sizes"]:
        staged, metadata = execute_staged_density_halation_default(
            base,
            tile_size=int(tile),
            source_row_chunk=int(config["source_row_chunk"]),
            coarse_row_chunk=int(config["coarse_row_chunk"]),
        )
        repeated, repeated_metadata = execute_staged_density_halation_default(
            base,
            tile_size=int(tile),
            source_row_chunk=int(config["source_row_chunk"]),
            coarse_row_chunk=int(config["coarse_row_chunk"]),
        )
        staged_output = composite_layers(base, [staged])
        row = _comparison(
            staged,
            staged_output,
            reference,
            reference_output,
            legacy,
            legacy_output,
            tile,
        )
        row["metadata"] = _metadata_summary(metadata)
        row["repeat_float_and_metadata_identical"] = (
            staged.rgb.tobytes() == repeated.rgb.tobytes()
            and staged.alpha.tobytes() == repeated.alpha.tobytes()
            and metadata == repeated_metadata
        )
        row["passed"] = _row_pass(row, config["gates"])
        rows.append(row)
        if not visual:
            visual = {
                "base": base,
                "legacy": legacy_output,
                "reference": reference_output,
                "staged": staged_output,
                "difference": np.abs(staged_output - reference_output),
            }
    return {
        "case_id": case["case_id"],
        "shape": list(base.shape),
        "reference_alpha_sha256": _sha_bytes(reference.alpha.tobytes()),
        "reference_output_sha256": _sha_bytes(reference_output.tobytes()),
        "legacy_output_sha256": _sha_bytes(legacy_output.tobytes()),
        "tile_policies": rows,
        "passed": all(row["passed"] for row in rows),
    }, visual


def _panel(array: np.ndarray, label: str, *, amplify: float = 1.0) -> Image.Image:
    display = np.clip(array * np.float32(amplify), 0.0, 1.0)
    image = Image.fromarray(np.rint(display * 255).astype(np.uint8), "RGB")
    image.thumbnail((440, 300), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (460, 340), "white")
    canvas.paste(image, ((460 - image.width) // 2, 30))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
    return canvas


def _write_visuals(items: list[tuple[str, dict]], sheet_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    outputs = {}
    for case_id, item in items:
        panels = [
            _panel(item["base"], f"{case_id} input"),
            _panel(item["legacy"], "legacy"),
            _panel(item["reference"], "materialized v2"),
            _panel(item["staged"], "staged"),
            _panel(item["difference"], "abs(staged-v2) x1000", amplify=1000.0),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), 340), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
        output_path = output_dir / f"{case_id}__staged.png"
        Image.fromarray(np.rint(np.clip(item["staged"], 0, 1) * 255).astype(np.uint8), "RGB").save(
            output_path, "PNG"
        )
        outputs[case_id] = {
            "path": str(output_path.as_posix()),
            "sha256": _sha_path(output_path),
        }
    sheet = Image.new(
        "RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white"
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, "PNG")
    return outputs


def _resource_arithmetic() -> dict:
    shape = (4000, 6000)
    tile = 512
    plan = build_halation_resource_plan(
        DENSITY_FAMILY,
        shape,
        tile_size=tile,
        available_capabilities=available_halation_integration_capabilities(),
    )
    expanded = min(shape[0], tile + 60) * min(shape[1], tile + 60)
    return {
        "shape": list(shape),
        "input_bytes": shape[0] * shape[1] * 3 * 4,
        "required_output_bytes": shape[0] * shape[1] * 4 * 4,
        "global_context_bytes": plan.peak_context_bytes,
        "percentile_histogram_bytes": plan.peak_workspace_bytes,
        "declared_max_tile_workspace_bytes": expanded * 64 * 4,
        "persistent_derived_full_scalar_bytes": 0,
        "scratch_disk_bytes": 0,
        "claim": "arithmetic categories only; not measured 24MP execution or 100MP readiness",
    }


def run(config: dict, *, sheet_path: Path, output_dir: Path) -> dict:
    synthetic = []
    for case in config["synthetic_cases"]:
        result, _ = _run_case(case, _synthetic(case), config)
        synthetic.append(result)
    real = []
    visual_items = []
    for case in config["real_confirmatory_cases"]:
        result, visual = _run_case(case, _real(case), config)
        real.append(result)
        visual_items.append((case["case_id"], visual))
    visual_outputs = _write_visuals(visual_items, sheet_path, output_dir)
    automatic_pass = all(row["passed"] for row in synthetic + real)
    return {
        "schema_version": 1,
        "node": config["node"],
        "executor_version": config["executor_version"],
        "synthetic_cases": synthetic,
        "real_confirmatory_cases": real,
        "resource_arithmetic_24mp": _resource_arithmetic(),
        "visual_outputs": visual_outputs,
        "gate_result": {
            "automatic_pass": automatic_pass,
            "visual_gate_status": "pending_manual_autonomous_review",
            "promotion_pass": False,
        },
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--visual-output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(
        config,
        sheet_path=args.contact_sheet,
        output_dir=args.visual_output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
