"""Run the frozen U1.6G4F explicit new-operator value/safety audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.filmfx import (  # noqa: E402
    composite_layers,
    density_halation_layer,
    execute_staged_density_halation_default,
    materialized_density_halation_v2_reference,
)
from src.filmfx.fast_blur import gaussian_filter_direct  # noqa: E402


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _load_case(case: dict) -> np.ndarray:
    path = ROOT / case["path"]
    if _sha_path(path) != case["sha256"]:
        raise ValueError(f"source hash mismatch: {case['case_id']}")
    with Image.open(path) as source:
        base = np.asarray(source.convert("RGB"), np.float32) / np.float32(255.0)
    if list(base.shape) != case["shape"]:
        raise ValueError(f"source shape mismatch: {case['case_id']}")
    if not np.isfinite(base).all():
        raise ValueError(f"nonfinite source: {case['case_id']}")
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


def effect_metrics(base: np.ndarray, output: np.ndarray, alpha: np.ndarray) -> dict:
    """Return the frozen identity-relative value and safety measurements."""
    if alpha.ndim == 3 and alpha.shape[2] == 1:
        alpha = alpha[..., 0]
    difference = np.abs(output - base)
    base_u8 = np.rint(base * 255.0).astype(np.uint8)
    output_u8 = np.rint(output * 255.0).astype(np.uint8)
    code_difference = np.abs(output_u8.astype(np.int16) - base_u8.astype(np.int16))
    luma = base @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    threshold = float(np.quantile(luma, 0.9))
    highlight = alpha[luma >= threshold]
    other = alpha[luma < threshold]
    highlight_mean = float(highlight.mean()) if highlight.size else 0.0
    other_mean = float(other.mean()) if other.size else 0.0
    high_clip = (output >= np.float32(254.5 / 255.0)) & (
        base < np.float32(254.5 / 255.0)
    )
    low_clip = (output <= np.float32(0.5 / 255.0)) & (
        base > np.float32(0.5 / 255.0)
    )
    return {
        "changed_uint8_channel_fraction": float(np.mean(code_difference > 0)),
        "composite_abs_p99": float(np.quantile(difference, 0.99)),
        "uint8_max_code_delta": int(code_difference.max()),
        "alpha_active_fraction": float(np.mean(alpha > np.float32(1e-4))),
        "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean()),
        "highlight_alpha_mean": highlight_mean,
        "other_alpha_mean": other_mean,
        "highlight_to_other_alpha_ratio": float(
            highlight_mean / max(other_mean, 1e-12)
            if highlight.size and other.size
            else 0.0
        ),
        "new_high_clip_fraction": float(np.mean(high_clip)),
        "new_low_clip_fraction": float(np.mean(low_clip)),
        "finite_output": bool(np.isfinite(output).all() and np.isfinite(alpha).all()),
        "bounded_output": bool(
            output.min() >= 0.0
            and output.max() <= 1.0
            and alpha.min() >= 0.0
            and alpha.max() <= 0.26
        ),
    }


def _effect_pass(metrics: dict, gates: dict) -> bool:
    return (
        metrics["changed_uint8_channel_fraction"]
        >= gates["effect_changed_uint8_channel_fraction_min"]
        and metrics["composite_abs_p99"] >= gates["effect_composite_abs_p99_min"]
        and metrics["uint8_max_code_delta"]
        >= gates["effect_uint8_max_code_delta_min"]
        and metrics["alpha_active_fraction"]
        >= gates["effect_alpha_active_fraction_min"]
        and metrics["highlight_to_other_alpha_ratio"]
        >= gates["effect_highlight_to_other_alpha_ratio_min"]
        and metrics["new_high_clip_fraction"]
        <= gates["effect_new_high_clip_fraction_max"]
        and metrics["new_low_clip_fraction"]
        <= gates["effect_new_low_clip_fraction_max"]
        and metrics["finite_output"]
        and metrics["bounded_output"]
    )


def _run_case(case: dict, config: dict) -> tuple[dict, dict]:
    base = _load_case(case)
    reference = materialized_density_halation_v2_reference(
        base, source_row_chunk=int(config["source_row_chunk"])
    )
    reference_output = composite_layers(base, [reference])
    legacy = density_halation_layer(base)
    legacy_output = composite_layers(base, [legacy])
    tile_rows = []
    primary = None
    for tile_size in config["tile_sizes"]:
        layer, metadata = execute_staged_density_halation_default(
            base,
            tile_size=int(tile_size),
            source_row_chunk=int(config["source_row_chunk"]),
            coarse_row_chunk=int(config["coarse_row_chunk"]),
        )
        output = composite_layers(base, [layer])
        alpha_difference = np.abs(layer.alpha - reference.alpha)
        output_difference = np.abs(output - reference_output)
        staged_u8 = np.rint(output * 255.0).astype(np.uint8)
        reference_u8 = np.rint(reference_output * 255.0).astype(np.uint8)
        row = {
            "tile_size": int(tile_size),
            "alpha_max_error": float(alpha_difference.max()),
            "composite_max_error": float(output_difference.max()),
            "seam_max_error": _seam_max(output_difference, int(tile_size)),
            "srgb8_byte_identical": staged_u8.tobytes() == reference_u8.tobytes(),
            "layer_alpha_sha256": _sha_bytes(layer.alpha.tobytes()),
            "output_sha256": _sha_bytes(output.tobytes()),
            "source_read_calls": metadata.source_read_calls,
            "max_source_window_shape": list(metadata.max_source_window_shape),
            "declared_max_tile_workspace_bytes": metadata.declared_max_tile_workspace_bytes,
        }
        gates = config["gates"]
        row["passed"] = (
            row["alpha_max_error"] <= gates["staged_v2_alpha_max_error"]
            and row["composite_max_error"] <= gates["staged_v2_composite_max_error"]
            and row["seam_max_error"] <= gates["staged_v2_seam_max_error"]
            and row["srgb8_byte_identical"]
        )
        tile_rows.append(row)
        if int(tile_size) == int(config["runtime_tile_size"]):
            primary = (layer, output)
    if primary is None:
        raise ValueError("runtime tile size must be one of tile_sizes")
    primary_layer, primary_output = primary
    metrics = effect_metrics(base, primary_output, primary_layer.alpha)
    legacy_difference = np.abs(primary_output - legacy_output)
    legacy_code = np.abs(
        np.rint(primary_output * 255.0).astype(np.int16)
        - np.rint(legacy_output * 255.0).astype(np.int16)
    )
    timings = []
    for _ in range(int(config["runtime_repeats"])):
        started = perf_counter()
        execute_staged_density_halation_default(
            base,
            tile_size=int(config["runtime_tile_size"]),
            source_row_chunk=int(config["source_row_chunk"]),
            coarse_row_chunk=int(config["coarse_row_chunk"]),
        )
        timings.append(perf_counter() - started)
    megapixels = base.shape[0] * base.shape[1] / 1_000_000.0
    seconds_per_megapixel = float(np.median(timings) / megapixels)
    runtime_pass = (
        seconds_per_megapixel
        <= config["gates"]["runtime_median_seconds_per_megapixel_max"]
    )
    deterministic = {
        "case_id": case["case_id"],
        "shape": list(base.shape),
        "source_sha256": case["sha256"],
        "tile_policies": tile_rows,
        "effect_metrics": metrics,
        "legacy_diagnostic": {
            "composite_max_drift": float(legacy_difference.max()),
            "composite_mean_drift": float(legacy_difference.mean()),
            "changed_uint8_channel_fraction": float(np.mean(legacy_code > 0)),
            "uint8_max_code_delta": int(legacy_code.max()),
        },
        "reference_alpha_sha256": _sha_bytes(reference.alpha.tobytes()),
        "reference_output_sha256": _sha_bytes(reference_output.tobytes()),
        "effect_pass": _effect_pass(metrics, config["gates"]),
    }
    result = {
        **deterministic,
        "runtime": {
            "seconds": timings,
            "median_seconds_per_megapixel": seconds_per_megapixel,
            "passed": runtime_pass,
        },
        "automatic_pass": bool(
            all(row["passed"] for row in tile_rows)
            and deterministic["effect_pass"]
            and runtime_pass
        ),
    }
    visual = {
        "base": base,
        "output": primary_output,
        "effect": np.abs(primary_output - base),
        "alpha": primary_layer.alpha,
    }
    return result, visual


def _panel(
    array: np.ndarray,
    label: str,
    *,
    amplify: float = 1.0,
    preserve_1to1: bool = False,
) -> Image.Image:
    display = np.clip(array * np.float32(amplify), 0.0, 1.0)
    image = Image.fromarray(np.rint(display * 255).astype(np.uint8), "RGB")
    if preserve_1to1:
        canvas = Image.new("RGB", (image.width + 20, image.height + 35), "white")
        canvas.paste(image, (10, 28))
    else:
        image.thumbnail((340, 250), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (360, 285), "white")
        canvas.paste(image, ((360 - image.width) // 2, 28))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
    return canvas


def _crop(array: np.ndarray, center: tuple[int, int], size: int = 512) -> np.ndarray:
    height, width = array.shape[:2]
    crop_height = min(size, height)
    crop_width = min(size, width)
    y0 = min(max(center[0] - crop_height // 2, 0), height - crop_height)
    x0 = min(max(center[1] - crop_width // 2, 0), width - crop_width)
    return array[y0 : y0 + crop_height, x0 : x0 + crop_width]


def _write_visuals(items: list[tuple[str, dict]], path: Path) -> dict:
    rows = []
    crop_centers = {}
    for case_id, item in items:
        alpha = item["alpha"][..., 0]
        smoothed = gaussian_filter_direct(alpha, 8.0)
        center = tuple(int(value) for value in np.unravel_index(np.argmax(smoothed), smoothed.shape))
        crop_centers[case_id] = list(center)
        alpha_rgb = np.repeat(item["alpha"], 3, axis=2)
        panels = [
            _panel(item["base"], f"{case_id} input"),
            _panel(item["output"], "v2 output"),
            _panel(item["effect"], "abs effect x20", amplify=20.0),
            _panel(alpha_rgb, "alpha x10", amplify=10.0),
            _panel(_crop(item["base"], center), "1:1 crop input", preserve_1to1=True),
            _panel(_crop(item["output"], center), "1:1 crop output", preserve_1to1=True),
            _panel(
                _crop(item["effect"], center),
                "1:1 crop effect x20",
                amplify=20.0,
                preserve_1to1=True,
            ),
        ]
        row = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "white",
        )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "PNG")
    return {"path": path.as_posix(), "sha256": _sha_path(path), "crop_centers_yx": crop_centers}


def _deterministic_payload_sha(cases: list[dict]) -> str:
    payload = []
    for case in cases:
        payload.append({key: value for key, value in case.items() if key not in {"runtime", "automatic_pass"}})
    return _sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def run(config: dict, *, contact_sheet: Path) -> dict:
    cases = []
    visuals = []
    for case in config["confirmatory_cases"]:
        result, visual = _run_case(case, config)
        cases.append(result)
        visuals.append((case["case_id"], visual))
    visual_evidence = _write_visuals(visuals, contact_sheet)
    return {
        "schema_version": 1,
        "node": config["node"],
        "operator_version": config["operator_version"],
        "confirmatory_cases": cases,
        "deterministic_payload_sha256": _deterministic_payload_sha(cases),
        "visual_evidence": visual_evidence,
        "gate_result": {
            "automatic_pass": all(case["automatic_pass"] for case in cases),
            "visual_gate_status": "pending_autonomous_review",
            "retention_pass": False,
        },
        "legacy_comparison": config["legacy_comparison"],
        "renderer_integration_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, contact_sheet=args.contact_sheet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate_result"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
