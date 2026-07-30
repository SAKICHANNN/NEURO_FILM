"""Fresh-photo OOD validation for a fixed FilmMatch code-domain operator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.eval.filmmatch_chart_pairs import linear_reflection_to_slog3
from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file
from src.eval.filmmatch_validation_scene import _apply_rows, _save_rgb16_png
from src.roll2film.factorized_monotone_bernstein import (
    FactorizedMonotoneBernsteinOperator,
)


def _rgb_to_xyz_matrix(
    primaries: np.ndarray, white_xy: tuple[float, float]
) -> np.ndarray:
    xy = np.asarray(primaries, dtype=np.float64)
    if xy.shape != (3, 2):
        raise ValueError("primaries must have shape (3, 2)")
    xyz = np.stack(
        (
            xy[:, 0] / xy[:, 1],
            np.ones(3, dtype=np.float64),
            (1.0 - xy[:, 0] - xy[:, 1]) / xy[:, 1],
        ),
        axis=0,
    )
    wx, wy = map(float, white_xy)
    white = np.asarray([wx / wy, 1.0, (1.0 - wx - wy) / wy])
    return xyz * np.linalg.solve(xyz, white)[None, :]


def linear_srgb_to_sgamut3_cine(values: np.ndarray) -> np.ndarray:
    """Convert D65 linear sRGB to D65 linear S-Gamut3.Cine."""
    rgb = np.asarray(values, dtype=np.float64)
    if (
        rgb.ndim < 1
        or rgb.shape[-1] != 3
        or not np.all(np.isfinite(rgb))
    ):
        raise ValueError("linear sRGB must be finite RGB data")
    white = (0.3127, 0.3290)
    srgb = _rgb_to_xyz_matrix(
        np.asarray([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]),
        white,
    )
    sgamut = _rgb_to_xyz_matrix(
        np.asarray([[0.766, 0.275], [0.225, 0.8], [0.089, -0.087]]),
        white,
    )
    return rgb @ np.linalg.solve(sgamut, srgb).T


def _srgb_eotf(values: np.ndarray) -> np.ndarray:
    code = np.asarray(values, dtype=np.float64)
    return np.where(
        code <= 0.04045,
        code / 12.92,
        np.power((code + 0.055) / 1.055, 2.4),
    )


def display_srgb_to_sgamut3_cine_slog3(values: np.ndarray) -> np.ndarray:
    """Deterministically adapt relative display sRGB into the operator rail."""
    code = np.asarray(values, dtype=np.float64)
    if (
        code.ndim < 1
        or code.shape[-1] != 3
        or not np.all(np.isfinite(code))
        or np.any(code < 0.0)
        or np.any(code > 1.0)
    ):
        raise ValueError("display sRGB must be finite [0, 1] RGB")
    linear = linear_srgb_to_sgamut3_cine(_srgb_eotf(code))
    if np.any(linear < -1e-12) or np.any(linear > 1.0 + 1e-12):
        raise ValueError("sRGB to S-Gamut3.Cine escaped the expected cube")
    return linear_reflection_to_slog3(np.clip(linear, 0.0, 1.0))


def _contact_sheet(
    rows: list[dict[str, Any]], output_dir: Path, width: int
) -> str:
    cell_width = int(width)
    cell_height = int(round(cell_width * 0.75))
    label_height = 28
    canvas = Image.new(
        "RGB",
        (cell_width * 2, len(rows) * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * (cell_height + label_height)
        for column, key in enumerate(("source_path", "candidate_path")):
            image = Image.open(output_dir / row[key]).convert("RGB")
            image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            x = column * cell_width + (cell_width - image.width) // 2
            offset_y = y + label_height + (cell_height - image.height) // 2
            canvas.paste(image, (x, offset_y))
        draw.text((6, y + 6), f"{row['id']} source", fill="black")
        draw.text((cell_width + 6, y + 6), "candidate", fill="black")
    path = output_dir / "contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=6)
    return sha256_file(path)


def evaluate_fresh_ood(
    operator_report: Mapping[str, Any],
    manifest: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    operator = FactorizedMonotoneBernsteinOperator.from_dict(
        operator_report["final_fit"]["operator"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest:
        source_path = root / str(manifest_row["decoded_path"])
        if sha256_file(source_path) != manifest_row["decoded_sha256"]:
            raise ValueError("fresh OOD decoded source identity drift")
        source = (
            np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float64)
            / 255.0
        )
        native_code = display_srgb_to_sgamut3_cine_slog3(source)
        candidate = _apply_rows(
            operator, native_code, int(config["execution"]["row_chunk"])
        )
        identifier = str(manifest_row["id"])
        source_name = f"{identifier}_source.png"
        candidate_name = f"{identifier}_candidate.png"
        source_sha = _save_rgb16_png(output_dir / source_name, source)
        candidate_sha = _save_rgb16_png(
            output_dir / candidate_name, candidate
        )
        boundary = float(
            np.mean(
                np.any(
                    ((candidate <= 0.0) & (native_code > 0.0))
                    | ((candidate >= 1.0) & (native_code < 1.0)),
                    axis=2,
                )
            )
        )
        rows.append(
            {
                "id": identifier,
                "make": manifest_row["make"],
                "source_manifest_sha256": manifest_row["decoded_sha256"],
                "source_path": source_name,
                "source_output_sha256": source_sha,
                "candidate_path": candidate_name,
                "candidate_output_sha256": candidate_sha,
                "finite": bool(np.all(np.isfinite(candidate))),
                "minimum_native_code": float(np.min(native_code)),
                "maximum_native_code": float(np.max(native_code)),
                "minimum_output": float(np.min(candidate)),
                "maximum_output": float(np.max(candidate)),
                "new_boundary_fraction": boundary,
                "style_rgb_rmse_in_operator_code_domain": float(
                    np.sqrt(np.mean(np.square(candidate - native_code)))
                ),
            }
        )
    contact_sheet_sha = _contact_sheet(
        rows, output_dir, int(config["execution"]["contact_sheet_cell_width"])
    )
    gates = config["automatic_gate"]
    passed = bool(
        len(rows) >= int(gates["minimum_source_count"])
        and len({row["make"] for row in rows})
        >= int(gates["minimum_camera_make_count"])
        and all(row["finite"] for row in rows)
        and max(row["new_boundary_fraction"] for row in rows)
        <= float(gates["maximum_new_boundary_fraction"])
        and np.median(
            [
                row["style_rgb_rmse_in_operator_code_domain"]
                for row in rows
            ]
        )
        >= float(gates["minimum_median_style_rgb_rmse"])
    )
    report = {
        "schema": config["report_schema"],
        "experiment_id": config["experiment_id"],
        "operator": operator.to_dict(),
        "domain_adapter": {
            "input": "relative display sRGB approximation",
            "linearization": "IEC 61966-2-1 sRGB EOTF",
            "gamut": "D65 linear sRGB to Sony S-Gamut3.Cine primaries",
            "transfer": "Sony published scene-linear reflection to S-Log3",
            "not_camera_idt": True,
            "not_calibrated": True,
        },
        "rows": rows,
        "contact_sheet_sha256": contact_sheet_sha,
        "automatic_gate_passed": passed,
        "visual_review_opened": passed,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "display_srgb_to_sgamut3_cine_slog3",
    "evaluate_fresh_ood",
    "linear_srgb_to_sgamut3_cine",
]
