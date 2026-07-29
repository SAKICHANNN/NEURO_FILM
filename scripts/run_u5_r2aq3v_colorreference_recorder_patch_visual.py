#!/usr/bin/env python
"""Render and audit labeled recorder proxy patch evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_u5_r2an0_paired_positive_film_recovery import (  # noqa: E402
    _atomic_write,
    _canonical_json,
)
from src.roll2film.colorreference_bounded_bernstein import (  # noqa: E402
    BoundedBernsteinModel,
    apply_bounded_bernstein,
)
from src.roll2film.colorreference_proxy_baselines import (  # noqa: E402
    xyz_to_lab_d50,
)


CONFIG_SHA256 = "1332ad10ff028a8c27747c5b6e4648c9de3fb725fb894ca40b3fb4307c2bc567"
EXPERIMENT_ID = "u5.r2aq3v-colorreference-recorder-patch-visual-v1"
REPORT_SCHEMA = "neuro-film.u5.r2aq3v.recorder-patch-visual.v1"
_D50_TO_D65 = np.asarray(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_LINEAR_SRGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AQ3V requires a clean tracked worktree")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AQ3V config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"] != EXPERIMENT_ID
        or config["ordinary_photo_application_allowed"]
        or config["production_integration_allowed"]
        or config["consensus_contract"]["patches_per_slide"] != 288
    ):
        raise ValueError("AQ3V frozen contract mismatch")
    return config


def sample_position(sample_id: str) -> tuple[int, int]:
    grayscale = re.fullmatch(r"GS([0-9]|1[0-9]|2[0-3])", sample_id)
    if grayscale is not None:
        return 12, int(grayscale.group(1))
    match = re.fullmatch(
        r"(A|B|C|D|E|F|G|H|I|J|K|L)([1-9]|1[0-9]|2[0-2])",
        sample_id,
    )
    if match is None:
        raise ValueError("unsupported ColorReference sample id")
    prefix, number = match.groups()
    row = ord(prefix) - ord("A")
    column = int(number) - 1
    return row, column


def xyz_d50_to_srgb_preview(
    xyz: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(xyz, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("preview XYZ must be finite Nx3")
    d65 = values @ _D50_TO_D65.T
    linear = d65 @ _XYZ_D65_TO_LINEAR_SRGB.T
    low_fraction = float(np.mean(linear < 0.0))
    high_fraction = float(np.mean(linear > 1.0))
    clipped = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return encoded, {
        "preclamp_negative_component_fraction": low_fraction,
        "preclamp_above_one_component_fraction": high_fraction,
    }


def _load_bundle(config: dict[str, Any]) -> BoundedBernsteinModel:
    payload = (ROOT / config["parent"]["bundle"]).read_bytes()
    if _sha256(payload) != config["parent"]["bundle_sha256"]:
        raise ValueError("AQ3V bundle hash mismatch")
    bundle = json.loads(payload)
    if (
        bundle["model_id"] != "bernstein_d3"
        or "recorder-device" not in bundle["input_semantics"]
        or "not applicable to ordinary digital photographs"
        not in bundle["claim_ceiling"]
    ):
        raise ValueError("AQ3V bundle semantics mismatch")
    return BoundedBernsteinModel(
        degree=int(bundle["degree"]),
        coefficients=np.asarray(bundle["coefficients"], dtype=np.float64),
        lower_bound=float(bundle["lower_bound"]),
        upper_bound=float(bundle["upper_bound"]),
    )


def _load_consensus(
    config: dict[str, Any],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    payload = (ROOT / config["parent"]["pair_table"]).read_bytes()
    if _sha256(payload) != config["parent"]["pair_table_sha256"]:
        raise ValueError("AQ3V pair-table hash mismatch")
    rows = list(
        csv.DictReader(
            io.StringIO(payload.decode("utf-8")), delimiter="\t"
        )
    )
    grouped: dict[
        tuple[int, str], list[tuple[np.ndarray, np.ndarray]]
    ] = {}
    for row in rows:
        key = (int(row["slide_index"]), row["sample_id"])
        grouped.setdefault(key, []).append(
            (
                np.asarray(
                    [
                        float(row["source_r"]),
                        float(row["source_g"]),
                        float(row["source_b"]),
                    ],
                    dtype=np.float64,
                ),
                0.01
                * np.asarray(
                    [
                        float(row["xyz_x"]),
                        float(row["xyz_y"]),
                        float(row["xyz_z"]),
                    ],
                    dtype=np.float64,
                ),
            )
        )
    result: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for (slide, sample_id), values in grouped.items():
        if len(values) != 6:
            raise RuntimeError("AQ3V consensus support mismatch")
        sources = np.stack([value[0] for value in values])
        if not np.all(sources == sources[0]):
            raise RuntimeError("AQ3V source differs across target sets")
        result.setdefault(slide, {})[sample_id] = {
            "source": sources[0],
            "target_xyz": np.median(
                np.stack([value[1] for value in values]), axis=0
            ),
        }
    if set(result) != {1, 2, 3, 4, 5} or any(
        len(result[slide]) != 288 for slide in result
    ):
        raise RuntimeError("AQ3V slide inventory mismatch")
    return result


def _heatmap(delta: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(delta / maximum, 0.0, 1.0)
    anchors = np.asarray(
        [
            [30, 60, 180],
            [30, 190, 170],
            [240, 220, 40],
            [220, 40, 30],
        ],
        dtype=np.float64,
    ) / 255.0
    position = normalized * 3.0
    index = np.minimum(position.astype(np.int64), 2)
    fraction = position - index
    return (
        anchors[index] * (1.0 - fraction[:, None])
        + anchors[index + 1] * fraction[:, None]
    )


def _patch_panel(
    sample_ids: list[str],
    colors: np.ndarray,
    *,
    patch_pixels: int,
) -> Image.Image:
    background = (24, 24, 24)
    image = Image.new(
        "RGB", (24 * patch_pixels, 13 * patch_pixels), background
    )
    draw = ImageDraw.Draw(image)
    for sample_id, color in zip(sample_ids, colors):
        row, column = sample_position(sample_id)
        rgb = tuple(
            int(value)
            for value in np.rint(np.clip(color, 0.0, 1.0) * 255.0)
        )
        left = column * patch_pixels
        top = row * patch_pixels
        draw.rectangle(
            (
                left,
                top,
                left + patch_pixels - 1,
                top + patch_pixels - 1,
            ),
            fill=rgb,
        )
    return image


def run_experiment(
    config: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    model = _load_bundle(config)
    consensus = _load_consensus(config)
    preview = config["preview_contract"]
    patch_pixels = preview["patch_pixels"]
    panel_width = 24 * patch_pixels
    panel_height = 13 * patch_pixels
    panel_gap = preview["panel_gap_pixels"]
    slide_gap = preview["slide_gap_pixels"]
    label_height = 34
    left_label = 96
    canvas = Image.new(
        "RGB",
        (
            left_label + 4 * panel_width + 3 * panel_gap,
            5 * (label_height + panel_height) + 4 * slide_gap,
        ),
        (12, 12, 12),
    )
    draw = ImageDraw.Draw(canvas)
    panel_names = preview["panels"]
    all_delta = []
    clipping_rows = []
    for slide_offset, slide in enumerate((1, 2, 3, 4, 5)):
        rows = consensus[slide]
        sample_ids = sorted(
            rows,
            key=lambda sample: sample_position(sample),
        )
        source = np.stack([rows[sample]["source"] for sample in sample_ids])
        target_xyz = np.stack(
            [rows[sample]["target_xyz"] for sample in sample_ids]
        )
        predicted_xyz = apply_bounded_bernstein(model, source)
        delta = np.linalg.norm(
            xyz_to_lab_d50(predicted_xyz)
            - xyz_to_lab_d50(target_xyz),
            axis=1,
        )
        all_delta.append(delta)
        predicted_preview, predicted_clipping = (
            xyz_d50_to_srgb_preview(predicted_xyz)
        )
        target_preview, target_clipping = xyz_d50_to_srgb_preview(
            target_xyz
        )
        clipping_rows.append(
            {
                "slide_index": slide,
                "predicted": predicted_clipping,
                "measured_consensus": target_clipping,
            }
        )
        panels = [
            _patch_panel(
                sample_ids, source, patch_pixels=patch_pixels
            ),
            _patch_panel(
                sample_ids,
                predicted_preview,
                patch_pixels=patch_pixels,
            ),
            _patch_panel(
                sample_ids,
                target_preview,
                patch_pixels=patch_pixels,
            ),
            _patch_panel(
                sample_ids,
                _heatmap(
                    delta,
                    config["automatic_gates"]["maximum_deltae76_max"],
                ),
                patch_pixels=patch_pixels,
            ),
        ]
        top = slide_offset * (
            label_height + panel_height + slide_gap
        )
        draw.text((8, top + label_height + 4), f"Slide {slide}", fill="white")
        for index, (name, panel) in enumerate(
            zip(panel_names, panels)
        ):
            left = left_label + index * (panel_width + panel_gap)
            draw.text((left, top + 8), name, fill="white")
            canvas.paste(panel, (left, top + label_height))
    delta = np.concatenate(all_delta)
    gates = config["automatic_gates"]
    checks = [
        {
            "name": "consensus_patch_count_and_finite",
            "passed": delta.shape == (1440,)
            and bool(np.all(np.isfinite(delta))),
        },
        {
            "name": "mean_deltae76",
            "passed": float(np.mean(delta)) <= gates["mean_deltae76_max"],
        },
        {
            "name": "p95_deltae76",
            "passed": float(np.percentile(delta, 95.0))
            <= gates["p95_deltae76_max"],
        },
        {
            "name": "maximum_deltae76",
            "passed": float(np.max(delta))
            <= gates["maximum_deltae76_max"],
        },
    ]
    image_buffer = io.BytesIO()
    canvas.save(
        image_buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    image_bytes = image_buffer.getvalue()
    passed = all(check["passed"] for check in checks)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "bundle_sha256": config["parent"]["bundle_sha256"],
        "pair_table_sha256": config["parent"]["pair_table_sha256"],
        "contact_sheet_sha256": _sha256(image_bytes),
        "patch_count": int(delta.shape[0]),
        "consensus_prediction": {
            "mean_deltae76": float(np.mean(delta)),
            "median_deltae76": float(np.median(delta)),
            "p95_deltae76": float(np.percentile(delta, 95.0)),
            "maximum_deltae76": float(np.max(delta)),
        },
        "preview_clipping": clipping_rows,
        "automatic_checks": checks,
        "automatic_pass": passed,
        "visual_review_required": passed,
        "ordinary_photo_application_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, image_bytes


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    config = load_config(config_path, CONFIG_SHA256)
    report, image_bytes = run_experiment(config)
    report_bytes = _canonical_json(report)
    _atomic_write(output_root / "contact_sheet.png", image_bytes)
    _atomic_write(output_root / "report.json", report_bytes)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "consensus_prediction": report[
                    "consensus_prediction"
                ],
                "contact_sheet_sha256": report[
                    "contact_sheet_sha256"
                ],
                "report_sha256": _sha256(report_bytes),
                "visual_review_required": report[
                    "visual_review_required"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aq3v_colorreference_recorder_patch_visual_v1.json",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=CONFIG_SHA256,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/u5_r2aq3v_colorreference_recorder_patch_visual_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_config(args.config, args.expected_config_sha256)
    run(args.config, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
