"""Build source-ink-snapped Kodak 250D/2383 curve annotations.

The physical-value guides below are human-guided semantic traces. Each guide
is converted through frozen graph axes and snapped to nearby black source ink.
Generated overlays remain ignored; the resulting pixel coordinates are
committed only after visual review and hash binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/u5_r2aa0_source_audit/extracted"
DEFAULT_OUTPUT = ROOT / "outputs/u5_r2aa1_curve_binding"
DEFAULT_JSON = ROOT / "configs/data/kodak_250d_2383_curve_pixels_v1.json"


AXES = {
    "250d_characteristic": {
        "x_pairs": [[0.0, 62.0], [5.0, 506.0]],
        "y_pairs": [[0.0, 506.0], [3.0, 61.0]],
        "mask_x": [62, 506],
        "mask_y": [61, 506],
    },
    "250d_sensitivity": {
        "x_pairs": [[250.0, 105.0], [750.0, 693.0]],
        "y_pairs": [[0.0, 603.0], [4.0, 158.0]],
        "mask_x": [105, 164, 223, 281, 340, 399, 458, 517, 575, 634, 693],
        "mask_y": [158, 269, 380, 491, 603],
    },
    "250d_dye_density": {
        "x_pairs": [[400.0, 105.0], [800.0, 645.0]],
        "y_pairs": [[-0.2, 670.0], [1.8, 130.0]],
        "mask_x": [105, 645],
        "mask_y": [130, 670],
    },
    "2383_characteristic": {
        "x_pairs": [[-3.0, 69.0], [3.0, 454.0]],
        "y_pairs": [[0.0, 438.0], [6.0, 54.0]],
        "mask_x": [69, 454],
        "mask_y": [54, 438],
    },
    "2383_sensitivity": {
        "x_pairs": [[250.0, 59.0], [750.0, 393.0]],
        "y_pairs": [[-3.0, 292.0], [1.0, 40.0]],
        "mask_x": [59, 92, 125, 159, 192, 225, 259, 292, 326, 359, 393],
        "mask_y": [40, 103, 166, 229, 292],
    },
    "2383_dye_density": {
        "x_pairs": [[250.0, 64.0], [800.0, 371.0]],
        "y_pairs": [[0.0, 343.0], [1.4, 36.0]],
        "mask_x": [64, 371],
        "mask_y": [36, 343],
    },
}


GUIDES = {
    "250d_characteristic": {
        "blue": [
            [0.0, 1.00], [0.5, 1.00], [1.0, 1.02], [1.5, 1.15],
            [2.0, 1.45], [2.5, 1.75], [3.0, 2.05], [3.5, 2.35],
            [4.0, 2.65], [4.5, 2.88], [4.70, 2.94], [4.85, 2.97],
        ],
        "green": [
            [0.0, 0.62], [0.6, 0.62], [1.0, 0.65], [1.4, 0.78],
            [1.8, 1.00], [2.2, 1.20], [2.6, 1.45], [3.0, 1.65],
            [3.4, 1.90], [3.8, 2.15], [4.2, 2.38], [4.6, 2.60],
            [4.9, 2.70],
        ],
        "red": [
            [0.0, 0.20], [0.6, 0.20], [1.0, 0.20], [1.3, 0.22],
            [1.6, 0.35], [2.0, 0.55], [2.4, 0.75], [2.8, 0.98],
            [3.2, 1.20], [3.6, 1.42], [4.0, 1.62], [4.4, 1.80],
            [4.8, 1.96],
        ],
    },
    "250d_sensitivity": {
        "yellow_forming": [
            [360, 1.55], [380, 1.90], [400, 2.30], [420, 2.35],
            [440, 2.40], [460, 2.35], [475, 2.48], [490, 1.60],
            [500, 1.00], [510, 0.40], [520, 0.05],
        ],
        "magenta_forming": [
            [400, 0.60], [420, 0.45], [440, 0.30], [460, 0.55],
            [475, 0.80], [490, 1.60], [510, 2.00], [530, 2.20],
            [550, 2.40], [570, 2.20], [585, 1.80], [600, 0.30],
        ],
        "cyan_forming": [
            [550, 0.02], [565, 0.10], [580, 0.40], [590, 1.10],
            [600, 1.65], [620, 2.10], [640, 2.20], [660, 2.40],
            [675, 2.35], [690, 1.50], [700, 0.15],
        ],
    },
    "250d_dye_density": {
        "yellow": [
            [400, 0.50], [420, 0.85], [440, 1.00], [460, 0.90],
            [480, 0.65], [500, 0.25], [520, 0.05], [540, 0.00],
            [580, -0.02], [640, -0.02], [700, -0.02], [780, 0.00],
        ],
        "magenta": [
            [430, -0.03], [450, -0.05], [470, 0.05], [490, 0.25],
            [510, 0.65], [530, 0.95], [550, 1.00], [570, 0.75],
            [590, 0.35], [610, 0.12], [640, 0.05], [700, 0.00],
        ],
        "cyan": [
            [500, 0.00], [530, 0.02], [550, 0.05], [580, 0.15],
            [610, 0.35], [640, 0.65], [670, 0.90], [690, 1.00],
            [710, 0.95], [740, 0.75], [770, 0.40], [800, 0.10],
        ],
        "midscale_neutral": [
            [400, 1.10], [420, 1.45], [440, 1.56], [460, 1.50],
            [480, 1.40], [500, 1.25], [520, 1.28], [540, 1.25],
            [560, 1.20], [580, 1.00], [600, 0.70], [620, 0.65],
            [640, 0.72], [660, 0.78], [680, 0.83], [700, 0.80],
            [720, 0.70], [740, 0.60], [760, 0.45], [780, 0.25],
            [800, 0.15],
        ],
        "minimum_density": [
            [400, 0.65], [420, 0.80], [440, 0.90], [460, 0.85],
            [480, 0.80], [500, 0.65], [520, 0.58], [540, 0.53],
            [560, 0.52], [580, 0.50], [600, 0.35], [620, 0.23],
            [640, 0.20], [680, 0.20], [720, 0.20], [760, 0.15],
            [800, 0.10],
        ],
    },
    "2383_characteristic": {
        "blue": [
            [-0.6, 0.05], [-0.3, 0.10], [0.0, 0.30], [0.3, 0.90],
            [0.6, 2.00], [0.9, 3.20], [1.2, 3.80], [1.5, 4.05],
            [1.7, 4.10], [1.9, 4.10], [2.1, 4.10], [2.3, 4.10],
        ],
        "green": [
            [-0.3, 0.05], [0.0, 0.10], [0.3, 0.30], [0.6, 0.90],
            [0.9, 2.00], [1.2, 3.20], [1.5, 3.80], [1.8, 4.05],
            [2.0, 4.10], [2.2, 4.10], [2.4, 4.10], [2.55, 4.10],
        ],
        "red": [
            [0.0, 0.05], [0.3, 0.10], [0.6, 0.30], [0.9, 0.90],
            [1.2, 2.00], [1.5, 3.20], [1.8, 3.80], [2.1, 4.05],
            [2.25, 4.10], [2.35, 4.10], [2.45, 4.10], [2.55, 4.10],
        ],
    },
    "2383_sensitivity": {
        "yellow_forming": [
            [350, 0.00], [370, -0.05], [385, -0.30], [400, -1.50],
            [420, -0.70], [440, 0.30], [455, 0.70], [470, 0.60],
            [485, -0.50], [495, -1.70], [505, -2.80],
        ],
        "magenta_forming": [
            [400, -1.50], [420, -1.30], [450, -1.20], [480, -1.00],
            [510, -0.90], [535, -0.35], [550, -0.20], [565, -0.80],
            [575, -1.70], [585, -2.80],
        ],
        "cyan_forming": [
            [575, -2.50], [590, -2.40], [610, -2.15], [630, -2.00],
            [650, -1.80], [670, -1.55], [690, -1.25], [705, -1.10],
            [720, -1.35], [725, -1.70], [730, -2.30], [735, -2.90],
        ],
    },
    "2383_dye_density": {
        "yellow": [
            [400, 0.20], [410, 0.22], [420, 0.25], [430, 0.30],
            [440, 0.27], [450, 0.25], [470, 0.70], [490, 0.80], [510, 0.65],
            [530, 0.30], [550, 0.10], [580, 0.02], [620, 0.00],
        ],
        "magenta": [
            [400, 0.02], [430, 0.03], [450, 0.03], [480, 0.10],
            [510, 0.40], [540, 0.75], [570, 0.85], [600, 0.60],
            [630, 0.25], [660, 0.05], [700, 0.00], [740, 0.00],
        ],
        "cyan": [
            [480, 0.00], [510, 0.01], [540, 0.03], [560, 0.08],
            [600, 0.30], [640, 0.65], [680, 0.95], [720, 1.08],
            [750, 0.95], [780, 0.60], [790, 0.48], [800, 0.35],
        ],
        "visual_neutral": [
            [400, 0.30], [420, 0.80],
            [450, 1.08], [480, 0.95], [510, 0.80], [540, 0.90],
            [570, 1.10], [600, 0.95], [630, 0.85], [660, 1.05],
            [690, 1.20], [720, 1.18], [750, 0.95], [780, 0.60],
            [800, 0.35],
        ],
    },
}


def _fit(pairs: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(list(pairs), dtype=np.float64)
    return np.polyfit(values[:, 0], values[:, 1], 1)


def _snap(
    image: Image.Image,
    axis: dict[str, object],
    guides: list[list[float]],
) -> list[list[int]]:
    gray = np.asarray(image.convert("L"))
    ink = gray < 128
    for x in axis["mask_x"]:
        ink[:, max(0, int(x) - 2) : int(x) + 3] = False
    for y in axis["mask_y"]:
        ink[max(0, int(y) - 2) : int(y) + 3, :] = False

    x_fit = _fit(axis["x_pairs"])
    y_fit = _fit(axis["y_pairs"])
    output: list[list[int]] = []
    for x_value, y_value in guides:
        expected_x = float(np.polyval(x_fit, x_value))
        expected_y = float(np.polyval(y_fit, y_value))
        x0 = max(0, int(round(expected_x)) - 4)
        x1 = min(ink.shape[1], int(round(expected_x)) + 5)
        y0 = max(0, int(round(expected_y)) - 60)
        y1 = min(ink.shape[0], int(round(expected_y)) + 61)
        yy, xx = np.nonzero(ink[y0:y1, x0:x1])
        if xx.size == 0:
            raise ValueError(f"no source ink near guide {(x_value, y_value)}")
        xx = xx + x0
        yy = yy + y0
        score = ((xx - expected_x) / 2.0) ** 2 + ((yy - expected_y) / 6.0) ** 2
        index = int(np.argmin(score))
        point = [int(xx[index]), int(yy[index])]
        if output and point[0] <= output[-1][0]:
            candidates = np.flatnonzero(xx > output[-1][0])
            if candidates.size == 0:
                raise ValueError(f"non-increasing trace near {(x_value, y_value)}")
            index = int(candidates[np.argmin(score[candidates])])
            point = [int(xx[index]), int(yy[index])]
        output.append(point)
    return output


def build(input_dir: Path, output_dir: Path, json_path: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": "Kodak VISION3 250D and VISION 2383 official graph rasters",
        "annotation_method": (
            "human-guided semantic value traces converted through frozen axes "
            "and snapped to exact black source pixels before numerical evaluation"
        ),
        "limitations": [
            "source-ink proximity proves graph transcription, not calibration",
            "labels and overlapping black curves require visual overlay review",
            "Status A/M density is not analytical dye amount",
            "printer and viewing spectra remain nuisance hypotheses",
        ],
        "images": {},
        "graph_axes": {},
        "curves": {},
    }
    colours = ["#ff2d2d", "#00a650", "#1769ff", "#ff8c00", "#8a2be2"]
    for family, curves in GUIDES.items():
        source_path = input_dir / f"{family}.png"
        source_image = Image.open(source_path).convert("RGB")
        overlay = source_image.copy()
        payload["images"][family] = str(source_path.relative_to(ROOT)).replace("\\", "/")
        payload["graph_axes"][family] = {
            "x_value_pixels": AXES[family]["x_pairs"],
            "y_value_pixels": AXES[family]["y_pairs"],
            "known_vertical_grid_pixels": AXES[family]["mask_x"],
            "known_horizontal_grid_pixels": AXES[family]["mask_y"],
        }
        family_points = {}
        draw = ImageDraw.Draw(overlay)
        for colour, (name, guides) in zip(colours, curves.items()):
            # Every semantic trace must bind to the immutable source raster.
            # Drawing one trace into the search image would allow later curves
            # to snap to our own annotation instead of official source ink.
            points = _snap(source_image, AXES[family], guides)
            family_points[name] = points
            draw.line([tuple(point) for point in points], fill=colour, width=2)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colour, width=1)
            draw.text((8, 8 + 16 * list(curves).index(name)), name, fill=colour)
        payload["curves"][family] = family_points
        overlay.save(output_dir / f"{family}_overlay.png")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    payload = build(args.input, args.output, args.json)
    print(args.json)
    for family, curves in payload["curves"].items():
        print(f"{family}: " + ", ".join(f"{name}={len(points)}" for name, points in curves.items()))


if __name__ == "__main__":
    main()
