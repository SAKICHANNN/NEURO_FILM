"""Descriptive fixed-grid gain diagnostic; never a promotion or training gate."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_vcg_reference import checked_json, read_rgb, sha

REPORT = ROOT / "outputs/ai_vcg_reference_development_v3/report.json"
REPORT_SHA = "e1ff75bbd54333eba21e5dd0658c15651fbdcf64f0efafd624b96dfdb9c3284a"
LUT_SHAS = [
    "056dbbfea40648714c98e879e39b5bb2c72c25f57f4704bf0c9d2a5f5fb40983",
    "4ed163a6c6e0dcb76805b6b69560ed611fad9e6ed1908dfc8d789e559f371d63",
    "e68b1e2bda54f1877a6224b3162c708a6bfd51c243cdae0f51a0084d1b92be84",
]


def official_functions(report):
    path = ROOT / "data/ai_models/video_color_grading_v1/source/utils/util.py"
    if sha(path) != report["config"]["source_hashes"]["utils/util.py"]:
        raise ValueError("preprocessing source drift")
    nodes = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"vars", "transfer", "preprocess"}
    ]
    if len(nodes) != 3:
        raise ValueError("preprocessing functions")
    scope = {"np": np, "Image": Image}
    exec(  # noqa: S102 - Three reviewed exact hash-bound official functions.
        compile(ast.Module(nodes, []), "official_preprocess", "exec"), scope
    )
    return scope


def interpolate(cube, rgb):
    """Pillow's B,G,R node ordering, continuous trilinear counterpart."""
    points = np.clip(np.asarray(rgb, dtype=np.float64), 0, 1) * 15
    lower = np.minimum(np.floor(points).astype(int), 14)
    frac = points - lower
    result = np.zeros_like(points)
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                weight = np.prod(np.where(np.array([r, g, b]), frac, 1 - frac), axis=-1)
                result += (
                    weight[..., None]
                    * cube[lower[..., 2] + b, lower[..., 1] + g, lower[..., 0] + r]
                )
    return result


def jacobians(cube, points):
    columns = []
    for channel in range(3):
        low, high = points.copy(), points.copy()
        low[:, channel] = np.maximum(0, low[:, channel] - 1e-4)
        high[:, channel] = np.minimum(1, high[:, channel] + 1e-4)
        columns.append(
            (interpolate(cube, high) - interpolate(cube, low))
            / (high[:, channel] - low[:, channel])[:, None]
        )
    return np.stack(columns, axis=-1)


def describe(jac):
    gain = np.linalg.svd(jac, compute_uv=False)[..., 0]
    return {
        "largest_singular_value_p50_p95_max": np.quantile(
            gain, [0.5, 0.95, 1]
        ).tolist(),
        "negative_determinant_fraction": float(np.mean(np.linalg.det(jac) < 0)),
    }


def png_sha(image):
    import hashlib

    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def main():
    report = checked_json(REPORT, REPORT_SHA)
    scope = official_functions(report)
    output = ROOT / "outputs/ai_vcg_stage_gain_diagnostic_v1"
    if output.exists() or output.resolve().drive.upper() != "P:":
        raise ValueError("fresh P-backed destination required")
    rows = []
    for i, row in enumerate(report["rows"]):
        pair = row["pair"]
        source, ref = Path(pair["content_path"]), Path(pair["reference_path"])
        if (
            sha(source) != pair["content_sha256"]
            or sha(ref) != pair["reference_sha256"]
        ):
            raise ValueError("photo identity drift")
        before = np.asarray(read_rgb(source))
        style = np.asarray(read_rgb(ref).resize((512, 512), Image.Resampling.BICUBIC))
        corrected, _ = scope["preprocess"](before[None], style, 512, False)
        lut_path = REPORT.parent / f"{i:02d}_lut.npy"
        if sha(lut_path) != LUT_SHAS[i]:
            raise ValueError("LUT identity drift")
        lut = np.clip(np.load(lut_path, allow_pickle=False), 0, 1)
        image = Image.fromarray(corrected[0])
        assert png_sha(image) == row["images"]["precorrection"]
        assert (
            png_sha(image.filter(ImageFilter.Color3DLUT(16, lut.flatten())))
            == row["images"]["learned"]
        )
        small = np.array([np.array(Image.fromarray(before).resize((256, 256)))])
        variables = scope["vars"](small, style)
        transferred = scope["transfer"](before, style, variables)
        normalized_matrix = variables[4] * 255 / np.ptp(transferred)
        eigenvalues = np.linalg.eigvalsh(variables[0])
        ys = np.linspace(0, before.shape[0] - 1, 48).astype(int)
        xs = np.linspace(0, before.shape[1] - 1, 48).astype(int)
        points = corrected[0][np.ix_(ys, xs)].reshape(-1, 3).astype(float) / 255
        jac = jacobians(lut.reshape(16, 16, 16, 3), points)
        rows.append(
            {
                "id": i,
                "precorrection_and_learned_png_exact": True,
                "source_covariance_eigenvalues": eigenvalues.tolist(),
                "source_covariance_condition": float(eigenvalues[-1] / eigenvalues[0]),
                "precorrection_singular_values": np.linalg.svd(
                    normalized_matrix, compute_uv=False
                ).tolist(),
                "lut_at_source_grid": describe(jac),
                "composition_at_source_grid": describe(jac @ normalized_matrix),
                "sample_count": len(points),
                "lut_file_sha256": LUT_SHAS[i],
            }
        )
    output.mkdir()
    result = {
        "scope": "Post-result descriptive local derivative diagnosis, no quality gate",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "parent_report_sha256": REPORT_SHA,
        "rows": rows,
        "product_promotion": False,
        "caveat": "Continuous trilinear local derivatives omit uint8 rounding; fixed image-grid samples are not full RGB-cube certification or proof of visual cause",
    }
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
