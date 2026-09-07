"""Fixed learned colour candidate review, no training and no target pixels."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_paired_recipe_pilot import sha
from src.inference.style_safe_engine import render_resolved_safe_lab_rgb
from src.models.color_lut.conditioned_rgb_lut import ConditionedRGBLUT
from src.models.color_lut.lut import apply_lut


def validate_rows(rows, prior, count):
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError("Count/ID mismatch")
    if {r["raw_sha256"] for r in rows} & {r["sha256"] for r in prior}:
        raise ValueError("Prior transfer overlap")
    for r in rows:
        if r["rights_scope"] != "CC0_public_domain_internal_evaluation":
            raise ValueError("Source rights mismatch")
        if not r["decoded_path"].startswith("outputs/"):
            raise ValueError("Unexpected source path")


def simple_contrast(x, slope=1.15, pivot=0.5):
    return np.clip((x - pivot) * slope + pivot, 0, 1).astype(np.float32)


def diagnostics(x, y):
    if y.shape != x.shape or not np.isfinite(y).all() or y.min() < 0 or y.max() > 1:
        raise ValueError("Output invalid")
    interior = (x > 0) & (x < 1)
    return {
        "mean_absolute_rgb_change": float(np.mean(np.abs(y - x))),
        "new_exact_boundary_component_fraction": float(
            np.mean(interior & ((y == 0) | (y == 1)))
        ),
        "luma_quantiles": np.quantile(
            y @ np.array([0.2126, 0.7152, 0.0722]), [0, 0.01, 0.5, 0.99, 1]
        ).tolist(),
    }


def write_png(values, path):
    image = Image.fromarray(np.rint(values * 255).astype(np.uint8))
    with path.open("xb") as stream:
        image.save(stream, format="PNG")
    return sha(path)


def main():
    cfg = json.loads((ROOT / "configs/ai_classneg_photo_review_v1.json").read_text())
    manifest, checkpoint = ROOT / cfg["source_manifest"], ROOT / cfg["checkpoint"]
    if (
        sha(manifest) != cfg["source_manifest_sha256"]
        or sha(checkpoint) != cfg["checkpoint_sha256"]
    ):
        raise ValueError("Frozen identity drift")
    rows = json.loads(manifest.read_text())
    prior = json.loads((ROOT / cfg["prior_transfer_manifest"]).read_text())
    validate_rows(rows, prior, cfg["source_count"])
    # Verify every encoded source before reading any pixels.
    for row in rows:
        p = ROOT / row["decoded_path"]
        if p.stat().st_size != row["decoded_bytes"] or sha(p) != row["decoded_sha256"]:
            raise ValueError("Encoded source drift")
    profile = json.loads((ROOT / cfg["profile"]).read_text())
    for asset in profile["assets"]:
        if sha(ROOT / asset["path"]) != asset["sha256"]:
            raise ValueError("Baseline asset drift")
    stats = json.loads((ROOT / "configs/film_color_stats.json").read_text())["styles"][
        "portra_400"
    ]
    guards = json.loads((ROOT / "configs/color_guardrails.json").read_text())
    guards = {**guards["defaults"], **guards["styles"]["portra_400"]}
    out = ROOT / cfg["destination"]
    if out.resolve().drive.upper() != "P:":
        raise ValueError("Require P-backed destination")
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = ConditionedRGBLUT(conditional=False).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    with torch.no_grad():
        lut = model.predict_lut(
            torch.zeros(1, 3, 1, 1), torch.tensor([cfg["style_index"]])
        )
    owned = [
        "configs/ai_classneg_photo_review_v1.json",
        "scripts/audit_ai_classneg_photo_review.py",
        "src/models/color_lut/conditioned_rgb_lut.py",
        "src/models/color_lut/lut.py",
        "src/inference/style_safe_engine.py",
        cfg["profile"],
    ]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("Owned implementation must be committed")
    report = {
        "config": cfg,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_hashes": {p: sha(ROOT / p) for p in owned},
        "torch": torch.__version__,
        "prior_raw_overlap": 0,
        "training": False,
        "rows": [],
        "visual_decision": "pending",
    }
    (out / "execution_lock.json").write_text(json.dumps(report, indent=2))
    sheets = []
    for i, row in enumerate(rows):
        with Image.open(ROOT / row["decoded_path"]) as im:
            if im.mode != "RGB" or im.size != (row["width"], row["height"]):
                raise ValueError("Encoded input header mismatch")
            x = np.asarray(im, dtype=np.float32) / 255
        t = torch.from_numpy(x).permute(2, 0, 1)[None]
        with torch.no_grad():
            y = apply_lut(t, lut)[0].permute(1, 2, 0).numpy()
            repeat = apply_lut(t, lut)[0].permute(1, 2, 0).numpy()
        if not np.array_equal(y, repeat):
            raise ValueError("Repeat drift")
        values = {
            "identity": x,
            "learned": y,
            "safe_rich_portra": render_resolved_safe_lab_rgb(
                x,
                style="portra_400",
                style_statistics=stats,
                style_parameters=profile["style_parameters"]["portra_400"],
                guardrails=guards,
                seed=cfg["seed"],
            ),
            "simple_contrast": simple_contrast(
                x, cfg["simple_contrast_slope"], cfg["simple_contrast_pivot"]
            ),
        }
        record = {"id": row["id"], "source_sha256": row["decoded_sha256"], "arms": {}}
        sheet = Image.new("RGB", (1600, 340), "white")
        for j, (arm, image) in enumerate(values.items()):
            p = out / f"{i:02d}_{arm}.png"
            record["arms"][arm] = {
                **diagnostics(x, image),
                "sha256": write_png(image, p),
            }
            thumb = ImageOps.contain(
                Image.fromarray(np.rint(image * 255).astype(np.uint8)), (396, 300)
            )
            sheet.paste(thumb, (j * 400, 25))
            ImageDraw.Draw(sheet).text((j * 400 + 3, 5), f"{i:02d} {arm}", fill="black")
        ImageDraw.Draw(sheet).text((3, 326), row["id"], fill="black")
        sheet.save(out / f"{i:02d}_comparison.png")
        sheets.append(sheet)
        report["rows"].append(record)
        print(i, row["id"], record["arms"]["learned"], flush=True)
    for start in range(0, len(sheets), 3):
        subset = sheets[start : start + 3]
        sheet = Image.new("RGB", (1600, 340 * len(subset)), "white")
        for i, tile in enumerate(subset):
            sheet.paste(tile, (0, i * 340))
        sheet.save(out / f"sheet_{start // 3:02d}.png")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("REPORT", sha(out / "report.json"), flush=True)


if __name__ == "__main__":
    main()
