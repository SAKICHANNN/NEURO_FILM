"""Single fixed published no-colour-correction ablation on consumed pairs."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_ai_vcg_stage_gain import REPORT, REPORT_SHA
from scripts.run_ai_vcg_reference import checked_json, initialize, read_rgb, sha


def main():
    parent = checked_json(REPORT, REPORT_SHA)
    cfg = parent["config"]
    out = ROOT / "outputs/ai_vcg_ncc_ablation_v1"
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise ValueError("fresh P-backed output required")
    owned = ["scripts/run_ai_vcg_ncc_ablation.py", "scripts/run_ai_vcg_reference.py"]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("commit owned scripts before execution")
    out.mkdir()
    report = {
        "parent_report_sha256": REPORT_SHA,
        "scope": "Consumed-development one-variable ncc=True ablation; no promotion",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "stage": "loading",
        "rows": [],
        "product_promotion": False,
    }
    started = time.monotonic()
    try:
        predict, preprocess = initialize(cfg)
        import torch

        torch.cuda.reset_peak_memory_stats()
        for i, row in enumerate(parent["rows"]):
            if time.monotonic() - started > 900:
                raise TimeoutError("ablation wall-time cap")
            pair = row["pair"]
            source, ref = Path(pair["content_path"]), Path(pair["reference_path"])
            if (
                sha(source) != pair["content_sha256"]
                or sha(ref) != pair["reference_sha256"]
            ):
                raise ValueError("source drift")
            content = read_rgb(source)
            style = np.asarray(
                read_rgb(ref).resize((512, 512), Image.Resampling.BICUBIC)
            )
            unchanged, small = preprocess(np.asarray(content)[None], style, 512, True)
            if not np.array_equal(unchanged[0], np.asarray(content)):
                raise ValueError("ncc unexpectedly changed source")
            report["stage"] = f"pair_{i}"
            lut = predict(small, style)
            output = content.filter(
                ImageFilter.Color3DLUT(16, np.clip(lut, 0, 1).flatten())
            )
            target = out / f"{i:02d}_ncc.png"
            with target.open("xb") as handle:
                output.save(handle, format="PNG")
            with (out / f"{i:02d}_lut.npy").open("xb") as handle:
                np.save(handle, lut, allow_pickle=False)
            arms = {"identity": content}
            for name in ("simple", "precorrection", "learned"):
                old = REPORT.parent / f"{i:02d}_{name}.png"
                if sha(old) != row["images"][name]:
                    raise ValueError("parent comparison drift")
                arms[name] = read_rgb(old)
            arms["learned_no_precorrection"] = output
            sheet = Image.new("RGB", (480 * len(arms), 390), "white")
            for n, (name, image) in enumerate(arms.items()):
                small_image = image.copy()
                small_image.thumbnail((480, 360))
                sheet.paste(small_image, (n * 480, 30))
                ImageDraw.Draw(sheet).text((n * 480 + 5, 8), name, fill="black")
            sheet.save(out / f"{i:02d}_comparison.png")
            report["rows"].append(
                {
                    "id": i,
                    "output_sha256": sha(target),
                    "lut_sha256": sha(out / f"{i:02d}_lut.npy"),
                    "lut_minimum": float(lut.min()),
                    "lut_maximum": float(lut.max()),
                    "lut_out_of_range_fraction": float(np.mean((lut < 0) | (lut > 1))),
                }
            )
            print(f"pair {i} complete", flush=True)
        report["stage"] = "rendered_pending_visual_review"
        report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        with (out / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
