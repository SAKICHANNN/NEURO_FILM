"""Consumed-development target diagnosis; no fitting or product promotion."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_paired_recipe_pilot import decode, sha
from src.models.color_lut.conditioned_rgb_lut import ConditionedRGBLUT

MANIFEST = "outputs/ai_conditioned_rgb_lut_v1/manifest.json"
MANIFEST_SHA = "c65c6b6db647a637fe512508bcc4656bb8352f00f482c967ac44594bc5314326"
CHECKPOINT = "outputs/ai_conditioned_rgb_lut_v1/run/global.pt"
CHECKPOINT_SHA = "bf0dcb11c1b6c5f7286b056e8610f27d43d980a56741093f8c0aa3d535d0cdd9"
PARENT = "outputs/ai_conditioned_rgb_lut_v1/run/report.json"
PARENT_SHA = "2c14e1f3e23015c85fffa994b7b4c2e49bb7e47d9aa965b18d5d4591e6c62436"


def select_rows(manifest):
    if manifest["config"]["styles"] != ["Cinema", "ClassNeg", "Velvia"]:
        raise ValueError("Style order drift")
    rows = [r for r in manifest["rows"] if r["role"] == "paired_development_evaluation"]
    if len(rows) != 16 or len({r["group"] for r in rows}) != 16:
        raise ValueError("Expected exactly16 development groups")
    fit = {r["group"] for r in manifest["rows"] if r["role"] == "paired_fit"}
    if fit & {r["group"] for r in rows}:
        raise ValueError("Fit overlap")
    for r in rows:
        for index, folder in ((0, "input"), (2, "ClassNeg")):
            path = Path(r["files"][index]["path"])
            if path.parts[:2] != ("train", folder) or ".." in path.parts:
                raise ValueError("Unexpected role path")
    return rows


def main():
    for path, expected in (
        (MANIFEST, MANIFEST_SHA),
        (CHECKPOINT, CHECKPOINT_SHA),
        (PARENT, PARENT_SHA),
    ):
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen identity changed")
    manifest = json.loads((ROOT / MANIFEST).read_text())
    rows = select_rows(manifest)
    source_root = ROOT / manifest["config"]["root"]
    for r in rows:
        for index in (0, 2):
            f = r["files"][index]
            path = source_root / f["path"]
            if path.stat().st_size != f["bytes"] or sha(path) != f["sha256"]:
                raise ValueError("Payload identity changed")
    owned = [
        "scripts/audit_ai_classneg_supervision.py",
        "scripts/run_ai_paired_recipe_pilot.py",
        "src/models/color_lut/conditioned_rgb_lut.py",
        "src/models/color_lut/lut.py",
    ]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("Relevant source must be committed")
    out = ROOT / "outputs/ai_classneg_supervision_diagnostic_v1"
    if out.resolve().drive.upper() != "P:":
        raise ValueError("Require P-backed artifacts")
    out.mkdir(exist_ok=False)
    report = {
        "scope": "post-result consumed-development diagnosis; not independent evidence",
        "manifest_sha256": MANIFEST_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "parent_report_sha256": PARENT_SHA,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "implementation_sha256": {p: sha(ROOT / p) for p in owned},
        "worktree": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "torch": torch.__version__,
        "arms": ["input", "digital_recipe_target", "learned_global"],
        "rows": [],
        "training_steps": 0,
        "product_promotion": False,
    }
    (out / "execution_lock.json").write_text(json.dumps(report, indent=2))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = ConditionedRGBLUT(conditional=False).eval()
    model.load_state_dict(
        torch.load(ROOT / CHECKPOINT, map_location="cpu", weights_only=True),
        strict=True,
    )
    parent = json.loads((ROOT / PARENT).read_text())
    old = {
        r["content"]: r["l1"]
        for r in parent["arms"]["global"]["evaluation"]
        if r["style"] == "ClassNeg"
    }
    with torch.no_grad():
        for i, r in enumerate(rows):
            x = decode(source_root, r["files"][0])
            target = decode(source_root, r["files"][2])
            y = model(x, torch.tensor([1]))
            if x.shape != target.shape or not torch.isfinite(y).all():
                raise ValueError("Invalid representation")
            l1 = float((y - target).abs().mean())
            # Only check that CPU replay is the same mathematical evaluation.
            if abs(l1 - old[r["old_content_id"]]) > 1e-6:
                raise ValueError("Parent numerical replay mismatch")
            h, w = x.shape[-2:]
            canvas = Image.new("RGB", (3 * w, h + 24), (240, 240, 240))
            draw = ImageDraw.Draw(canvas)
            for col, (label, value) in enumerate(zip(report["arms"], (x, target, y))):
                a = value[0].permute(1, 2, 0).numpy()
                if a.min() < 0 or a.max() > 1:
                    raise ValueError("Out-of-range output")
                canvas.paste(
                    Image.fromarray(np.rint(a * 255).astype(np.uint8)), (col * w, 24)
                )
                draw.text((col * w + 5, 5), f"{i:02d} {label}", fill="black")
            path = out / f"{i:02d}_comparison.png"
            with path.open("xb") as stream:
                canvas.save(stream, format="PNG")
            report["rows"].append(
                {
                    "index": i,
                    "content": r["old_content_id"],
                    "group": r["group"],
                    "source_files": [r["files"][j] for j in (0, 2)],
                    "shape": list(x.shape),
                    "prediction_target_l1": l1,
                    "input_target_l1": float((x - target).abs().mean()),
                    "prediction_input_l1": float((x - y).abs().mean()),
                    "parent_l1_absolute_difference": abs(l1 - old[r["old_content_id"]]),
                    "comparison": path.name,
                    "comparison_sha256": sha(path),
                }
            )
            print(i, r["old_content_id"], l1, flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", sha(out / "report.json"), flush=True)


if __name__ == "__main__":
    main()
