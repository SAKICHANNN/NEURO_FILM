"""Fixed three-photo development transfer check; no optimization or selection."""

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_paired_recipe_pilot import save, sha
from scripts.run_ai_photo_distribution_pilot import image_tensor
from src.models.color_lut.conditioned_triangular import ConditionedTriangular


def main():
    run = ROOT / "outputs/ai_paired_recipe_pilot_v1/run_icc_v2"
    if (
        sha(run / "report.json")
        != "84b620a5bdaa5b675c2a2f4efecf4535000fc450c52f0e3270bf3e16bf6be9e8"
    ):
        raise ValueError("Training report changed")
    freeze = json.loads((run / "checkpoint_freeze.json").read_text())
    report = json.loads((run / "report.json").read_text())
    out = run / "transfer"
    if out.resolve().drive.upper() != "P:":
        raise ValueError("Require P-backed output")
    out.mkdir(exist_ok=False)
    rows = json.loads(
        (
            ROOT
            / "outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/manifest.json"
        ).read_text()
    )
    prior = json.loads(
        (ROOT / "outputs/ai_structural_photo_pilot_v1/report.json").read_text()
    )
    images, sources = [], []
    for i in (6, 7, 8):
        if rows[i]["license"] != "CC0/Public Domain":
            raise ValueError("Wrong development source")
        path = Path(rows[i]["before"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != prior["source_hashes"][i]:
            raise ValueError("Source changed")
        images.append(image_tensor(path).cuda())
        sources.append(digest)
        save(images[-1], out / f"{i:02d}_input.png")
    records = []
    for arm in ("global", "conditional", "wrong_target"):
        if (
            sha(run / f"{arm}.pt") != freeze[arm]
            or freeze[arm] != report["arms"][arm]["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint changed")
        model = ConditionedTriangular(arm != "global")
        model.load_state_dict(
            torch.load(run / f"{arm}.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        model.cuda().eval()
        for i, x in zip((6, 7, 8), images, strict=True):
            for s, name in enumerate(report["config"]["styles"]):
                with torch.no_grad():
                    y = model(x, torch.tensor([s], device="cuda"))
                path = out / f"{i:02d}_{name}_{arm}.png"
                save(y, path)
                records.append({"path": path.name, "sha256": sha(path)})
    (out / "report.json").write_text(
        json.dumps(
            {
                "source_sha256": sources,
                "outputs": records,
                "claim": "previously used development transfer only",
            },
            indent=2,
        )
    )
    print("COMPLETE", sha(out / "report.json"))


if __name__ == "__main__":
    main()
