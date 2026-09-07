"""Learn a bounded colour operator from unpaired photo distributions."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_clip_lut_pilot import transform


def image_tensor(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((768, 768))
        return (
            torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None] / 255
        )


def main():
    cfg = json.loads((ROOT / "configs/ai_photo_distribution_pilot_v1.json").read_text())
    out = ROOT / "outputs/ai_photo_distribution_pilot_v1"
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise RuntimeError("Require new P-backed output")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    manifest = ROOT / cfg["reference_manifest"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == cfg["reference_sha256"]
    refs = json.loads(manifest.read_text())
    ref_samples = []
    for row in refs["rows"]:
        p = manifest.parent / row["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"]
        x = image_tensor(p).flatten(2)[0].T
        ref_samples.append(x[torch.randperm(len(x))[: cfg["samples_per_image"]]])
    rows = json.loads((ROOT / cfg["source_manifest"]).read_text())
    ids = cfg["train_rows"] + cfg["transfer_rows"]
    images, source_hashes = [], []
    for i in ids:
        assert rows[i]["license"] == "CC0/Public Domain"
        path = Path(rows[i]["before"])
        source_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        images.append(image_tensor(path))
    samples = []
    for x in images[: len(cfg["train_rows"])]:
        pixels = x.flatten(2)[0].T
        samples.append(pixels[torch.randperm(len(pixels))[: cfg["samples_per_image"]]])
    x = torch.cat(samples).T[None, :, None, :].cuda()
    directions = F.normalize(torch.randn(3, cfg["projections"]), dim=0).cuda()
    q = torch.linspace(0, 1, cfg["quantiles"], device="cuda")
    target = torch.quantile(torch.cat(ref_samples).cuda() @ directions, q, dim=0)
    out.mkdir()
    report = {"config": cfg, "source_hashes": source_hashes, "arms": {}}
    for arm in ("lut", "simple"):
        n = cfg["lattice"]
        p = torch.nn.Parameter(
            torch.zeros(
                (1, 3, n, n, n) if arm == "lut" else (1, 6, 1, 1), device="cuda"
            )
        )
        opt = torch.optim.Adam([p], lr=cfg["lr"])
        losses = []
        for step in range(cfg["steps"]):
            y = transform(x, p, arm)
            projected = y.flatten(2)[0].T @ directions
            distribution = (
                (torch.quantile(projected, q, dim=0) - target).square().mean()
            )
            loss = distribution + cfg["identity_weight"] * (y - x).square().mean()
            if arm == "lut":
                loss = loss + cfg["smoothness_weight"] * sum(
                    p.diff(dim=d).square().mean() for d in (2, 3, 4)
                )
            opt.zero_grad()
            loss.backward()
            if not torch.isfinite(p.grad).all():
                raise RuntimeError("nonfinite gradients")
            opt.step()
            losses.append(
                {
                    "total": float(loss.detach()),
                    "distribution": float(distribution.detach()),
                }
            )
            if step % 50 == 0:
                print(arm, step, losses[-1], flush=True)
        np.save(out / f"{arm}_parameters.npy", p.detach().cpu().numpy())
        for i, source in zip(ids, images, strict=True):
            with torch.no_grad():
                y = transform(source.cuda(), p, arm).cpu()
            assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
            a = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(a).save(out / f"{i:02d}_{arm}.png")
        report["arms"][arm] = {
            "losses": losses,
            "parameter_norm": float(p.detach().norm()),
        }
    for i, source in zip(ids, images, strict=True):
        a = (source[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(a).save(out / f"{i:02d}_original.png")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
