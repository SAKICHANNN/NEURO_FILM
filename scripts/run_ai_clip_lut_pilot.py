"""Fixed-budget semantic-learning development baseline, no target RGB teacher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def transform(x, p, arm):
    if arm == "lut":
        grid = x.permute(0, 2, 3, 1).unsqueeze(1) * 2 - 1
        delta = F.grid_sample(
            p, grid, align_corners=True, padding_mode="border"
        ).squeeze(2)
    else:
        delta = p[:, :3] + p[:, 3:] * (x - 0.5)
    z = torch.logit(x.clamp(1e-5, 1 - 1e-5)) + 2 * torch.tanh(delta)
    y = torch.sigmoid(z)
    return torch.where((x == 0) | (x == 1), x, y)


def main():
    cfg = json.loads((ROOT / "configs/ai_clip_lut_pilot_v1.json").read_text())
    out = ROOT / "outputs/ai_clip_lut_pilot_v1"
    if out.exists():
        raise RuntimeError("Refuse output overwrite")
    path = ROOT / cfg["model_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["model_sha256"]:
        raise RuntimeError("Model identity mismatch")
    rows = json.loads((ROOT / cfg["source_manifest"]).read_text())
    ids = cfg["train_rows"] + cfg["transfer_rows"]
    sources = []
    source_hashes = []
    for i in ids:
        r = rows[i]
        if r["license"] != "CC0/Public Domain":
            raise RuntimeError("Source rights differ")
        f = Path(r["before"])
        source_hashes.append(hashlib.sha256(f.read_bytes()).hexdigest())
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        im.thumbnail((768, 768))
        sources.append(
            torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()[None] / 255
        )
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    model, _ = clip.load(str(path), device="cuda", jit=False)
    model = model.float().eval().requires_grad_(False)
    with torch.no_grad():
        pos = F.normalize(
            model.encode_text(clip.tokenize(cfg["positive"]).cuda()), dim=-1
        ).mean(0)
        neg = F.normalize(
            model.encode_text(clip.tokenize(cfg["negative"]).cuda()), dim=-1
        ).mean(0)
        direction = F.normalize(pos - neg, dim=0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda")[
        None, :, None, None
    ]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda")[
        None, :, None, None
    ]
    train = [
        F.interpolate(x.cuda(), (224, 224), mode="bilinear", align_corners=False)
        for x in sources[:6]
    ]
    out.mkdir()
    report = {"config": cfg, "source_hashes": source_hashes, "arms": {}}
    for arm in cfg["arms"]:
        p = torch.nn.Parameter(
            torch.zeros(
                (1, 3, 17, 17, 17) if arm == "lut" else (1, 6, 1, 1), device="cuda"
            )
        )
        opt = torch.optim.Adam([p], lr=cfg["lr"])
        losses = []
        for step in range(cfg["steps"]):
            x = train[step % len(train)]
            y = transform(x, p, arm)
            feature = F.normalize(model.encode_image((y - mean) / std), dim=-1)
            loss = -(feature * direction).sum() + 0.05 * (y - x).square().mean()
            if arm == "lut":
                loss = loss + 0.02 * sum(
                    p.diff(dim=d).square().mean() for d in (2, 3, 4)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if step % 20 == 0:
                print(arm, step, losses[-1], flush=True)
        np.save(out / f"{arm}_parameters.npy", p.detach().cpu().numpy())
        for j, x in enumerate(sources):
            with torch.no_grad():
                y = transform(x.cuda(), p, arm).cpu()
            if not torch.isfinite(y).all() or y.min() < 0 or y.max() > 1:
                raise RuntimeError("Invalid output")
            a = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(a).save(out / f"{ids[j]:02d}_{arm}.png")
        report["arms"][arm] = {"losses": losses}
    for j, x in enumerate(sources):
        a = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(a).save(out / f"{ids[j]:02d}_original.png")
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
