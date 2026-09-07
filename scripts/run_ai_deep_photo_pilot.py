"""Fixed deep-photo supervision experiment; output RGB remains a pointwise LUT."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_clip_lut_pilot import transform
from scripts.run_ai_photo_distribution_pilot import image_tensor


def checked_json(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("manifest mismatch")
    return json.loads(data)


def moments(x):
    return x.mean((2, 3)), x.std((2, 3), correction=0)


def main(config="configs/ai_deep_photo_pilot_v1.json", structural=False):
    cfg = json.loads((ROOT / config).read_text())
    out = ROOT / cfg.get("output", "outputs/ai_deep_photo_pilot_v1")
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise RuntimeError("Require new P-backed output")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    feature_path = ROOT / cfg["feature_manifest"]
    feature = checked_json(feature_path, cfg["feature_sha256"])
    for row in feature["files"]:
        assert (
            hashlib.sha256((feature_path.parent / row["path"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
    spec = importlib.util.spec_from_file_location(
        "locked_vgg", feature_path.parent / "net.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(
        feature_path.parent / "models/vgg_normalised.pth",
        map_location="cpu",
        weights_only=True,
    )
    module.vgg.load_state_dict(state, strict=True)
    encoder = module.Net(module.vgg).cuda().eval().requires_grad_(False)

    def resize(x):
        return F.interpolate(
            x.cuda(), (cfg["size"], cfg["size"]), mode="bilinear", align_corners=False
        )

    manifest = ROOT / cfg["reference_manifest"]
    refs = checked_json(manifest, cfg["reference_sha256"])
    reference_moments = []
    with torch.no_grad():
        for row in refs["rows"]:
            p = manifest.parent / row["path"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"]
            reference_moments.append(
                [
                    moments(f)
                    for f in encoder.encode_with_intermediate(resize(image_tensor(p)))
                ]
            )
    targets = [
        (
            torch.stack([r[k][0] for r in reference_moments]).mean(0),
            torch.stack([r[k][1] for r in reference_moments]).mean(0),
        )
        for k in range(4)
    ]
    rows = json.loads((ROOT / cfg["source_manifest"]).read_text())
    ids = cfg["train_rows"] + cfg["transfer_rows"]
    images, hashes = [], []
    for i in ids:
        assert rows[i]["license"] == "CC0/Public Domain"
        path = Path(rows[i]["before"])
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        images.append(image_tensor(path))
    train = [resize(x) for x in images[: len(cfg["train_rows"])]]
    with torch.no_grad():
        contents = [encoder.encode(x).detach() for x in train]
    out.mkdir()
    report = {"config": cfg, "source_hashes": hashes, "arms": {}}
    for arm in (("triangular", "simple") if structural else ("lut", "simple")):
        n = cfg["lattice"]
        p = torch.nn.Parameter(
            torch.zeros(
                (1, 3, n, n, n) if arm == "lut" else (1, 6, 1, 1), device="cuda"
            )
        )
        model = None
        if arm == "triangular":
            from src.models.color_lut.triangular_photo import TriangularPhoto

            model = TriangularPhoto().cuda()
            p = model.parameters_raw

        def apply(x, model=model, arm=arm, p=p):
            return model(x) if arm == "triangular" else transform(x, p, arm)

        opt = torch.optim.Adam([p], lr=cfg["lr"])
        losses = []
        for step in range(cfg["steps"]):
            i = step % len(train)
            y = apply(train[i])
            feats = encoder.encode_with_intermediate(y)
            style = y.new_zeros(())
            for f, (mean, std) in zip(feats, targets, strict=True):
                m, s = moments(f)
                style = style + (
                    (m - mean).square().mean() + (s - std).square().mean()
                ) / (mean.square().mean() + std.square().mean() + 1e-6)
            content = (feats[-1] - contents[i]).square().mean() / (
                contents[i].square().mean() + 1e-6
            )
            loss = cfg["style_weight"] * style + cfg["content_weight"] * content
            if arm == "lut":
                loss = loss + cfg["smoothness_weight"] * sum(
                    p.diff(dim=d).square().mean() for d in (2, 3, 4)
                )
            opt.zero_grad()
            loss.backward()
            assert torch.isfinite(p.grad).all()
            opt.step()
            losses.append(
                {
                    "style": float(style.detach()),
                    "content": float(content.detach()),
                    "total": float(loss.detach()),
                }
            )
            if step % 30 == 0:
                print(arm, step, losses[-1], flush=True)
        np.save(out / f"{arm}_parameters.npy", p.detach().cpu().numpy())
        for i, x in zip(ids, images, strict=True):
            with torch.no_grad():
                y = apply(x.cuda()).cpu()
            assert torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
            Image.fromarray(
                (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            ).save(out / f"{i:02d}_{arm}.png")
        report["arms"][arm] = {
            "losses": losses,
            "parameter_norm": float(p.detach().norm()),
        }
    for i, x in zip(ids, images, strict=True):
        Image.fromarray(
            (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        ).save(out / f"{i:02d}_original.png")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
