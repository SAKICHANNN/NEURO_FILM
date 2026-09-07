"""New supervised development roles; never accepts historical lockbox inputs."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.color_lut.conditioned_triangular import ConditionedTriangular


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(cfg):
    source = ROOT / cfg["source_manifest"]
    if sha(source) != cfg["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    inventory = {
        r["path"]: r
        for r in (
            json.loads(line)
            for line in (ROOT / cfg["inventory"]).read_text().splitlines()
        )
    }
    chosen, groups = [], set()
    for row in sorted(rows, key=lambda r: r["sha256"]):
        if (
            row["research_pool"] != "source_train"
            or row["distributed_split"] != "train"
        ):
            raise ValueError("Forbidden role")
        if row["duplicate_cluster_id"] in groups:
            continue
        groups.add(row["duplicate_cluster_id"])
        files = [row] + [
            inventory[f"train/{style}/{Path(row['path']).name}"]
            for style in cfg["styles"]
        ]
        chosen.append(
            {
                "old_content_id": row["content_id"],
                "group": row["duplicate_cluster_id"],
                "role": "paired_fit"
                if len(chosen) < cfg["fit_count"]
                else "paired_development_evaluation",
                "files": [
                    {key: f[key] for key in ("path", "sha256", "bytes")} for f in files
                ],
            }
        )
        if len(chosen) == cfg["fit_count"] + cfg["evaluation_count"]:
            break
    if len(chosen) != cfg["fit_count"] + cfg["evaluation_count"]:
        raise ValueError("Insufficient groups")
    return {
        "config": cfg,
        "inventory_sha256": sha(ROOT / cfg["inventory"]),
        "rows": chosen,
    }


def decode(root, row, size=None):
    path = root / row["path"]
    if not row["path"].startswith("train/") or sha(path) != row["sha256"]:
        raise ValueError("Forbidden path or changed payload")
    with Image.open(path) as im:
        profile = im.info.get("icc_profile")
        if not profile:
            raise ValueError(f"Missing RGB profile: {row['path']}")
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        profile_name = ImageCms.getProfileName(source_profile).strip()
        if profile_name not in (
            "Adobe RGB (1998)",
            "sRGB built-in",
            "sRGB IEC61966-2.1",
        ):
            raise ValueError(f"Unsupported RGB profile: {profile_name}")
        if im.mode != "RGB":
            raise ValueError("Expected RGB")
        converted = ImageCms.profileToProfile(
            im,
            source_profile,
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
        )
        a = np.asarray(converted, dtype=np.float32).copy() / 255
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    return (
        F.interpolate(x, (size, size), mode="bilinear", align_corners=False)
        if size
        else x
    )


def save(x, path):
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Nonfinite/out-of-range render")
    a = x.detach().cpu()[0].permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(a * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/ai_paired_recipe_pilot_v1.json").read_text())
    out = ROOT / cfg["destination"]
    if out.resolve().drive.upper() != "P:":
        raise ValueError("Require P-backed output")
    if args.freeze:
        out.mkdir(parents=True, exist_ok=False)
        with (out / "manifest.json").open("x") as f:
            json.dump(freeze(cfg), f, indent=2)
        print("MANIFEST", sha(out / "manifest.json"))
        return
    if sha(out / "manifest.json") != cfg["paired_manifest_sha256"]:
        raise ValueError("Paired manifest changed")
    manifest = json.loads((out / "manifest.json").read_text())
    for key in manifest["config"]:
        if manifest["config"][key] != cfg[key]:
            raise ValueError("Frozen configuration changed")
    run = out / "run_icc_v2"
    run.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = ROOT / cfg["root"]
    fit = [row for row in manifest["rows"] if row["role"] == "paired_fit"]
    xs, ts = [], []
    for row in fit:
        xs.append(decode(root, row["files"][0], cfg["size"]))
        ts.append(torch.cat([decode(root, f, cfg["size"]) for f in row["files"][1:]]))
    x = torch.cat(xs).cuda()
    target = torch.stack(ts).cuda()
    report = {
        "config": cfg,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "arms": {},
    }
    models = {}
    for arm in ("global", "conditional", "wrong_target"):
        torch.manual_seed(cfg["seed"])
        model = ConditionedTriangular(arm != "global").cuda()
        opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        generator = torch.Generator().manual_seed(cfg["seed"])
        losses = []
        for step in range(cfg["steps"]):
            ids = torch.randint(len(fit), (cfg["batch"],), generator=generator).cuda()
            styles = torch.randint(3, (cfg["batch"],), generator=generator).cuda()
            target_ids = (ids + 1) % len(fit) if arm == "wrong_target" else ids
            y = model(x[ids], styles)
            loss = (y - target[target_ids, styles]).abs().mean()
            opt.zero_grad()
            loss.backward()
            if any(
                not torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise ValueError("Nonfinite gradient")
            opt.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(arm, step, losses[-1], flush=True)
        torch.save(model.cpu().state_dict(), run / f"{arm}.pt")
        report["arms"][arm] = {
            "losses": losses,
            "checkpoint_sha256": sha(run / f"{arm}.pt"),
            "evaluation": [],
        }
        models[arm] = model.cuda().eval()
    # All three checkpoints exist before any development-evaluation pixel decode.
    (run / "checkpoint_freeze.json").write_text(
        json.dumps(
            {k: v["checkpoint_sha256"] for k, v in report["arms"].items()}, indent=2
        )
    )
    evaluations = [
        row
        for row in manifest["rows"]
        if row["role"] == "paired_development_evaluation"
    ]
    for i, row in enumerate(evaluations):
        source = decode(root, row["files"][0]).cuda()
        if i < 3:
            save(source, run / f"eval_{i:02d}_input.png")
        for s, name in enumerate(cfg["styles"]):
            truth = decode(root, row["files"][s + 1]).cuda()
            identity = float((source - truth).abs().mean())
            if i < 3:
                save(truth, run / f"eval_{i:02d}_{name}_target.png")
            for arm, model in models.items():
                with torch.no_grad():
                    y = model(source, torch.tensor([s], device="cuda"))
                error = float((y - truth).abs().mean())
                report["arms"][arm]["evaluation"].append(
                    {
                        "content": row["old_content_id"],
                        "style": name,
                        "l1": error,
                        "identity_l1": identity,
                    }
                )
                if i < 3:
                    save(y, run / f"eval_{i:02d}_{name}_{arm}.png")
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", sha(run / "report.json"), flush=True)


if __name__ == "__main__":
    main()
