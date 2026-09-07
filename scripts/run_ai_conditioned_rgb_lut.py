"""Fixed internal digital-recipe development; never loads official test pixels."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_ai_paired_operator_capacity import (
    cell_centre_determinants,
    curvature,
)
from scripts.run_ai_paired_recipe_pilot import decode, freeze, save, sha
from scripts.run_ai_photo_distribution_pilot import image_tensor
from src.models.color_lut.conditioned_rgb_lut import ConditionedRGBLUT
from src.models.color_lut.conditioned_triangular import ConditionedTriangular
from src.models.color_lut.lut import apply_lut


def new_manifest(cfg):
    prior_path = ROOT / cfg["prior_manifest"]
    if sha(prior_path) != cfg["prior_manifest_sha256"]:
        raise ValueError("Prior manifest changed")
    prior = json.loads(prior_path.read_text())
    excluded = {r["group"] for r in prior["rows"]}
    candidates = freeze({**cfg, "evaluation_count": 64})
    fit = [r for r in prior["rows"] if r["role"] == "paired_fit"]
    if len(fit) != cfg["fit_count"]:
        raise ValueError("Prior fit count mismatch")
    evaluation = [
        r
        for r in candidates["rows"]
        if r["group"] not in excluded
        and Path(r["files"][0]["path"]).name not in cfg["excluded_preflight_names"]
    ]
    evaluation = evaluation[: cfg["evaluation_count"]]
    if len(evaluation) != cfg["evaluation_count"]:
        raise ValueError("Insufficient new development groups")
    for row in evaluation:
        row["role"] = "paired_development_evaluation"
    return {
        "config": cfg,
        "rows": fit + evaluation,
        "inventory_sha256": candidates["inventory_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/ai_conditioned_rgb_lut_v1.json").read_text())
    out = ROOT / cfg["destination"]
    if out.resolve().drive.upper() != "P:":
        raise ValueError("Require P-backed output")
    if args.freeze:
        manifest = new_manifest(cfg)
        out.mkdir(parents=True, exist_ok=False)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("MANIFEST", sha(out / "manifest.json"), flush=True)
        return
    if sha(out / "manifest.json") != args.manifest_sha256:
        raise ValueError("Manifest identity mismatch")
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["config"] != cfg:
        raise ValueError("Frozen config mismatch")
    prior_path = ROOT / cfg["prior_report"]
    if sha(prior_path) != cfg["prior_report_sha256"]:
        raise ValueError("Prior report changed")
    prior = json.loads(prior_path.read_text())
    checkpoint = prior_path.parent / "conditional.pt"
    if sha(checkpoint) != prior["arms"]["conditional"]["checkpoint_sha256"]:
        raise ValueError("Prior checkpoint changed")
    run = out / "run"
    run.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    owned = [
        "src/models/color_lut/conditioned_rgb_lut.py",
        "src/models/color_lut/lut.py",
        "scripts/run_ai_conditioned_rgb_lut.py",
        "configs/ai_conditioned_rgb_lut_v1.json",
        "scripts/run_ai_paired_recipe_pilot.py",
        "scripts/audit_ai_paired_operator_capacity.py",
    ]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("Relevant code must be committed")
    report = {
        "config": cfg,
        "manifest_sha256": args.manifest_sha256,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "implementation_sha256": {p: sha(ROOT / p) for p in owned},
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "arms": {},
    }
    (run / "execution_lock.json").write_text(json.dumps(report, indent=2))
    fit = [r for r in manifest["rows"] if r["role"] == "paired_fit"]
    root = ROOT / cfg["root"]
    x = torch.cat([decode(root, r["files"][0], cfg["size"]) for r in fit]).cuda()
    target = torch.stack(
        [torch.cat([decode(root, f, cfg["size"]) for f in r["files"][1:]]) for r in fit]
    ).cuda()
    models = {}
    for arm in ("global", "conditional", "wrong_target"):
        torch.manual_seed(cfg["seed"])
        model = ConditionedRGBLUT(arm != "global", cfg["lut_size"]).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        generator = torch.Generator().manual_seed(cfg["seed"])
        losses = []
        for step in range(cfg["steps"]):
            if time.monotonic() - started > cfg["max_seconds"]:
                raise TimeoutError("Fixed local-run budget exceeded")
            ids = torch.randint(len(fit), (cfg["batch"],), generator=generator).cuda()
            styles = torch.randint(3, (cfg["batch"],), generator=generator).cuda()
            t = target[(ids + 1) % len(fit) if arm == "wrong_target" else ids, styles]
            lut = model.predict_lut(x[ids], styles)
            y = apply_lut(x[ids], lut)
            data_loss = (y - t).abs().mean()
            fold = (
                (cfg["determinant_margin"] - cell_centre_determinants(lut))
                .relu()
                .square()
                .mean()
            )
            loss = (
                data_loss
                + cfg["curvature_weight"] * curvature(lut)
                + cfg["fold_weight"] * fold
            )
            opt.zero_grad()
            loss.backward()
            if any(
                not torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise ValueError("Nonfinite gradient")
            opt.step()
            losses.append(
                {"l1": float(data_loss.detach()), "fold_penalty": float(fold.detach())}
            )
            if step % 100 == 0:
                print(arm, step, losses[-1], flush=True)
        torch.save(model.cpu().state_dict(), run / f"{arm}.pt")
        report["arms"][arm] = {
            "losses": losses,
            "checkpoint_sha256": sha(run / f"{arm}.pt"),
            "evaluation": [],
        }
        models[arm] = model.cuda().eval()
    (run / "checkpoint_freeze.json").write_text(
        json.dumps(
            {k: a["checkpoint_sha256"] for k, a in report["arms"].items()}, indent=2
        )
    )
    old = ConditionedTriangular().cuda()
    old.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True
    )
    models["prior_triangular"] = old.eval()
    report["arms"]["prior_triangular"] = {
        "checkpoint_sha256": sha(checkpoint),
        "evaluation": [],
    }
    evaluation = [
        r for r in manifest["rows"] if r["role"] == "paired_development_evaluation"
    ]
    with torch.no_grad():
        for i, row in enumerate(evaluation):
            source = decode(root, row["files"][0]).cuda()
            if i < 3:
                save(source, run / f"eval_{i:02d}_input.png")
            for s, style in enumerate(cfg["styles"]):
                truth = decode(root, row["files"][s + 1]).cuda()
                if i < 3:
                    save(truth, run / f"eval_{i:02d}_{style}_target.png")
                for arm, model in models.items():
                    sid = torch.tensor([s], device="cuda")
                    y = model(source, sid)
                    record = {
                        "content": row["old_content_id"],
                        "style": style,
                        "l1": float((y - truth).abs().mean()),
                        "identity_l1": float((source - truth).abs().mean()),
                        "new_boundary_fraction": float(
                            (((source > 0) & (source < 1)) & ((y <= 0) | (y >= 1)))
                            .float()
                            .mean()
                        ),
                    }
                    if arm != "prior_triangular":
                        det = cell_centre_determinants(model.predict_lut(source, sid))
                        record["cell_centre_nonpositive_fraction"] = float(
                            (det <= 0).float().mean()
                        )
                    report["arms"][arm]["evaluation"].append(record)
                    if i < 3:
                        save(y, run / f"eval_{i:02d}_{style}_{arm}.png")
        # Previously exposed CC0 development photographs only; no optimization.
        sources = json.loads(
            (
                ROOT
                / "outputs/color_baseline/velvia50_rawpixls20_s0p50_gamutsafe/manifest.json"
            ).read_text()
        )
        source_lock = json.loads(
            (ROOT / "outputs/ai_structural_photo_pilot_v1/report.json").read_text()
        )
        report["transfer"] = []
        for i in (6, 7, 8):
            path = Path(sources[i]["before"])
            if (
                sources[i]["license"] != "CC0/Public Domain"
                or sha(path) != source_lock["source_hashes"][i]
            ):
                raise ValueError("Transfer identity changed")
            source = image_tensor(path).cuda()
            save(source, run / f"transfer_{i}_input.png")
            for arm in ("global", "conditional"):
                for s, style in enumerate(cfg["styles"]):
                    path = run / f"transfer_{i}_{style}_{arm}.png"
                    save(models[arm](source, torch.tensor([s], device="cuda")), path)
                    report["transfer"].append(
                        {
                            "source_sha256": source_lock["source_hashes"][i],
                            "output": path.name,
                            "sha256": sha(path),
                        }
                    )
    errors = {
        k: np.array([r["l1"] for r in a["evaluation"]])
        for k, a in report["arms"].items()
    }
    report["summary"] = {}
    for arm in ("global", "conditional"):
        gain = (errors["prior_triangular"] - errors[arm]) / np.maximum(
            errors["prior_triangular"], 1e-12
        )
        wrong_gain = (errors["wrong_target"] - errors[arm]) / np.maximum(
            errors["wrong_target"], 1e-12
        )
        report["summary"][arm] = {
            "median_gain_vs_prior": float(np.median(gain)),
            "wins_vs_prior": int((gain > 0).sum()),
            "median_gain_vs_wrong": float(np.median(wrong_gain)),
            "numerical_progress": bool(
                np.median(gain) >= 0.25
                and (gain > 0).sum() >= 36
                and np.median(wrong_gain) >= 0.2
            ),
        }
    report["summary"]["conditional_median_gain_vs_global"] = float(
        np.median(
            (errors["global"] - errors["conditional"])
            / np.maximum(errors["global"], 1e-12)
        )
    )
    report["elapsed_seconds"] = time.monotonic() - started
    report["product_promotion"] = False
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", sha(run / "report.json"), report["summary"], flush=True)


if __name__ == "__main__":
    main()
