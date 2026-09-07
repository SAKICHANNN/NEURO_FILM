"""Fit-pool oracle diagnostic: no predictive-model or independent-test claim."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_ai_paired_recipe_pilot import decode, save, sha
from src.models.color_lut.conditioned_triangular import apply_parameters
from src.models.color_lut.lut import apply_lut, identity_lut


def select_fit(manifest, count):
    rows = [r for r in manifest["rows"] if r["role"] == "paired_fit"]
    if count < 1 or len(rows) < count:
        raise ValueError("Insufficient fit roles")
    selected = rows[:count]
    if len({r["group"] for r in selected}) != count:
        raise ValueError("Repeated group")
    for row in selected:
        if len(row["files"]) != 4 or any(
            not f["path"].startswith("train/") for f in row["files"]
        ):
            raise ValueError("Forbidden source")
    return selected


def checkerboard(height, width, block):
    if min(height, width, block) < 1 or min(height, width) <= block:
        raise ValueError("Need both checkerboard partitions")
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return ((y // block + x // block) % 2) == 0


def curvature(lut):
    return sum(lut.diff(n=2, dim=axis).square().mean() for axis in (1, 2, 3))


def cell_centre_determinants(lut):
    """Exact trilinear Jacobian at cell centres only, not a whole-cell proof."""
    scale = lut.shape[1] - 1
    dr = lut[:, 1:] - lut[:, :-1]
    dr = (
        dr[:, :, :-1, :-1] + dr[:, :, 1:, :-1] + dr[:, :, :-1, 1:] + dr[:, :, 1:, 1:]
    ) * (scale / 4)
    dg = lut[:, :, 1:] - lut[:, :, :-1]
    dg = (
        dg[:, :-1, :, :-1] + dg[:, 1:, :, :-1] + dg[:, :-1, :, 1:] + dg[:, 1:, :, 1:]
    ) * (scale / 4)
    db = lut[:, :, :, 1:] - lut[:, :, :, :-1]
    db = (db[:, :-1, :-1] + db[:, 1:, :-1] + db[:, :-1, 1:] + db[:, 1:, 1:]) * (
        scale / 4
    )
    return torch.linalg.det(torch.stack([dr, dg, db], dim=-1))


def main():
    config_path = ROOT / "configs/ai_paired_operator_capacity_v1.json"
    cfg = json.loads(config_path.read_text())
    manifest_path = ROOT / cfg["manifest"]
    if sha(manifest_path) != cfg["manifest_sha256"]:
        raise ValueError("Changed paired manifest")
    manifest = json.loads(manifest_path.read_text())
    if cfg["styles"] != manifest["config"]["styles"]:
        raise ValueError("Style order mismatch")
    rows = select_fit(manifest, cfg["fit_images"])
    out = ROOT / cfg["destination"]
    if out.resolve().drive.upper() != "P:":
        raise ValueError("Need P-backed output")
    owned = [
        "scripts/audit_ai_paired_operator_capacity.py",
        "configs/ai_paired_operator_capacity_v1.json",
        "src/models/color_lut/conditioned_triangular.py",
        "src/models/color_lut/lut.py",
        "scripts/run_ai_paired_recipe_pilot.py",
    ]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("Commit relevant code before running")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    report = {
        "config": cfg,
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
        "sources": rows,
        "arms": {},
        "claim": cfg["claim"],
    }
    (out / "execution_lock.json").write_text(json.dumps(report, indent=2))
    xs, targets = [], []
    for row in rows:
        x = decode(ROOT / cfg["root"], row["files"][0], cfg["size"])
        for f in row["files"][1:]:
            xs.append(x)
            targets.append(decode(ROOT / cfg["root"], f, cfg["size"]))
    x, target = torch.cat(xs).cuda(), torch.cat(targets).cuda()
    mask = checkerboard(*x.shape[-2:], cfg["checker_block"]).cuda()
    # All pixels are loaded, but odd blocks never enter optimization or selection.
    for arm in ("triangular", "full_rgb_lut"):
        torch.manual_seed(cfg["seed"])
        parameters = nn.Parameter(
            torch.zeros(len(x), 3, 12, device="cuda")
            if arm == "triangular"
            else identity_lut(cfg["lut_size"], device=x.device)
            .unsqueeze(0)
            .repeat(len(x), 1, 1, 1, 1)
        )
        operator = apply_parameters if arm == "triangular" else apply_lut
        optimizer = torch.optim.Adam([parameters], lr=cfg["learning_rate"])
        losses = []
        for step in range(cfg["steps"]):
            if time.monotonic() - started > cfg["max_seconds"]:
                raise TimeoutError("Fixed diagnostic time cap")
            y = operator(x, parameters)
            data_loss = (y - target)[:, :, mask].abs().mean()
            loss = data_loss + (
                cfg["lut_curvature_weight"] * curvature(parameters)
                if arm == "full_rgb_lut"
                else 0
            )
            optimizer.zero_grad()
            loss.backward()
            if not torch.isfinite(parameters.grad).all():
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            if arm == "full_rgb_lut":
                with torch.no_grad():
                    parameters.clamp_(0, 1)
            losses.append(float(data_loss.detach()))
            if step % 100 == 0:
                print(arm, step, losses[-1], flush=True)
        checkpoint = out / f"{arm}.pt"
        torch.save(parameters.detach().cpu(), checkpoint)
        with torch.no_grad():
            y = operator(x, parameters)
            error = (y - target).abs()
            fit_error = error[:, :, mask].mean((1, 2)).cpu().tolist()
            odd_error = error[:, :, ~mask].mean((1, 2)).cpu().tolist()
            channel_error = error[:, :, ~mask].mean(2).cpu().tolist()
            identity = (x - target)[:, :, ~mask].abs().mean((1, 2)).cpu().tolist()
            new_boundary = ((x > 0) & (x < 1)) & ((y <= 0) | (y >= 1))
            records = []
            for i, row in enumerate(rows):
                for s, style in enumerate(cfg["styles"]):
                    j = i * len(cfg["styles"]) + s
                    records.append(
                        {
                            "content": row["old_content_id"],
                            "style": style,
                            "fit_l1": fit_error[j],
                            "odd_block_l1": odd_error[j],
                            "odd_block_rgb_l1": channel_error[j],
                            "identity_odd_block_l1": identity[j],
                            "new_boundary_components": int(new_boundary[j].sum()),
                        }
                    )
                    if i < 2:
                        save(y[j : j + 1], out / f"fit_{i:02d}_{style}_{arm}.png")
                        if arm == "triangular":
                            save(
                                target[j : j + 1],
                                out / f"fit_{i:02d}_{style}_target.png",
                            )
                            if s == 0:
                                save(x[j : j + 1], out / f"fit_{i:02d}_input.png")
            arm_report = {
                "parameter_count_per_image_style": parameters[0].numel(),
                "losses": losses,
                "records": records,
                "checkpoint_sha256": sha(checkpoint),
            }
            if arm == "full_rgb_lut":
                det = cell_centre_determinants(parameters)
                arm_report["cell_centre_nonpositive_jacobian_fraction"] = (
                    (det <= 0).float().flatten(1).mean(1).cpu().tolist()
                )
            report["arms"][arm] = arm_report
    a = np.array([r["odd_block_l1"] for r in report["arms"]["triangular"]["records"]])
    b = np.array([r["odd_block_l1"] for r in report["arms"]["full_rgb_lut"]["records"]])
    gains = (a - b) / np.maximum(a, 1e-12)
    report["headroom"] = {
        "median_relative_gain": float(np.median(gains)),
        "worst_relative_gain": float(gains.min()),
        "wins": int((b < a).sum()),
        "numerical_signal": bool(
            np.median(gains) >= cfg["headroom_median_gain"]
            and (b < a).sum() >= cfg["headroom_min_wins"]
        ),
        "product_promotion": False,
    }
    report["elapsed_seconds"] = time.monotonic() - started
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", sha(out / "report.json"), report["headroom"], flush=True)


if __name__ == "__main__":
    main()
