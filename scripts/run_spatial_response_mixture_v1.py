import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_ai_paired_operator_capacity import cell_centre_determinants, curvature
from scripts.run_ai_paired_recipe_pilot import decode, save, sha
from scripts.run_local_reference_field_v1 import gpu_snapshot, write_json
from src.color_match.research.local_reference_field_v1 import load_image
from src.models.color_lut.conditioned_rgb_lut import ConditionedRGBLUT
from src.models.color_lut.lut import apply_lut
from src.models.color_lut.spatial_response_mixture_v1 import SpatialResponseMixture, effective_probe_luts, spatial_weight_penalty


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/spatial_response_mixture_v1.json"
OWNED_CODE = [
    "src/models/color_lut/spatial_response_mixture_v1.py",
    "scripts/run_spatial_response_mixture_v1.py",
    "configs/spatial_response_mixture_v1.json",
    "tests/test_spatial_response_mixture_v1.py",
    "src/models/color_lut/conditioned_rgb_lut.py",
    "src/models/color_lut/lut.py",
    "scripts/run_ai_paired_recipe_pilot.py",
    "scripts/audit_ai_paired_operator_capacity.py",
    "scripts/run_local_reference_field_v1.py",
    "src/color_match/research/local_reference_field_v1.py",
]


def secondary_split(prior: dict, config: dict) -> tuple[list, list]:
    rows = [copy.deepcopy(row) for row in prior["rows"] if row["role"] == "paired_fit"]
    if len(rows) != 64 or len({row["group"] for row in rows}) != 64:
        raise ValueError("Expected exactly64 admitted paired-fit groups")
    rows.sort(key=lambda row: hashlib.sha256((config["selection_seed"] + "|" + row["group"]).encode()).hexdigest())
    for row in rows:
        if len(row["files"]) != 4 or any(not file["path"].startswith("train/") for file in row["files"]):
            raise ValueError("Forbidden source role/path")
    check = rows[:config["check_count"]]
    fit = rows[config["check_count"]:]
    if len(fit) != config["fit_count"] or len(check) != config["check_count"]:
        raise ValueError("Split cardinality mismatch")
    for role, subset in (("fit", fit), ("secondary_check", check)):
        for row in subset:
            row["experiment_role"] = role
    return fit, check


def make_model(arm: str, config: dict) -> torch.nn.Module:
    torch.manual_seed(config["seed"])
    if arm in ("local", "pooled"):
        return SpatialResponseMixture(arm == "local", config["lut_size"], config["basis_count"], len(config["styles"]), config["initialization_noise"])
    return ConditionedRGBLUT(arm == "full_global", config["lut_size"])


def freeze(config: dict) -> None:
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError("Frozen destination already exists")
    prior_path = ROOT / config["prior_manifest"]
    if sha(prior_path) != config["prior_manifest_sha256"]:
        raise ValueError("Prior manifest changed")
    prior = json.loads(prior_path.read_text())
    if prior["config"]["styles"] != config["styles"]:
        raise ValueError("Style identity/order changed")
    fit, check = secondary_split(prior, config)
    challenge_path = ROOT / config["challenge_manifest"]
    challenges = json.loads(challenge_path.read_text())["sources"]
    if len(challenges) != 4 or any(row["role"] != "already_consumed_development_only" for row in challenges):
        raise ValueError("Challenge population changed")
    generator = torch.Generator().manual_seed(config["seed"])
    schedule = torch.stack([torch.stack((torch.randint(len(fit), (config["batch_size"],), generator=generator), torch.randint(3, (config["batch_size"],), generator=generator))) for _ in range(config["steps"])]).numpy()
    output.mkdir(parents=True)
    np.save(output / "sampling_schedule.npy", schedule)
    manifest = {"schema": "spatial-response-mixture-roles-v1", "config": config, "config_sha256": sha(CONFIG_PATH), "prior_manifest_sha256": sha(prior_path), "code_sha256": {p: sha(ROOT / p) for p in OWNED_CODE}, "fit_rows": fit, "secondary_check_rows": check, "challenge_rows": challenges, "challenge_manifest_sha256": sha(challenge_path), "sampling_schedule_sha256": sha(output / "sampling_schedule.npy"), "original_evaluation_rows_excluded": [row["group"] for row in prior["rows"] if row["role"] != "paired_fit"], "claim_ceiling": config["claim_ceiling"], "pixel_decodes_at_freeze": 0}
    write_json(output / "manifest.json", manifest)
    write_json(output / "config_frozen.json", config)
    print(json.dumps({"manifest_sha256": sha(output / "manifest.json"), "config_sha256": sha(CONFIG_PATH), "fit_groups": len(fit), "secondary_check_groups": len(check), "fit_check_group_overlap": len({r['group'] for r in fit} & {r['group'] for r in check}), "pixel_decodes": 0}, indent=2))


def fit_loss(model: torch.nn.Module, arm: str, x: torch.Tensor, style: torch.Tensor, target: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    if arm in ("local", "pooled"):
        bases, weights = model.predict_bases(style), model.predict_weights(x, style)
        y = model.render(x, bases, weights)
        luts = effective_probe_luts(bases, weights)
        smooth_weights = spatial_weight_penalty(weights)
        curve = curvature(bases.flatten(0, 1))
    else:
        luts = model.predict_lut(x, style)
        y = apply_lut(x, luts)
        smooth_weights = y.sum() * 0
        curve = curvature(luts)
    l1 = (y - target).abs().mean()
    fold = (config["determinant_margin"] - cell_centre_determinants(luts)).relu().square().mean()
    total = l1 + config["curvature_weight"] * curve + config["fold_weight"] * fold + config["weight_spatial_penalty"] * smooth_weights
    return total, {"l1": float(l1.detach()), "curvature": float(curve.detach()), "fold": float(fold.detach()), "weight_spatial": float(smooth_weights.detach())}


@torch.no_grad()
def render_native(model: torch.nn.Module, arm: str, x: torch.Tensor, style: torch.Tensor, tile_size: int, misalign: bool = False) -> tuple[torch.Tensor, dict]:
    if arm in ("local", "pooled"):
        bases, weights = model.predict_bases(style), model.predict_weights(x, style)
        if misalign:
            weights = weights.roll((weights.shape[-2] // 2, weights.shape[-1] // 2), (-2, -1))
        dense = F.interpolate(weights, x.shape[-2:], mode="bilinear", align_corners=False)
        dense /= dense.sum(1, keepdim=True)
        auxiliary = {"weights": weights.cpu().numpy(), "bases": bases.cpu().numpy()}
    else:
        lut = model.predict_lut(x, style)
        auxiliary = {"lut": lut.cpu().numpy()}
    result = torch.empty_like(x)
    for top in range(0, x.shape[-2], tile_size):
        for left in range(0, x.shape[-1], tile_size):
            cut = x[..., top:top + tile_size, left:left + tile_size]
            if arm in ("local", "pooled"):
                w = dense[..., top:top + tile_size, left:left + tile_size]
                y = sum(apply_lut(cut, bases[:, index]) * w[:, index:index + 1] for index in range(model.bases))
            else:
                y = apply_lut(cut, lut)
            result[..., top:top + tile_size, left:left + tile_size] = y.clamp(0, 1)
    return result, auxiliary


def metrics(source: torch.Tensor, result: torch.Tensor, target: torch.Tensor | None) -> dict:
    return {"finite": bool(torch.isfinite(result).all()), "minimum": float(result.min()), "maximum": float(result.max()), "mean_absolute_change": float((result - source).abs().mean()), "new_boundary_fraction": float((((source > 0) & (source < 1)) & ((result <= 0) | (result >= 1))).float().mean()), "target_l1": float((result - target).abs().mean()) if target is not None else None}


def run(config: dict) -> None:
    output = ROOT / config["output"]
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"] != config or manifest["config_sha256"] != sha(CONFIG_PATH):
        raise ValueError("Frozen configuration changed")
    for path, expected in manifest["code_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen code changed: {path}")
    if subprocess.check_output(["git", "diff", "HEAD", "--", *OWNED_CODE], cwd=ROOT):
        raise ValueError("Commit the frozen relevant code before execution")
    if sha(output / "sampling_schedule.npy") != manifest["sampling_schedule_sha256"]:
        raise ValueError("Frozen sampling changed")
    snapshot = gpu_snapshot()
    if len(snapshot["gpus"]) != 1 or snapshot["gpus"][0][3] < config["minimum_free_gib"] * 1024:
        raise RuntimeError(f"Local GPU not eligible; no process touched: {snapshot}")
    destination = output / "run"
    destination.mkdir(exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(1.0, config["max_process_gib"] * 1024 ** 3 / torch.cuda.get_device_properties(device).total_memory), device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    report = {"schema": "spatial-response-mixture-report-v1", "status": "RUNNING", "manifest_sha256": sha(manifest_path), "config_sha256": sha(CONFIG_PATH), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "gpu_snapshot": snapshot, "torch": torch.__version__, "claim_ceiling": config["claim_ceiling"], "arms": {}, "secondary_check": [], "challenge_rows": [], "elapsed_seconds": 0}

    def guard(stage):
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError(f"Fixed total budget exceeded at {stage}")
        if torch.cuda.max_memory_reserved(device) > config["max_process_gib"] * 1024 ** 3 or torch.cuda.mem_get_info(device)[0] < 512 * 1024 ** 2:
            raise RuntimeError(f"Fixed memory guard at {stage}; no foreign process touched")

    def rel(path):
        return path.relative_to(ROOT).as_posix()

    def save_prediction(folder, name, source, result, auxiliary, target):
        image_path, float_path, auxiliary_path = folder / f"{name}.png", folder / f"{name}.npy", folder / f"{name}_operator.npz"
        save(result, image_path)
        np.save(float_path, result.cpu().numpy()[0].transpose(1, 2, 0))
        np.savez_compressed(auxiliary_path, **auxiliary)
        return {"image_path": rel(image_path), "float_path": rel(float_path), "operator_path": rel(auxiliary_path), "image_sha256": sha(image_path), "float_sha256": sha(float_path), "metrics": metrics(source, result, target)}

    try:
        data_root = ROOT / config["data_root"]
        fit = manifest["fit_rows"]
        x = torch.cat([decode(data_root, row["files"][0], config["training_size"]) for row in fit]).to(device)
        target = torch.stack([torch.cat([decode(data_root, file, config["training_size"]) for file in row["files"][1:]]) for row in fit]).to(device)
        schedule = torch.from_numpy(np.load(output / "sampling_schedule.npy")).to(device)
        models = {}
        for arm in config["arms"]:
            guard(arm)
            model = make_model(arm, config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            trace = []
            for step in range(config["steps"]):
                guard(f"{arm}/{step}")
                ids, styles = schedule[step]
                optimizer.zero_grad(set_to_none=True)
                loss, terms = fit_loss(model, arm, x[ids], styles, target[ids, styles], config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError("Nonfinite parameter gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                trace.append({"step": step + 1, "total": float(loss.detach()), **terms})
                if (step + 1) % 100 == 0:
                    write_json(destination / "progress.json", {"stage": "training", "arm": arm, "step": step + 1, "elapsed_seconds": time.monotonic() - started})
                    print(arm, step + 1, terms, flush=True)
            checkpoint = destination / f"{arm}.pt"
            torch.save({"model": {key: value.cpu() for key, value in model.state_dict().items()}, "arm": arm, "config_sha256": sha(CONFIG_PATH), "manifest_sha256": sha(manifest_path)}, checkpoint)
            write_json(destination / f"{arm}_trace.json", trace)
            report["arms"][arm] = {"steps": config["steps"], "parameter_count": sum(p.numel() for p in model.parameters()), "checkpoint_path": rel(checkpoint), "checkpoint_sha256": sha(checkpoint), "trace_path": rel(destination / f"{arm}_trace.json")}
            models[arm] = model.eval()
            del optimizer
        write_json(destination / "checkpoint_freeze.json", report["arms"])
        del x, target, schedule
        with torch.no_grad():
            for index, row in enumerate(manifest["secondary_check_rows"]):
                guard(f"secondary_check/{index}")
                source = decode(data_root, row["files"][0]).to(device)
                folder = destination / f"check_{index:02d}"
                panel = index < config["panel_check_images"]
                if panel:
                    folder.mkdir()
                    save(source, folder / "original.png")
                record = {"source_id": f"check_{index:02d}", "group": row["group"], "source_identity": row["old_content_id"], "source_sha256": row["files"][0]["sha256"], "size_wh": [source.shape[-1], source.shape[-2]], "original_path": rel(folder / "original.png") if panel else None, "styles": {}}
                for style_index, style_name in enumerate(config["styles"]):
                    style = torch.tensor([style_index], device=device)
                    truth = decode(data_root, row["files"][style_index + 1]).to(device)
                    if truth.shape != source.shape:
                        raise ValueError("Paired native image geometry differs")
                    if panel:
                        save(truth, folder / f"{style_name}_target.png")
                    style_record = {"target_path": rel(folder / f"{style_name}_target.png") if panel else None, "target_sha256": row["files"][style_index + 1]["sha256"], "identity_l1": float((source - truth).abs().mean()), "arms": {}}
                    for arm in [*config["arms"], "misaligned"]:
                        name = "local" if arm == "misaligned" else arm
                        result, auxiliary = render_native(models[name], name, source, style, config["native_tile_size"], arm == "misaligned")
                        style_record["arms"][arm] = save_prediction(folder, f"{style_name}_{arm}", source, result, auxiliary, truth) if panel else {"metrics": metrics(source, result, truth)}
                    record["styles"][style_name] = style_record
                report["secondary_check"].append(record)
                write_json(destination / "report.json", report)
            for index, row in enumerate(manifest["challenge_rows"]):
                guard(f"challenge/{index}")
                if sha(ROOT / row["path"]) != row["sha256"]:
                    raise ValueError("Challenge source changed")
                source = load_image(ROOT / row["path"]).to(device)
                folder = destination / f"challenge_{index:02d}"
                folder.mkdir()
                save(source, folder / "original.png")
                record = {"source_id": f"challenge_{index:02d}", "source_path": row["path"], "source_sha256": row["sha256"], "original_path": rel(folder / "original.png"), "size_wh": row["size_wh"], "crop_boxes": row["existing_crop_boxes"], "styles": {}}
                for style_index, style_name in enumerate(config["styles"]):
                    style = torch.tensor([style_index], device=device)
                    record["styles"][style_name] = {"arms": {}}
                    for arm in [*config["arms"], "misaligned"]:
                        guard(f"challenge/{index}/{style_name}/{arm}")
                        name = "local" if arm == "misaligned" else arm
                        result, auxiliary = render_native(models[name], name, source, style, config["native_tile_size"], arm == "misaligned")
                        record["styles"][style_name]["arms"][arm] = save_prediction(folder, f"{style_name}_{arm}", source, result, auxiliary, None)
                report["challenge_rows"].append(record)
                write_json(destination / "report.json", report)
        means = {arm: float(np.mean([np.mean([row["styles"][style]["arms"][arm]["metrics"]["target_l1"] for style in config["styles"]]) for row in report["secondary_check"]])) for arm in [*config["arms"], "misaligned"]}
        best_global = min(("pooled", "full_global", "shared_global"), key=means.get)
        gain = 1 - means["local"] / means[best_global]
        report["primary"] = {"per_source_then_style_mean_l1": means, "strongest_global_comparator": best_global, "relative_gain": gain, "numeric_pass": gain >= config["minimum_primary_gain"], "visual_decision": "PENDING_ACTUAL_REVIEW_NO_PROMOTION"}
        report["status"] = "COMPLETE_VISUAL_REVIEW_PENDING"
    except Exception as error:
        report["status"] = "STOPPED_NO_RESTART"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        write_json(destination / "report.json", report)
        write_json(destination / "progress.json", {"stage": report["status"], "elapsed_seconds": report["elapsed_seconds"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    configuration = json.loads(CONFIG_PATH.read_text())
    freeze(configuration) if arguments.freeze else run(configuration)
