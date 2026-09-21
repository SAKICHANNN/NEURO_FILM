import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from scripts.run_ai_paired_recipe_pilot import save, sha
from scripts.run_local_reference_field_v1 import gpu_snapshot, write_json
from src.color_match.research.local_reference_field_v1 import load_image
from src.models.color_lut.conditioned_rgb_lut import ConditionedRGBLUT
from src.models.color_lut.lut import apply_lut
from src.models.source_preparation_v1 import SourcePreparation, coefficient_penalties, masked_l1, mean_field, preview_frame, render_frames, render_native


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/source_preparation_v1.json"
BOUND_CODE = [
    "src/models/source_preparation_v1.py", "scripts/run_source_preparation_v1.py",
    "configs/source_preparation_v1.json", "tests/test_source_preparation_v1.py",
    "src/models/color_lut/conditioned_rgb_lut.py", "src/models/color_lut/lut.py",
    "scripts/run_ai_paired_recipe_pilot.py", "scripts/run_local_reference_field_v1.py",
    "src/color_match/research/local_reference_field_v1.py",
]


def admitted_rows(manifest: dict) -> list:
    fields = ["pair_id", "camera_model", "source_name", "source_path", "source_sha256", "source_dng_sha256", "aligned_expert_path", "aligned_expert_sha256", "expert_c_sha256", "shape", "dtype", "gradient_correlation", "phase_shift_pixels_at_audit_scale", "split"]
    rows = [{key: row[key] for key in fields} for row in manifest["rows"] if row["split"] == "development"]
    rows.sort(key=lambda row: row["pair_id"])
    if len(rows) != 381 or len({r["pair_id"] for r in rows}) != 381 or len({r["camera_model"] for r in rows}) != 24:
        raise ValueError("Exact381 development /24 camera contract changed")
    return rows


def freeze(config: dict) -> None:
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError("Destination exists; no overwrite")
    for name in ("fivek_manifest", "response_manifest", "response_checkpoint"):
        if sha(ROOT / config[name]) != config[name + "_sha256"]:
            raise ValueError(f"Pinned {name} changed")
    rows = admitted_rows(json.loads((ROOT / config["fivek_manifest"]).read_text()))
    for row in rows:
        for field in ("source", "aligned_expert"):
            if sha(Path(row[field + "_path"])) != row[field + "_sha256"]:
                raise ValueError(f"Development file changed: {row['pair_id']}/{field}")
    response = json.loads((ROOT / config["response_manifest"]).read_text())
    if len(response["fit_rows"]) != 48 or response["config"]["styles"] != config["styles"]:
        raise ValueError("Response training roles/styles changed")
    challenge_path = ROOT / config["challenge_manifest"]
    challenges = json.loads(challenge_path.read_text())["sources"]
    if len(challenges) != 4 or any(r["role"] != "already_consumed_development_only" for r in challenges):
        raise ValueError("Challenge roles changed")
    for row in challenges:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Challenge bytes changed")
    schedule = np.random.default_rng(config["seed"]).integers(0, len(rows), size=(config["steps"], config["batch_size"]), dtype=np.int64)
    output.mkdir(parents=True)
    np.save(output / "sampling_schedule.npy", schedule)
    manifest = {"schema": "source-preparation-roles-v1", "config": config, "config_sha256": sha(CONFIG_PATH), "code_sha256": {p: sha(ROOT / p) for p in BOUND_CODE}, "development_rows": rows, "challenge_rows": challenges, "challenge_manifest_sha256": sha(challenge_path), "sampling_schedule_sha256": sha(output / "sampling_schedule.npy"), "response_fit_groups": [r["group"] for r in response["fit_rows"]], "confirmation_assets_opened": 0, "pixel_decodes_at_freeze": 0}
    write_json(output / "manifest.json", manifest)
    write_json(output / "config_frozen.json", config)
    print(json.dumps({"manifest_sha256": sha(output / "manifest.json"), "config_sha256": sha(CONFIG_PATH), "development_rows": len(rows), "confirmation_assets_opened": 0, "pixel_decodes": 0}, indent=2))


def load_development_image(row: dict, prefix: str) -> torch.Tensor:
    if row["split"] != "development" or prefix not in ("source", "aligned_expert"):
        raise ValueError("Forbidden role/target")
    path = Path(row[prefix + "_path"])
    if sha(path) != row[prefix + "_sha256"]:
        raise ValueError("Asset bytes changed")
    array = tifffile.imread(path)
    if array.dtype != np.uint16 or list(array.shape) != row["shape"] or array.shape[-1] != 3:
        raise ValueError("Normalized RGB16 geometry/dtype changed")
    return torch.from_numpy(array.astype(np.float32) / np.float32(65535)).permute(2, 0, 1)[None]


def make_model(arm: str, config: dict) -> SourcePreparation:
    torch.manual_seed(config["seed"])
    return SourcePreparation(arm == "local", config["grid_size"], config["common_bound"], config["balance_bound"])


def training_loss(model: SourcePreparation, source: torch.Tensor, expert: torch.Tensor, mask: torch.Tensor, boxes: list, config: dict) -> tuple[torch.Tensor, dict]:
    source_coefficients, expert_coefficients = model(source, mask), model(expert, mask)
    supervised = masked_l1(render_frames(source, source_coefficients, boxes), expert, mask)
    identity = masked_l1(render_frames(expert, expert_coefficients, boxes), expert, mask)
    smooth, size = coefficient_penalties(torch.cat((source_coefficients, expert_coefficients)))
    loss = supervised + config["identity_weight"] * identity + config["smooth_weight"] * smooth + config["size_weight"] * size
    return loss, {"expert_l1": float(supervised.detach()), "identity_l1": float(identity.detach()), "smooth": float(smooth.detach()), "size": float(size.detach())}


def run(config: dict) -> None:
    output = ROOT / config["output"]
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"] != config or manifest["config_sha256"] != sha(CONFIG_PATH):
        raise ValueError("Frozen config drift")
    for path, expected in manifest["code_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen code drift: {path}")
    if subprocess.check_output(["git", "diff", "HEAD", "--", *BOUND_CODE], cwd=ROOT):
        raise ValueError("Bound implementation is not committed")
    for name in ("fivek_manifest", "response_manifest", "response_checkpoint"):
        if sha(ROOT / config[name]) != config[name + "_sha256"]:
            raise ValueError(f"Pinned {name} drift")
    schedule_path = output / "sampling_schedule.npy"
    if sha(schedule_path) != manifest["sampling_schedule_sha256"]:
        raise ValueError("Schedule drift")
    destination = output / "run"
    if destination.exists():
        raise FileExistsError("Single run already attempted")
    snapshot = gpu_snapshot()
    if len(snapshot["gpus"]) != 1 or snapshot["gpus"][0][3] < config["minimum_free_gib"] * 1024:
        raise RuntimeError("Insufficient GPU headroom; no foreign processes touched")
    destination.mkdir()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(min(1.0, config["max_process_gib"] * 1024 ** 3 / torch.cuda.get_device_properties(device).total_memory), device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    report = {"schema": "source-preparation-report-v1", "status": "RUNNING", "manifest_sha256": sha(manifest_path), "config_sha256": sha(CONFIG_PATH), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "pid": os.getpid(), "torch": torch.__version__, "gpu_snapshot": snapshot, "response_checkpoint_sha256": config["response_checkpoint_sha256"], "claim_ceiling": config["claim_ceiling"], "arms": {}, "rows": [], "confirmation_assets_opened": 0}

    def guard(stage):
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError(f"Fixed30minute cap: {stage}")
        if torch.cuda.max_memory_reserved(device) > config["max_process_gib"] * 1024 ** 3 or torch.cuda.mem_get_info(device)[0] < 512 * 1024 ** 2:
            raise RuntimeError(f"Memory guard: {stage}; no foreign processes touched")

    def relative(path):
        return path.relative_to(ROOT).as_posix()

    def save_array(folder, name, value):
        image_path, float_path = folder / f"{name}.png", folder / f"{name}.npy"
        save(value, image_path)
        array = value.cpu().numpy()[0].transpose(1, 2, 0)
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Invalid native output")
        np.save(float_path, array)
        return {"image_path": relative(image_path), "image_sha256": sha(image_path), "float_path": relative(float_path), "float_sha256": sha(float_path), "size_wh": [array.shape[1], array.shape[0]]}

    try:
        xs, ys, masks, boxes = [], [], [], []
        for row in manifest["development_rows"]:
            guard("decode development")
            x, mask, box = preview_frame(load_development_image(row, "source"), config["preview_side"])
            y, target_mask, target_box = preview_frame(load_development_image(row, "aligned_expert"), config["preview_side"])
            if box != target_box or not torch.equal(mask, target_mask):
                raise ValueError("Pair frame mismatch")
            xs.append(x); ys.append(y); masks.append(mask); boxes.append(box)
        x, y, mask = (torch.cat(values).to(device) for values in (xs, ys, masks))
        del xs, ys, masks
        schedule = np.load(schedule_path)
        models = {}
        report["unprepared_development_l1"] = float(masked_l1(x, y, mask))
        for arm in config["arms"]:
            model = make_model(arm, config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            trace = []
            for step, indices in enumerate(schedule):
                guard(f"{arm}/{step}")
                ids = torch.as_tensor(indices, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss, terms = training_loss(model, x[ids], y[ids], mask[ids], [boxes[int(i)] for i in indices], config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise FloatingPointError("Nonfinite gradients")
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                trace.append({"step": step + 1, "total": float(loss.detach()), **terms})
                if (step + 1) % 100 == 0:
                    write_json(destination / "progress.json", {"stage": "training", "arm": arm, "step": step + 1, "elapsed_seconds": time.monotonic() - started})
                    print(arm, step + 1, terms, flush=True)
            checkpoint = destination / f"{arm}.pt"
            torch.save({"model": {key: value.cpu() for key, value in model.state_dict().items()}, "arm": arm, "config_sha256": sha(CONFIG_PATH), "manifest_sha256": sha(manifest_path)}, checkpoint)
            write_json(destination / f"{arm}_trace.json", trace)
            report["arms"][arm] = {"steps": len(trace), "parameter_count": sum(p.numel() for p in model.parameters()), "checkpoint_path": relative(checkpoint), "checkpoint_sha256": sha(checkpoint), "trace_path": relative(destination / f"{arm}_trace.json")}
            models[arm] = model.eval()
            del optimizer
        write_json(destination / "checkpoint_freeze.json", report["arms"])
        del x, y, mask
        response = ConditionedRGBLUT(False, 9).to(device)
        response_checkpoint = torch.load(ROOT / config["response_checkpoint"], map_location=device, weights_only=True)
        if response_checkpoint["arm"] != "shared_global" or response_checkpoint["manifest_sha256"] != config["response_manifest_sha256"]:
            raise ValueError("Frozen response identity mismatch")
        response.load_state_dict(response_checkpoint["model"])
        response.eval().requires_grad_(False)
        with torch.no_grad():
            for index, row in enumerate(manifest["challenge_rows"]):
                guard(f"native challenge/{index}")
                if sha(ROOT / row["path"]) != row["sha256"]:
                    raise ValueError("Challenge bytes changed")
                source = load_image(ROOT / row["path"]).to(device)
                preview, valid, box = preview_frame(source, config["preview_side"])
                coefficients = {arm: model(preview, valid) for arm, model in models.items()}
                coefficients["meanfield"] = mean_field(coefficients["local"])
                prepared = {arm: render_native(source, nodes, config["native_tile_size"]) for arm, nodes in coefficients.items()}
                folder = destination / f"challenge_{index:02d}"
                folder.mkdir()
                original = save_array(folder, "original", source)
                record = {"source_id": f"challenge_{index:02d}", "source_path": row["path"], "source_sha256": row["sha256"], "source_metadata": row, "original": original, "original_path": original["image_path"], "size_wh": original["size_wh"], "preview_box_tlhw": list(box), "preparation": {}, "styles": {}}
                for arm, nodes in coefficients.items():
                    coeff_path = folder / f"{arm}_coefficients.npy"
                    np.save(coeff_path, nodes.cpu().numpy())
                    record["preparation"][arm] = {**save_array(folder, f"P_{arm}", prepared[arm]), "coefficient_path": relative(coeff_path), "coefficient_sha256": sha(coeff_path), "coefficient_spatial_range": float((nodes.amax((-2, -1)) - nodes.amin((-2, -1))).max()), "mean_absolute_change": float((prepared[arm] - source).abs().mean())}
                for style_index, style in enumerate(config["styles"]):
                    guard(f"native challenge/{index}/{style}")
                    lut = response.predict_lut(source, torch.tensor([style_index], device=device))
                    style_record = {"arms": {}}
                    for arm, input_image in {"shared": source, **prepared}.items():
                        result = torch.empty_like(source)
                        tile = config["native_tile_size"]
                        for top in range(0, source.shape[-2], tile):
                            for left in range(0, source.shape[-1], tile):
                                sl = (..., slice(top, top + tile), slice(left, left + tile))
                                result[sl] = apply_lut(input_image[sl], lut)
                        style_record["arms"][arm] = {**save_array(folder, f"{style}_{arm}", result), "mean_absolute_change": float((result - source).abs().mean())}
                    record["styles"][style] = style_record
                report["rows"].append(record)
                write_json(destination / "report.json", report)
        report["status"] = "COMPLETE_DEVELOPMENT_VISUAL_REVIEW_PENDING"
    except Exception as exc:
        report["status"] = "FAILED_NO_RESTART"
        report["failure"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["response_checkpoint_unchanged"] = sha(ROOT / config["response_checkpoint"]) == config["response_checkpoint_sha256"]
        write_json(destination / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "peak_reserved_bytes", "confirmation_assets_opened")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(CONFIG_PATH.read_text())
    freeze(configuration) if args.freeze else run(configuration)
