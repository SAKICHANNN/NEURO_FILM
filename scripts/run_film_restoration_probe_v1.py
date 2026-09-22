import argparse
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageCms
import tifffile
import torch

from src.models.source_preparation_v1 import SourcePreparation, apply_coefficients, coefficient_penalties, masked_l1, mean_field, preview_frame, render_frames, render_native


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/film_restoration_probe_v1.json"
BOUND_CODE = ["scripts/run_film_restoration_probe_v1.py", "configs/film_restoration_probe_v1.json", "tests/test_film_restoration_probe_v1.py", "src/models/source_preparation_v1.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def corruption_schedule(seed: int, config: dict, evaluation: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    if evaluation:
        indices = np.repeat(np.arange(7, dtype=np.int64), config["evaluation_repeats_per_image"])[:, None]
    else:
        indices = rng.integers(0, 7, size=(config["steps"], config["batch_size"]), dtype=np.int64)
    common = rng.uniform(-config["corruption_common_bound"], config["corruption_common_bound"], (*indices.shape, 1))
    balance = rng.uniform(-config["corruption_balance_draw_bound"], config["corruption_balance_draw_bound"], (*indices.shape, 3))
    balance -= balance.mean(-1, keepdims=True)
    coefficients = np.concatenate((common, balance), -1).astype(np.float32)[..., None, None]
    identity = np.zeros(len(indices), dtype=np.bool_)
    if not evaluation:
        identity[rng.choice(len(indices), round(len(indices) * config["identity_batch_fraction"]), replace=False)] = True
        coefficients[identity] = 0
    return {"indices": indices, "coefficients": coefficients, "identity_batches": identity}


def make_model(seed: int, config: dict) -> SourcePreparation:
    torch.manual_seed(seed)
    return SourcePreparation(True, config["grid_size"], config["common_bound"], config["balance_bound"])


def state_sha(model: SourcePreparation) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def training_loss(model: SourcePreparation, target: torch.Tensor, mask: torch.Tensor, boxes: list, corruption: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    corrupted = apply_coefficients(target, corruption)
    coefficients = model(corrupted, mask)
    reconstruction = masked_l1(render_frames(corrupted, coefficients, boxes), target, mask)
    smooth, size = coefficient_penalties(coefficients)
    loss = reconstruction + config["smooth_weight"] * smooth + config["size_weight"] * size
    return loss, {"reconstruction_l1": float(reconstruction.detach()), "smooth": float(smooth.detach()), "size": float(size.detach())}


def target_rows(config: dict) -> dict:
    acquisition = json.loads((ROOT / config["film_acquisition"]).read_text(encoding="utf-8-sig"))
    index = json.loads((ROOT / config["digital_index"]).read_text(encoding="utf-8-sig"))
    fivek_path = ROOT / config["fivek_manifest"]
    if sha(fivek_path) != config["fivek_manifest_sha256"] or index["manifest_sha256"] != config["fivek_manifest_sha256"]:
        raise ValueError("FiveK manifest identity changed")
    development = {row["pair_id"]: row for row in json.loads(fivek_path.read_text())["rows"] if row["split"] == "development"}
    films = {row["id"]: row for row in acquisition["rows"]}
    digitals = {row["index"]: row for row in index["rows"]}
    result = {"film": [], "digital": []}
    for identity in config["film_ids"]:
        row = films[identity]
        if row["status"] != "ACQUIRED" or row["rights"]["license"] != "https://creativecommons.org/licenses/by/4.0/":
            raise ValueError("Film identity/rights not admitted")
        result["film"].append({"id": identity, "path": row["image_path"], "sha256": row["image_sha256"], "decode": "film", "provenance": row})
    for identity in config["digital_indices"]:
        row = digitals[identity]
        original = development[row["pair_id"]]
        if row["split"] != "development" or any(row[key] != original[key] for key in ("aligned_expert_path", "aligned_expert_sha256")):
            raise ValueError("Only exact previously consumed development Expert C targets admitted")
        result["digital"].append({"id": row["pair_id"], "path": row["aligned_expert_path"], "sha256": row["aligned_expert_sha256"], "decode": "digital", "shape": original["shape"], "provenance": row})
    if any(len(rows) != 7 or len({r["sha256"] for r in rows}) != 7 for rows in result.values()):
        raise ValueError("Exactly seven unique targets per corpus required")
    return result


def decode_image(row: dict) -> tuple[torch.Tensor, dict]:
    path = ROOT / row["path"]
    if sha(path) != row["sha256"]:
        raise ValueError(f"Asset changed: {path}")
    metadata = {"target_precision": "encoded_sRGB_uint8_then_float32_div255", "icc_sha256": None}
    if row["decode"] == "digital":
        array = tifffile.imread(path)
        if array.dtype != np.uint16 or list(array.shape) != row["shape"] or array.shape[-1] != 3:
            raise ValueError("Expected frozen normalized RGB16 target")
        array = np.rint(array.astype(np.float64) * (255 / 65535)).astype(np.uint8)
        metadata["color_action"] = "already_normalized_encoded_sRGB16_quantized_to8"
    else:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.getexif().get(274, 1) != 1:
                raise ValueError("Expected upright RGB image; no implicit geometry changes")
            icc = image.info.get("icc_profile")
            if icc and row["decode"] == "film":
                metadata["icc_sha256"] = hashlib.sha256(icc).hexdigest()
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                image = ImageCms.profileToProfile(image, profile, ImageCms.createProfile("sRGB"), renderingIntent=1, outputMode="RGB", flags=0)
                metadata["color_action"] = "embedded_ICC_to_sRGB8_relative_colorimetric_no_BPC"
            else:
                metadata["color_action"] = "assume_untagged_sRGB" if row["decode"] == "film" else "existing_normalized_challenge_sRGB8_no_reconversion"
            array = np.asarray(image).copy()
    metadata["decoded_shape_hwc"] = list(array.shape)
    metadata["decoded_uint8_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
    return torch.from_numpy(array.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None], metadata


def freeze(config: dict) -> None:
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError("Freeze destination already exists")
    rows = target_rows(config)
    challenges = json.loads((ROOT / config["challenge_manifest"]).read_text())["sources"]
    if len(challenges) != 4 or any(row["role"] != "already_consumed_development_only" for row in challenges):
        raise ValueError("Only four consumed challenges admitted")
    for row in [*rows["film"], *rows["digital"], *challenges]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Input bytes changed")
    output.mkdir(parents=True)
    schedules = {}
    for label, seed, evaluation in [(str(seed), seed, False) for seed in config["seeds"]] + [("evaluation", config["evaluation_seed"], True)]:
        path = output / f"schedule_{label}.npz"
        np.savez(path, **corruption_schedule(seed, config, evaluation))
        schedules[label] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    manifest = {"schema": "film-restoration-probe-freeze-v1", "config": config, "config_sha256": sha(CONFIG_PATH), "code_sha256": {path: sha(ROOT / path) for path in BOUND_CODE}, "source_manifests": {key: sha(ROOT / config[key]) for key in ("film_acquisition", "digital_index", "fivek_manifest", "challenge_manifest")}, "target_rows": rows, "challenge_rows": challenges, "schedules": schedules, "pixel_decodes_at_freeze": 0, "confirmation_assets_opened": 0}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest_sha256": sha(output / "manifest.json"), "target_counts": {k: len(v) for k, v in rows.items()}, "pixel_decodes": 0}, indent=2))


def gpu_snapshot() -> dict:
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
    return {"gpus": [[int(v.strip()) for v in line.split(",")] for line in lines], "compute_processes": processes}


def run(config: dict, manifest_sha256: str) -> None:
    output = ROOT / config["output"]
    manifest_path = output / "manifest.json"
    if sha(manifest_path) != manifest_sha256:
        raise ValueError("Explicit frozen manifest hash required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != config or manifest["config_sha256"] != sha(CONFIG_PATH):
        raise ValueError("Frozen config drift")
    for path, digest in manifest["code_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen code drift: {path}")
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "diff", "HEAD", "--", *BOUND_CODE], cwd=ROOT):
        raise ValueError("Bound code must be committed")
    for name, digest in manifest["source_manifests"].items():
        if sha(ROOT / config[name]) != digest:
            raise ValueError("Source manifest drift")
    schedules = {}
    for label, row in manifest["schedules"].items():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Schedule drift")
        with np.load(ROOT / row["path"], allow_pickle=False) as arrays:
            schedules[label] = {key: arrays[key] for key in arrays.files}
    destination = output / "run"
    if destination.exists():
        raise FileExistsError("Single attempt already exists; no restart")
    snapshot = gpu_snapshot()
    if len(snapshot["gpus"]) != 1 or snapshot["gpus"][0][3] < config["minimum_free_gib"] * 1024:
        raise RuntimeError("Insufficient local GPU headroom; no foreign process touched")
    if snapshot["compute_processes"]:
        raise RuntimeError("Foreign compute process present; no process touched")
    destination.mkdir()
    started = time.monotonic()
    report = {"status": "RUNNING", "manifest_sha256": manifest_sha256, "pid": os.getpid(), "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "environment": {"torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__, "tifffile": tifffile.__version__}, "gpu_snapshot": snapshot, "claim_ceiling": config["claim_ceiling"], "models": {}, "targets": {}, "diagnostics": {}, "rows": [], "confirmation_assets_opened": 0}
    write_json(destination / "report.json", report)
    device = torch.device("cuda:0")

    def guard(stage):
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError(f"30 minute total cap: {stage}")
        if torch.cuda.max_memory_reserved(device) > config["max_process_gib"] * 1024 ** 3 or torch.cuda.mem_get_info(device)[0] < 512 * 1024 ** 2:
            raise RuntimeError(f"GPU memory cap: {stage}")

    def save_array(folder, name, value):
        array = value.detach().cpu().numpy()[0].transpose(1, 2, 0)
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Nonfinite or out-of-range output")
        image_path, float_path = folder / f"{name}.png", folder / f"{name}.npy"
        Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(image_path)
        np.save(float_path, array)
        return {"image_path": image_path.relative_to(ROOT).as_posix(), "image_sha256": sha(image_path), "float_path": float_path.relative_to(ROOT).as_posix(), "float_sha256": sha(float_path), "size_wh": [array.shape[1], array.shape[0]]}

    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(min(1.0, config["max_process_gib"] * 1024 ** 3 / torch.cuda.get_device_properties(device).total_memory), device)
        torch.cuda.reset_peak_memory_stats(device)
        corpora = {}
        for domain, rows in manifest["target_rows"].items():
            targets, masks, boxes, metadata = [], [], [], []
            for row in rows:
                guard("decode targets")
                image, decode = decode_image(row)
                target, mask, box = preview_frame(image, config["preview_side"])
                targets.append(target); masks.append(mask); boxes.append(box)
                metadata.append({"id": row["id"], "decode": decode, "preview_box_tlhw": list(box)})
            corpora[domain] = (torch.cat(targets).to(device), torch.cat(masks).to(device), boxes)
            target_path = destination / f"{domain}_targets.npz"
            np.savez_compressed(target_path, targets=torch.cat(targets).numpy(), masks=torch.cat(masks).numpy(), boxes=np.asarray(boxes))
            report["targets"][domain] = {"rows": metadata, "path": target_path.relative_to(ROOT).as_posix(), "sha256": sha(target_path)}
        models = {}
        for seed in config["seeds"]:
            schedule = schedules[str(seed)]
            for domain in config["domains"]:
                name = f"{domain}_{seed}"
                model = make_model(seed, config).to(device)
                initial_sha = state_sha(model)
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                targets, masks, boxes = corpora[domain]
                trace = []
                trace_path = destination / f"{name}_trace.jsonl"
                with trace_path.open("x", encoding="utf-8") as stream:
                    for step, indices in enumerate(schedule["indices"]):
                        guard(f"train {name}/{step}")
                        ids = torch.as_tensor(indices, device=device)
                        corruption = torch.from_numpy(schedule["coefficients"][step]).to(device)
                        optimizer.zero_grad(set_to_none=True)
                        loss, terms = training_loss(model, targets[ids], masks[ids], [boxes[int(i)] for i in indices], corruption, config)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("Nonfinite training loss")
                        loss.backward()
                        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                            raise FloatingPointError("Nonfinite gradients")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                        optimizer.step()
                        record = {"step": step + 1, "total": float(loss.detach()), "identity_batch": bool(schedule["identity_batches"][step]), **terms}
                        trace.append(record)
                        stream.write(json.dumps(record) + "\n")
                        if (step + 1) % 100 == 0:
                            stream.flush()
                            write_json(destination / "progress.json", {"model": name, "step": step + 1, "elapsed_seconds": time.monotonic() - started})
                            print(name, step + 1, terms, flush=True)
                checkpoint = destination / f"{name}.pt"
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "name": name, "manifest_sha256": manifest_sha256}, checkpoint)
                report["models"][name] = {"domain": domain, "seed": seed, "steps": len(trace), "initial_state_sha256": initial_sha, "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha(checkpoint), "trace_sha256": sha(trace_path), "parameter_count": sum(p.numel() for p in model.parameters())}
                models[name] = model.eval().requires_grad_(False)
                del optimizer
                write_json(destination / "report.json", report)
        with torch.no_grad():
            evaluation = schedules["evaluation"]
            for name, model in models.items():
                report["diagnostics"][name] = {}
                for domain, (targets, masks, boxes) in corpora.items():
                    identity = render_frames(targets, model(targets, masks), boxes)
                    identity_errors = [float(masked_l1(identity[i:i + 1], targets[i:i + 1], masks[i:i + 1])) for i in range(7)]
                    restored, corrupted_all, coefficients_all, errors, baseline_errors = [], [], [], [], []
                    for step, indices in enumerate(evaluation["indices"]):
                        guard("synthetic diagnostics")
                        i = int(indices[0])
                        target, mask = targets[i:i + 1], masks[i:i + 1]
                        corrupted = apply_coefficients(target, torch.from_numpy(evaluation["coefficients"][step]).to(device))
                        nodes = model(corrupted, mask)
                        result = render_frames(corrupted, nodes, [boxes[i]])
                        errors.append(float(masked_l1(result, target, mask)))
                        baseline_errors.append(float(masked_l1(corrupted, target, mask)))
                        restored.append(result.cpu().numpy()); corrupted_all.append(corrupted.cpu().numpy()); coefficients_all.append(nodes.cpu().numpy())
                    path = destination / f"diagnostic_{name}_on_{domain}.npz"
                    np.savez_compressed(path, identity_outputs=identity.cpu().numpy(), corrupted=np.concatenate(corrupted_all), restored=np.concatenate(restored), coefficients=np.concatenate(coefficients_all), target_indices=evaluation["indices"].ravel())
                    report["diagnostics"][name][domain] = {"reused_development_images_no_generalization": True, "evaluated_on_own_training_corpus": report["models"][name]["domain"] == domain, "identity_l1_per_image": identity_errors, "identity_l1": float(np.mean(identity_errors)), "reconstruction_l1_per_corruption": errors, "reconstruction_l1": float(np.mean(errors)), "corrupted_l1_per_corruption": baseline_errors, "corrupted_l1": float(np.mean(baseline_errors)), "artifact_path": path.relative_to(ROOT).as_posix(), "artifact_sha256": sha(path)}
                write_json(destination / "report.json", report)
            for index, row in enumerate(manifest["challenge_rows"]):
                guard("challenge render")
                source, metadata = decode_image({**row, "decode": "challenge"})
                source = source.to(device)
                preview, mask, box = preview_frame(source, config["preview_side"])
                folder = destination / f"challenge_{index:02d}"
                folder.mkdir()
                record = {"source_id": f"challenge_{index:02d}", "source_metadata": row, "decode": metadata, "preview_box_tlhw": list(box), "original": save_array(folder, "original", source), "models": {}}
                outputs = {}
                for name, model in models.items():
                    guard(f"challenge {index}/{name}")
                    nodes = model(preview, mask)
                    result = render_native(source, nodes, config["native_tile_size"])
                    mean_result = render_native(source, mean_field(nodes), config["native_tile_size"])
                    coeff_path = folder / f"{name}_coefficients.npy"
                    np.save(coeff_path, nodes.cpu().numpy())
                    record["models"][name] = {"local": save_array(folder, name, result), "meanfield": save_array(folder, name + "_meanfield", mean_result), "coefficient_path": coeff_path.relative_to(ROOT).as_posix(), "coefficient_sha256": sha(coeff_path), "coefficient_spatial_range": float((nodes.amax((-2, -1)) - nodes.amin((-2, -1))).max()), "mean_absolute_change": float((result - source).abs().mean()), "local_vs_meanfield_mae": float((result - mean_result).abs().mean())}
                    outputs[name] = result
                seed_a, seed_b = config["seeds"]
                record["descriptive_differences"] = {"same_seed_cross_corpus_mae": {str(seed): float((outputs[f"film_{seed}"] - outputs[f"digital_{seed}"]).abs().mean()) for seed in config["seeds"]}, "same_corpus_cross_seed_mae": {domain: float((outputs[f"{domain}_{seed_a}"] - outputs[f"{domain}_{seed_b}"]).abs().mean()) for domain in config["domains"]}, "interpretation": "Descriptive n=2 only; no causal attribution or promotion threshold"}
                report["rows"].append(record)
                write_json(destination / "report.json", report)
        guard("completion")
        report["status"] = "COMPLETE_DEVELOPMENT_VISUAL_REVIEW_PENDING"
    except Exception as exc:
        report["status"] = "FAILED_NO_RESTART"
        report["failure"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if torch.cuda.is_initialized() else 0
        write_json(destination / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "peak_reserved_bytes", "confirmation_assets_opened")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.freeze:
        freeze(configuration)
    elif not args.manifest_sha256:
        parser.error("--run requires --manifest-sha256 from the reviewed freeze")
    else:
        run(configuration, args.manifest_sha256)
