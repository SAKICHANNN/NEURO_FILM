import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.color_match.research.local_reference_field_v1 import (
    apply_field, bounded_nodes, centers_for, descriptors, image_metrics,
    interpolate_nodes, load_descriptor, load_image, match_references, objective,
    projected_quantiles, reference_threshold, sha256, support_field,
    target_permutation,
)


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def gpu_snapshot() -> dict:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
    return {"gpus": [[int(v.strip()) for v in row.split(",")] for row in rows], "compute_processes": processes}


def save_image(path: Path, x: torch.Tensor) -> None:
    array = (x.detach().cpu()[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def run(root: Path, config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"Run destination already exists; no overwrite or restart: {output}")
    snapshot = gpu_snapshot()
    if len(snapshot["gpus"]) != 1:
        raise RuntimeError("This pilot admits exactly the known single local GPU")
    if snapshot["gpus"][0][3] < config["minimum_free_gib"] * 1024:
        raise RuntimeError(f"Insufficient free local GPU memory, no process touched: {snapshot}")
    input_path = root / config["input_manifest"]
    reference_path = root / config["reference_manifest"]
    if sha256(reference_path) != config["reference_manifest_sha256"]:
        raise ValueError("Reference manifest hash mismatch")
    inputs = json.loads(input_path.read_text(encoding="utf-8-sig"))["sources"]
    refs = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    if len(inputs) != 4 or len(refs["rows"]) != 19:
        raise ValueError("Frozen source/reference cardinality mismatch")
    for row in inputs:
        if row["role"] != "already_consumed_development_only" or sha256(root / row["path"]) != row["sha256"]:
            raise ValueError("Source role or content mismatch")
    for row in refs["rows"]:
        if sha256(reference_path.parent / row["path"]) != row["sha256"]:
            raise ValueError("Reference content mismatch")
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / "config_frozen.json")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, config["peak_process_gib"] * 1024 ** 3 / total), device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    report = {
        "schema": "local-reference-field-report-v1", "status": "RUNNING",
        "config_sha256": sha256(config_path), "input_manifest_sha256": sha256(input_path),
        "claim_ceiling": config["claim_ceiling"], "decision_rule": config["decision_rule"],
        "environment": {"pid": os.getpid(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "initial_gpu_snapshot": snapshot, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()},
        "code_sha256": {p: sha256(root / p) for p in ["src/color_match/research/local_reference_field_v1.py", "scripts/run_local_reference_field_v1.py"]},
        "reference_rows": [{**row, "path": str((reference_path.parent / row["path"]).relative_to(root)).replace("\\", "/")} for row in refs["rows"]],
        "rows": [], "elapsed_seconds": 0,
    }

    def relative(path):
        return path.relative_to(root).as_posix()

    def guard(stage):
        elapsed = time.monotonic() - started
        if elapsed > config["wall_time_seconds"]:
            raise TimeoutError(f"Fixed wall-time cap reached at {stage}")
        if torch.cuda.max_memory_reserved(device) > config["peak_process_gib"] * 1024 ** 3:
            raise RuntimeError(f"Fixed process VRAM cap exceeded at {stage}")
        free, _ = torch.cuda.mem_get_info(device)
        if free < 512 * 1024 ** 2:
            raise RuntimeError(f"Less than 512 MiB free at {stage}; stop without touching other processes")

    try:
        model = load_descriptor(root, config, device)
        features, targets, files, reference_centers = [], [], [], []
        with torch.no_grad():
            for index, row in enumerate(refs["rows"]):
                guard(f"reference {index}")
                x = load_image(reference_path.parent / row["path"], config["working_long_edge"]).to(device)
                centers = centers_for(x, config["grid_spacing"])
                features.append(descriptors(x, centers, model, config))
                targets.append(projected_quantiles(x, centers, config))
                files.append(torch.full((len(centers),), index, device=device, dtype=torch.long))
                reference_centers.extend([{"file_index": index, "center_xy": center, "working_size_wh": [x.shape[-1], x.shape[-2]]} for center in centers.cpu().tolist()])
                write_json(output / "progress.json", {"stage": "reference_features", "reference": index + 1, "total": 19, "elapsed_seconds": time.monotonic() - started})
        reference_features, reference_targets, reference_files = torch.cat(features), torch.cat(targets), torch.cat(files)
        tau = reference_threshold(reference_features, reference_files, config)
        report["matching_threshold"] = tau
        write_json(output / "reference_patch_index.json", reference_centers)
        for source_index, source in enumerate(inputs):
            guard(f"source {source_index}")
            source_id = f"source_{source_index + 1:02d}"
            folder = output / source_id
            folder.mkdir()
            original_path = folder / "original.png"
            shutil.copyfile(root / source["path"], original_path)
            native = load_image(root / source["path"]).to(device)
            x = load_image(root / source["path"], config["working_long_edge"]).to(device)
            working_shape = tuple(x.shape[-2:])
            centers = centers_for(x, config["grid_spacing"])
            with torch.no_grad():
                source_features = descriptors(x, centers, model, config)
                matched = match_references(source_features, reference_features, reference_files, reference_targets, tau, config)
            active_centers = centers[matched["selected"]]
            permutation = target_permutation(active_centers)
            matching_path = folder / "matching.json"
            write_json(matching_path, {"tau": tau, "source_centers_xy": centers.cpu().tolist(), "source_working_size_wh": [x.shape[-1], x.shape[-2]], "records": matched["records"], "selected_source_indices": matched["selected"].cpu().tolist(), "confidence": matched["confidence"].cpu().tolist(), "target_quantiles": matched["targets"].cpu().tolist(), "misalignment_permutation": permutation.cpu().tolist(), "misalignment_discriminating": len(active_centers) >= 2, "reference_patch_index_path": relative(output / "reference_patch_index.json")})
            support = support_field(working_shape, working_shape, active_centers, matched["confidence"], config)
            native_support = support_field(tuple(native.shape[-2:]), working_shape, active_centers, matched["confidence"], config)
            support_path = folder / "support.png"
            save_image(support_path, native_support.expand(-1, 3, -1, -1))
            np.save(folder / "support.npy", native_support.cpu().numpy()[0, 0])
            row = {"source_id": source_id, "source_path": source["path"], "original_path": relative(original_path), "source_sha256": source["sha256"], "size_wh": source["size_wh"], "crop_boxes": source["existing_crop_boxes"], "matching_path": relative(matching_path), "support_path": relative(support_path), "supported_centers": len(active_centers), "total_centers": len(centers), "arms": {}}
            report["rows"].append(row)
            for arm in config["arms"]:
                guard(f"{source_id}/{arm}")
                local = arm in ("L", "M")
                node_shape = (math.ceil(x.shape[-2] / config["grid_spacing"]) + 1, math.ceil(x.shape[-1] / config["grid_spacing"]) + 1) if local else (1, 1)
                raw = torch.nn.Parameter(torch.zeros(1, 12, *node_shape, device=device))
                optimizer = torch.optim.Adam([raw], lr=config["learning_rate"], betas=tuple(config["adam_betas"]))
                fit_target = matched["targets"][permutation] if arm == "M" else matched["targets"]
                arm_support = torch.ones_like(support) if arm == "G" else support
                trace, steps = [], 0
                if len(active_centers):
                    for step in range(config["steps"]):
                        guard(f"{source_id}/{arm}/{step}")
                        optimizer.zero_grad(set_to_none=True)
                        nodes = bounded_nodes(raw, config)
                        field = interpolate_nodes(nodes, working_shape, working_shape) * arm_support
                        y = apply_field(x, field)
                        y = torch.where(arm_support == 0, x, y)
                        loss, terms = objective(x, y, nodes, active_centers, fit_target, matched["confidence"], config)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(f"Nonfinite loss: {source_id}/{arm}/{step}")
                        loss.backward()
                        if not bool(torch.isfinite(raw.grad).all()):
                            raise FloatingPointError(f"Nonfinite gradient: {source_id}/{arm}/{step}")
                        torch.nn.utils.clip_grad_norm_([raw], config["gradient_clip"])
                        optimizer.step()
                        steps += 1
                        trace.append({"step": steps, "loss": float(loss.detach()), **terms})
                        if steps % 24 == 0:
                            write_json(output / "progress.json", {"stage": "fitting", "source_id": source_id, "arm": arm, "steps": steps, "elapsed_seconds": time.monotonic() - started, "loss": float(loss.detach())})
                with torch.no_grad():
                    nodes = bounded_nodes(raw, config)
                    field = interpolate_nodes(nodes, tuple(native.shape[-2:]), working_shape)
                    used_support = torch.ones_like(native_support) if arm == "G" else native_support
                    field = field * used_support
                    result = torch.where(used_support == 0, native, apply_field(native, field))
                    metrics = image_metrics(native, result, used_support, config)
                    if not metrics["finite"] or metrics["minimum"] < 0 or metrics["maximum"] > 1:
                        raise FloatingPointError(f"Invalid output: {source_id}/{arm}")
                    if arm != "G" and metrics["unsupported_max_rgb_change"] not in (None, 0.0):
                        raise FloatingPointError("Unsupported pixels did not remain exact")
                    probe_field = interpolate_nodes(nodes, working_shape, working_shape) * arm_support
                    palette = native.new_tensor([[0.18,0.18,0.18],[0.5,0.5,0.5],[0.8,0.8,0.8],[0.65,0.4,0.3],[0.7,0.3,0.5],[0.2,0.5,0.2]])
                    indices = centers.round().long()
                    center_fields = probe_field[0, :, indices[:, 1], indices[:, 0]].T
                    probes = []
                    for rgb in palette:
                        probe_x = rgb[None, :, None, None].expand(len(centers), -1, 1, 1)
                        probes.append(apply_field(probe_x, center_fields[:, :, None, None]).squeeze(-1).squeeze(-1))
                    probe_values = torch.stack(probes)
                    metrics["same_rgb_position_range_max"] = float((probe_values.max(1).values - probe_values.min(1).values).abs().max())
                    image_path, float_path = folder / f"{arm}.png", folder / f"{arm}.npy"
                    coefficient_path, trace_path = folder / f"{arm}_coefficients.npz", folder / f"{arm}_trace.json"
                    save_image(image_path, result)
                    np.save(float_path, result.cpu().numpy()[0].transpose(1, 2, 0))
                    np.savez_compressed(coefficient_path, raw=raw.detach().cpu().numpy(), bounded_nodes=nodes.cpu().numpy(), center_fields=center_fields.cpu().numpy(), palette=palette.cpu().numpy(), palette_outputs=probe_values.cpu().numpy())
                    write_json(trace_path, trace)
                    row["arms"][arm] = {"image_path": relative(image_path), "float_path": relative(float_path), "coefficients_path": relative(coefficient_path), "trace_path": relative(trace_path), "metrics": metrics, "steps": steps, "status": "FIT_COMPLETE_UNREVIEWED" if steps else "NO_SUPPORT_IDENTITY", "image_sha256": sha256(image_path), "float_sha256": sha256(float_path)}
                report["elapsed_seconds"] = time.monotonic() - started
                write_json(output / "report.json", report)
                print(f"{source_id} {arm}: {steps} updates; {len(active_centers)}/{len(centers)} supported centers; {report['elapsed_seconds']:.1f}s", flush=True)
                del raw, optimizer, nodes, field, result, probe_field
            del native, x, source_features
        report["status"] = "COMPLETE_UNREVIEWED_NO_PROMOTION"
    except Exception as error:
        report["status"] = "STOPPED_ERROR_NO_RESTART"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["completed_fits"] = sum(len(row["arms"]) for row in report["rows"])
        report["actual_updates"] = sum(arm["steps"] for row in report["rows"] for arm in row["arms"].values())
        write_json(output / "report.json", report)
        write_json(output / "progress.json", {"stage": report["status"], "elapsed_seconds": report["elapsed_seconds"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_reference_field_v1.json")
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run(project_root, project_root / arguments.config)
