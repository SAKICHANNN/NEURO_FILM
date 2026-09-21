import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_local_reference_field_v1 import gpu_snapshot, write_json
from src.color_match.research.local_reference_field_v1 import (
    centers_for, descriptors, load_descriptor, load_image, sha256,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = root / "outputs/own_ai_local_context_20260921/pilot_v1"
    output = root / "outputs/own_ai_local_context_20260921/matcher_diagnostic_v1"
    original = json.loads((pilot / "report.json").read_text())
    config_path = pilot / "config_frozen.json"
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError("One diagnostic only; existing destination must remain unchanged")
    if sha256(config_path) != original["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    module_path = "src/color_match/research/local_reference_field_v1.py"
    if sha256(root / module_path) != original["code_sha256"][module_path]:
        raise ValueError("Original feature extraction implementation changed")
    if sha256(root / config["input_manifest"]) != original["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")
    if sha256(root / config["reference_manifest"]) != config["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed")
    sources = json.loads((root / config["input_manifest"]).read_text())["sources"]
    references = original["reference_rows"]
    for row in sources + references:
        if sha256(root / row["path"]) != row["sha256"]:
            raise ValueError("Frozen image content changed")
    snapshot = gpu_snapshot()
    if len(snapshot["gpus"]) != 1 or snapshot["gpus"][0][3] < config["minimum_free_gib"] * 1024:
        raise RuntimeError(f"Local GPU not eligible; no other process touched: {snapshot}")
    output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, config["peak_process_gib"] * 1024 ** 3 / total), device)
    torch.cuda.reset_peak_memory_stats(device)
    model = load_descriptor(root, config, device)
    feature_groups, centers_groups, file_groups = [], [], []
    with torch.no_grad():
        for index, row in enumerate(references + sources):
            if time.monotonic() - started > 180:
                raise TimeoutError("Three-minute diagnostic cap")
            free, _ = torch.cuda.mem_get_info(device)
            if free < 512 * 1024 ** 2:
                raise RuntimeError("GPU free memory guard; stop without touching other processes")
            x = load_image(root / row["path"], config["working_long_edge"]).to(device)
            centers = centers_for(x, config["grid_spacing"])
            feature_groups.append(descriptors(x, centers, model, config).cpu())
            centers_groups.append(centers.cpu())
            file_groups.append(torch.full((len(centers),), index, dtype=torch.long))
    del model
    ref_features = torch.cat(feature_groups[:len(references)])
    ref_files = torch.cat(file_groups[:len(references)])
    ref_distance = (1 - ref_features @ ref_features.T).clamp_min(0)
    self_distance = ref_distance.diagonal().clone()
    ref_distance.masked_fill_(ref_files[:, None] == ref_files[None, :], float("inf"))
    nearest_other_file = ref_distance.min(1).values
    raw_q90 = float(torch.quantile(nearest_other_file, config["distance_quantile"]))
    tau = original["matching_threshold"]
    recalculated_tau = min(config["distance_cap"], raw_q90)
    if abs(recalculated_tau - tau) > 1e-7:
        raise ValueError("Recomputed frozen matching threshold differs")
    arrays = {"reference_features": ref_features.numpy(), "reference_files": ref_files.numpy(), "reference_centers_xy": torch.cat(centers_groups[:len(references)]).numpy(), "reference_nearest_other_file_distances": nearest_other_file.numpy(), "reference_self_distances": self_distance.numpy()}
    report = {"schema": "local-reference-matcher-diagnostic-v1", "status": "COMPLETE_ATTRIBUTION_ONLY", "original_report_sha256": sha256(pilot / "report.json"), "config_sha256": sha256(config_path), "extractor_sha256": sha256(root / module_path), "diagnostic_code_sha256": sha256(Path(__file__)), "scope": "One recomputation of unchanged descriptors on the exact original files. No threshold scan, new references, photo render or optimizer. Stage counts diagnose admission, not semantic match validity or film quality.", "gpu_snapshot": snapshot, "existing_tau": tau, "uncapped_reference_q90": raw_q90, "reference_patches": len(ref_features), "reference_self_match_max_distance": float(self_distance.max()), "reference_self_match_positive_control_pass": bool((self_distance < 1e-5).all()), "rows": []}
    for source_index, features in enumerate(feature_groups[len(references):]):
        distances = (1 - features @ ref_features.T).clamp_min(0)
        reciprocal = distances.topk(min(config["reciprocal_k"], len(features)), dim=0, largest=False).indices
        file_minima, file_indices = [], []
        for file_index in range(len(references)):
            indices = torch.nonzero(ref_files == file_index).flatten()
            selected = distances[:, indices].argmin(1)
            file_indices.append(indices[selected])
            file_minima.append(distances[torch.arange(len(features)), indices[selected]])
        indices = torch.stack(file_indices, 1)
        minima = torch.stack(file_minima, 1)
        reciprocal_pass = (reciprocal[:, indices] == torch.arange(len(features))[None, :, None]).any(0)
        distance_pass = minima < tau
        joint_pass = distance_pass & reciprocal_pass
        source_self = (1 - features @ features.T).clamp_min(0).diagonal()
        per_patch = []
        for patch_index in range(len(features)):
            per_patch.append({"source_patch_index": patch_index, "nearest_reference_distance": float(minima[patch_index].min()), "third_distinct_file_distance": float(minima[patch_index].sort().values[2]), "distance_pass_files": int(distance_pass[patch_index].sum()), "reciprocal_pass_files_ignoring_distance": int(reciprocal_pass[patch_index].sum()), "joint_pass_files": int(joint_pass[patch_index].sum())})
        source_id = original["rows"][source_index]["source_id"]
        arrays.update({f"{source_id}_features": features.numpy(), f"{source_id}_centers_xy": centers_groups[len(references) + source_index].numpy(), f"{source_id}_reference_distances": distances.numpy(), f"{source_id}_nearest_per_file_indices": indices.numpy(), f"{source_id}_distance_pass": distance_pass.numpy(), f"{source_id}_reciprocal_pass": reciprocal_pass.numpy()})
        report["rows"].append({"source_id": source_id, "source_path": sources[source_index]["path"], "source_sha256": sources[source_index]["sha256"], "source_patches": len(features), "distance_passing_pairs": int(distance_pass.sum()), "reciprocal_passing_pairs_ignoring_distance": int(reciprocal_pass.sum()), "joint_passing_pairs": int(joint_pass.sum()), "distance_only_supported_centers": int((distance_pass.sum(1) >= config["minimum_reference_files"]).sum()), "reciprocal_only_supported_centers": int((reciprocal_pass.sum(1) >= config["minimum_reference_files"]).sum()), "joint_supported_centers": int((joint_pass.sum(1) >= config["minimum_reference_files"]).sum()), "nearest_distance_min": float(minima.min()), "nearest_distance_median": float(minima.min(1).values.median()), "self_match_max_distance": float(source_self.max()), "self_match_positive_control_pass": bool((source_self < 1e-5).all()), "per_patch": per_patch})
    np.savez_compressed(output / "frozen_descriptors_and_distances.npz", **arrays)
    report["arrays_path"] = (output / "frozen_descriptors_and_distances.npz").relative_to(root).as_posix()
    report["arrays_sha256"] = sha256(output / "frozen_descriptors_and_distances.npz")
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "uncapped_reference_q90": raw_q90, "existing_tau": tau, "elapsed_seconds": report["elapsed_seconds"], "sources": [{k: r[k] for k in ["source_id", "distance_passing_pairs", "reciprocal_passing_pairs_ignoring_distance", "joint_passing_pairs", "nearest_distance_min"]} for r in report["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
