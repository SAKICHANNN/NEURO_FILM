import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_local_reference_field_v1 import write_json
from src.color_match.research.local_reference_field_v1 import sha256


def reciprocal_third_scores(features: torch.Tensor, files: torch.Tensor, reciprocal_k: int = 3, required_files: int = 3) -> tuple[torch.Tensor, list]:
    scores, rows = [], []
    for source_file in files.unique(sorted=True).tolist():
        source_indices = torch.nonzero(files == source_file).flatten()
        donor_indices = torch.nonzero(files != source_file).flatten()
        source = features[source_indices]
        donors = features[donor_indices]
        donor_files = files[donor_indices]
        distances = (1 - source @ donors.T).clamp_min(0)
        reciprocal = distances.topk(min(reciprocal_k, len(source)), dim=0, largest=False).indices
        per_file = []
        for donor_file in donor_files.unique(sorted=True).tolist():
            indices = torch.nonzero(donor_files == donor_file).flatten()
            nearest = indices[distances[:, indices].argmin(1)]
            minima = distances[torch.arange(len(source)), nearest]
            accepted = (reciprocal[:, nearest] == torch.arange(len(source))[None, :]).any(0)
            per_file.append(torch.where(accepted, minima, torch.full_like(minima, float("inf"))))
        values = torch.stack(per_file, 1)
        selected = values.sort(1).values[:, required_files - 1] if values.shape[1] >= required_files else features.new_full((len(source),), float("inf"))
        scores.append(selected)
        rows.append({"reference_file_index": source_file, "patch_indices": source_indices.tolist(), "patch_count": len(source), "finite_third_scores": int(torch.isfinite(selected).sum()), "third_scores": [float(value) if torch.isfinite(value) else None for value in selected], "reciprocal_distinct_file_counts": torch.isfinite(values).sum(1).tolist()})
    return torch.cat(scores), rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs/own_ai_local_context_20260921"
    destination = base / "matcher_calibration_v2"
    config_destination = root / "configs/local_reference_field_v2.json"
    if destination.exists() or config_destination.exists():
        raise FileExistsError("Calibration/configuration already exists; no overwrite")
    original = json.loads((base / "pilot_v1/report.json").read_text())
    config = json.loads((base / "pilot_v1/config_frozen.json").read_text())
    diagnostic = json.loads((base / "matcher_diagnostic_v1/report.json").read_text())
    archive = root / diagnostic["arrays_path"]
    if sha256(archive) != diagnostic["arrays_sha256"]:
        raise ValueError("Frozen descriptor archive changed")
    if sha256(base / "pilot_v1/report.json") != diagnostic["original_report_sha256"]:
        raise ValueError("Original failed pilot changed")
    torch.set_num_threads(4)
    started = time.monotonic()
    with np.load(archive) as arrays:
        features = torch.from_numpy(arrays["reference_features"])
        files = torch.from_numpy(arrays["reference_files"])
    scores, rows = reciprocal_third_scores(features, files, config["reciprocal_k"], config["minimum_reference_files"])
    finite = torch.isfinite(scores)
    if not bool(finite.any()):
        raise RuntimeError("No finite reference-domain scores; no fallback")
    tau = float(torch.quantile(scores[finite], config["distance_quantile"]))
    if not 0 < tau < 2:
        raise RuntimeError("Degenerate threshold; no fallback")
    destination.mkdir(parents=True)
    receipt = {"schema": "reference-only-reciprocal-third-calibration-v2", "status": "FROZEN_BEFORE_NEW_SOURCE_OUTPUTS", "definition": "Leave one reference file out as query pool; choose nearest donor patch within each other file; require query in donor's top-3 query matches; score is third-smallest distance across these distinct reciprocal donor files. Threshold is Q90 of finite scores with no external distance cap. No source descriptors inspected for calibration.", "claim_ceiling": "Reference-domain engineering coverage, not semantic correctness or source-domain match accuracy. Italy19 remains one author/travel/scan group.", "descriptor_archive_path": diagnostic["arrays_path"], "descriptor_archive_sha256": diagnostic["arrays_sha256"], "original_report_sha256": diagnostic["original_report_sha256"], "original_config_sha256": original["config_sha256"], "calibration_code_sha256": sha256(Path(__file__)), "reference_manifest_sha256": config["reference_manifest_sha256"], "reference_files": len(files.unique()), "reference_patches": len(features), "finite_scores": int(finite.sum()), "finite_score_fraction": float(finite.float().mean()), "strict_below_tau_count": int((scores < tau).sum()), "strict_below_tau_all_patch_fraction": float((scores < tau).float().mean()), "quantile": config["distance_quantile"], "tau": tau, "elapsed_seconds": time.monotonic() - started, "rows": rows}
    receipt_path = destination / "calibration.json"
    write_json(receipt_path, receipt)
    config["schema"] = "local-reference-field-v2-reference-calibrated-gate"
    config["amendment"] += " V2 changes only the admission threshold calibration: reference-only leave-one-file-out third-order reciprocal distances, uncapped Q90. Original V1 and its no-support outcome remain unchanged. This is not a visual strength/loss/reference rescue. Cached descriptors are consumed without extraction."
    config.pop("distance_cap")
    config["calibrated_distance_threshold"] = tau
    config["calibration_receipt"] = receipt_path.relative_to(root).as_posix()
    config["calibration_receipt_sha256"] = sha256(receipt_path)
    config["frozen_descriptor_archive"] = diagnostic["arrays_path"]
    config["frozen_descriptor_archive_sha256"] = diagnostic["arrays_sha256"]
    config["frozen_input_manifest_sha256"] = original["input_manifest_sha256"]
    config["frozen_extractor_sha256"] = diagnostic["extractor_sha256"]
    config["output"] = "outputs/own_ai_local_context_20260921/pilot_v2"
    config["stopping_rule"] = "Exactly one reference-calibrated V2 batch: four sources x four arms, unchanged 384 updates per supported fit and 45-minute/8-GiB caps. No further threshold/strength/loss/reference changes. Empty support explicitly remains identity. Original V1 stays a no-support result."
    write_json(config_destination, config)
    print(json.dumps({"tau": tau, "finite_scores": int(finite.sum()), "reference_patches": len(features), "strict_below_tau_count": int((scores < tau).sum()), "calibration_sha256": sha256(receipt_path), "config_sha256": sha256(config_destination)}, indent=2))


if __name__ == "__main__":
    main()
