"""Colour-blind hard nearest-case selection between fixed BL5 and AO6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModel

from src.eval.global_frontier import sha256_file


CANDIDATE = "fixed_bl5_strict_interior_sigmoid"
FALLBACK = "fixed_ao6_colour_only_t15_c35"


class SemanticHardCaseError(RuntimeError):
    """Raised when a frozen BL10 identity or information boundary drifts."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_exact_json(root: Path, path: str, expected_sha256: str) -> Any:
    resolved = root / path
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise SemanticHardCaseError(f"identity drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        config.get("schema") != "neuro_film.u5_r2bl10_filmmatch_semantic_hard_case_development.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("product_integration_allowed")
        or config["feature_model"].get("training_allowed")
        or config["feature_model"].get("fine_tuning_allowed")
        or config["selector"].get("fitting_allowed")
        or config["selector"].get("dense_blending_allowed")
        or config["selector"].get("source_output_diagnostics_allowed")
    ):
        raise SemanticHardCaseError("BL10 frozen contract drift")
    parent = config["parent"]
    decision = _load_exact_json(root, parent["decision"], parent["decision_sha256"])
    if (
        decision.get("decision") != parent["required_decision"]
        or decision.get("preference_passed") is not parent["required_preference_passed"]
        or decision.get("nonbasic_passed") is not parent["required_nonbasic_passed"]
    ):
        raise SemanticHardCaseError("BL8 decision does not open BL10")
    manifest = _load_exact_json(root, parent["source_manifest"], parent["source_manifest_sha256"])
    rows = {str(row["id"]): row for row in manifest}
    source_ids = list(decision["per_source_candidate_choices"])
    if len(source_ids) != 17 or not set(source_ids).issubset(rows):
        raise SemanticHardCaseError("source inventory drift")
    model = config["feature_model"]
    model_root = root / model["root"]
    for filename, key in (
        ("model.safetensors", "model_sha256"),
        ("config.json", "config_sha256"),
        ("preprocessor_config.json", "preprocessor_sha256"),
    ):
        if sha256_file(model_root / filename) != model[key]:
            raise SemanticHardCaseError(f"model asset drift: {filename}")
    return decision, [rows[source_id] for source_id in source_ids]


def _source_view(path: Path, *, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = 256.0 / min(width, height)
        resized = rgb.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        crop = resized.crop((left, top, left + size, top + size))
        return np.asarray(crop, dtype=np.float32) / 255.0


def extract_semantic_features(
    *, root: Path, rows: list[dict[str, Any]], config: Mapping[str, Any]
) -> np.ndarray:
    spec = config["feature_model"]
    device = str(spec["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise SemanticHardCaseError("frozen CUDA device unavailable")
    model = AutoModel.from_pretrained(root / spec["root"], local_files_only=True).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    batches: list[np.ndarray] = []
    batch_size = int(spec["batch_size"])
    for start in range(0, len(rows), batch_size):
        views = []
        for row in rows[start : start + batch_size]:
            path = root / row["decoded_path"]
            if not path.is_file() or sha256_file(path) != row["decoded_sha256"]:
                raise SemanticHardCaseError("decoded source identity drift")
            views.append(_source_view(path, size=int(spec["image_size"])))
        colours = torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2)
        luma = 0.2126 * colours[:, 0:1] + 0.7152 * colours[:, 1:2] + 0.0722 * colours[:, 2:3]
        inputs = ((luma.repeat(1, 3, 1, 1) - mean) / std).to(device)
        with torch.inference_mode():
            tokens = model(pixel_values=inputs).last_hidden_state[:, 1:].float()
            pooled = F.normalize(tokens.mean(dim=1), dim=-1)
        batches.append(pooled.cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    output = np.concatenate(batches).astype(np.float32, copy=False)
    if output.shape[0] != len(rows) or not np.all(np.isfinite(output)):
        raise SemanticHardCaseError("semantic feature extraction failed")
    return output


def hard_top1_predictions(features: np.ndarray, labels: np.ndarray, source_ids: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    if values.ndim != 2 or len(values) != len(truth) or len(source_ids) != len(truth) or len(truth) < 3:
        raise ValueError("invalid hard-case inputs")
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(values)) or np.any(norms <= 0.0):
        raise ValueError("features must be finite and nonzero")
    normalized = values / norms[:, None]
    predictions = np.zeros(len(truth), dtype=bool)
    rows = []
    for held in range(len(truth)):
        candidates = [index for index in range(len(truth)) if index != held]
        candidates.sort(key=lambda index: (-float(normalized[held] @ normalized[index]), source_ids[index]))
        nearest = candidates[0]
        predictions[held] = truth[nearest]
        rows.append(
            {
                "source_id": source_ids[held],
                "nearest_source_id": source_ids[nearest],
                "nearest_cosine_similarity": float(normalized[held] @ normalized[nearest]),
                "selected_arm": CANDIDATE if predictions[held] else FALLBACK,
            }
        )
    return predictions, rows


def _utility(predictions: np.ndarray, candidate_choices: np.ndarray) -> int:
    return int(np.sum(np.where(predictions, candidate_choices, 3 - candidate_choices)))


def evaluate_selector(
    *, features: np.ndarray, source_ids: list[str], candidate_choices: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    choices = np.asarray(candidate_choices, dtype=np.int64)
    labels = choices >= 2
    predictions, rows = hard_top1_predictions(features, labels, source_ids)
    global_utility = int(np.sum(3 - choices))
    selector_utility = _utility(predictions, choices)
    oracle_utility = int(np.sum(np.maximum(choices, 3 - choices)))
    global_correct = int(np.sum(~labels))
    selector_correct = int(np.sum(predictions == labels))
    false_candidate = int(np.sum(predictions & ~labels))

    rng = random.Random(int(config["controls"]["negative_control_seed"]))
    negative_gains = []
    for _ in range(256):
        permuted = labels.copy()
        rng.shuffle(permuted)
        control_predictions, _ = hard_top1_predictions(features, permuted, source_ids)
        negative_gains.append(_utility(control_predictions, choices) - global_utility)
    negative_p95 = float(np.quantile(np.asarray(negative_gains, dtype=np.float64), 0.95, method="higher"))
    gate = config["automatic_gate"]
    checks = {
        "choice_gain": selector_utility - global_utility >= int(gate["minimum_selector_choice_gain_over_global_ao6"]),
        "correct_source_gain": selector_correct - global_correct >= int(gate["minimum_selector_correct_source_gain_over_global_ao6"]),
        "candidate_support": int(np.sum(predictions)) >= int(gate["minimum_candidate_selected_sources"]),
        "false_candidate": false_candidate <= int(gate["maximum_false_candidate_selections"]),
        "negative_control": selector_utility - global_utility - negative_p95 >= int(gate["minimum_choice_gain_over_negative_control_p95"]),
        "finite_distances": all(np.isfinite(row["nearest_cosine_similarity"]) for row in rows),
        "zero_self_matches": all(row["source_id"] != row["nearest_source_id"] for row in rows),
    }
    return {
        "global_ao6_choice_utility": global_utility,
        "selector_choice_utility": selector_utility,
        "oracle_choice_utility": oracle_utility,
        "selector_choice_gain": selector_utility - global_utility,
        "global_ao6_correct_sources": global_correct,
        "selector_correct_sources": selector_correct,
        "selector_correct_source_gain": selector_correct - global_correct,
        "candidate_selected_sources": int(np.sum(predictions)),
        "false_candidate_selections": false_candidate,
        "negative_control_choice_gain_p95": negative_p95,
        "negative_control_choice_gains": negative_gains,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
    }


def run_experiment(*, root: Path, config: Mapping[str, Any], config_path: Path, software_commit: str) -> dict[str, Any]:
    decision, source_rows = validate_contract(root, config)
    source_ids = list(decision["per_source_candidate_choices"])
    choices = np.asarray([decision["per_source_candidate_choices"][source_id] for source_id in source_ids])
    features = extract_semantic_features(root=root, rows=source_rows, config=config)
    result = evaluate_selector(features=features, source_ids=source_ids, candidate_choices=choices, config=config)
    core = {
        "schema": "neuro_film.u5_r2bl10_filmmatch_semantic_hard_case_report.v1",
        "software_commit": software_commit,
        "config_sha256": sha256_file(config_path),
        "parent_decision_sha256": config["parent"]["decision_sha256"],
        "feature_sha256": hashlib.sha256(features.astype("<f4", copy=False).tobytes()).hexdigest(),
        **result,
        "decision": (
            "retain_semantic_hard_case_for_fresh_confirmation"
            if result["passed"]
            else "close_semantic_hard_case_retain_ao6"
        ),
        "product_integration_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    core["stable_evidence_id"] = _canonical_sha256(core)
    return core


__all__ = ["evaluate_selector", "hard_top1_predictions", "run_experiment", "validate_contract"]
