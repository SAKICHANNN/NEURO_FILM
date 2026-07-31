"""Colour-blind semantic pseudo-pairs for a bounded explicit colour flow."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from transformers import AutoModel

from src.eval.canonicalizer_consensus_appearance import canonical_json_bytes, sha256_file
from src.roll2film.filmset_recipe_explainability import evaluate_flow_structure
from src.roll2film.hierarchical_colour_coupling import fit_paired_cube_diffeomorphic_flow


SCHEMA = "neuro_film.u5_r2bk24_fivek_semantic_pseudopair_flow_report.v1"


class FiveKSemanticPseudoPairError(ValueError):
    """Raised when the frozen BK24 contract or evidence is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(root: Path, path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema") != "neuro_film.u5_r2bk24_fivek_semantic_pseudopair_flow.v1":
        raise FiveKSemanticPseudoPairError("invalid BK24 contract")
    for item in (
        config["parent"],
        {
            "decision": config["source"]["manifest"],
            "decision_sha256": config["source"]["manifest_sha256"],
        },
    ):
        candidate = root / item["decision"]
        if not candidate.is_file() or sha256_file(candidate) != item["decision_sha256"]:
            raise FiveKSemanticPseudoPairError(f"evidence hash drift: {candidate}")
    parent = _read_json(root / config["parent"]["decision"])
    if parent.get("decision") != config["parent"]["required_decision"]:
        raise FiveKSemanticPseudoPairError("parent decision mismatch")
    model = config["feature_model"]
    model_root = root / model["root"]
    for name, key in (
        ("model.safetensors", "model_sha256"),
        ("config.json", "config_sha256"),
        ("preprocessor_config.json", "preprocessor_sha256"),
    ):
        if sha256_file(model_root / name) != model[key]:
            raise FiveKSemanticPseudoPairError(f"model asset drift: {name}")
    return config


def load_rows(root: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    source = config["source"]
    with (root / source["manifest"]).open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))[: int(source["rows"])]
    rows = [
        {
            "cell_id": row["source_name"],
            "source_path": row[source["source_field"]],
            "target_path": row[source["target_field"]],
        }
        for row in raw
    ]
    if len(rows) != int(source["rows"]) or len({row["cell_id"] for row in rows}) != len(rows):
        raise FiveKSemanticPseudoPairError("FiveK row inventory drift")
    return rows


def _resize_crop(image: np.ndarray, size: int = 224) -> np.ndarray:
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    height, width = image.shape[:2]
    scale = 256.0 / min(height, width)
    resized = F.interpolate(
        tensor,
        size=(round(height * scale), round(width * scale)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized.shape[-2] - size) // 2
    left = (resized.shape[-1] - size) // 2
    return resized[0, :, top : top + size, left : left + size].permute(1, 2, 0).numpy()


def _load_view(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise FiveKSemanticPseudoPairError(f"invalid raster: {path}")
    return _resize_crop(np.asarray(image, dtype=np.float32) / 65535.0)


def _extract(
    root: Path,
    rows: list[dict[str, str]],
    field: str,
    model: torch.nn.Module,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_features, all_colours = [], []
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for start in range(0, len(rows), batch_size):
        views = [_load_view(root / row[field]) for row in rows[start : start + batch_size]]
        colours = torch.from_numpy(np.stack(views)).permute(0, 3, 1, 2)
        luma = (
            0.2126 * colours[:, 0:1]
            + 0.7152 * colours[:, 1:2]
            + 0.0722 * colours[:, 2:3]
        )
        inputs = ((luma.repeat(1, 3, 1, 1) - mean) / std).to(device)
        with torch.inference_mode():
            tokens = model(pixel_values=inputs).last_hidden_state[:, 1:]
            tokens = F.normalize(tokens.float(), dim=-1)
        patch_colours = F.adaptive_avg_pool2d(colours, (16, 16))
        all_features.append(tokens.cpu().numpy())
        all_colours.append(
            patch_colours.permute(0, 2, 3, 1).reshape(len(views), 256, 3).numpy()
        )
    return np.concatenate(all_features), np.concatenate(all_colours)


def _sinkhorn(cost: np.ndarray, regularization: float, iterations: int) -> np.ndarray:
    kernel = np.exp(-cost / regularization).clip(1e-12)
    a = np.full(cost.shape[0], 1.0 / cost.shape[0])
    b = np.full(cost.shape[1], 1.0 / cost.shape[1])
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(iterations):
        u = a / np.maximum(kernel @ v, 1e-12)
        v = b / np.maximum(kernel.T @ u, 1e-12)
    plan = (u[:, None] * kernel) * v[None, :]
    return plan / np.maximum(plan.sum(axis=1, keepdims=True), 1e-12)


def build_pseudo_pairs(
    source_features: np.ndarray,
    source_colours: np.ndarray,
    target_features: np.ndarray,
    target_colours: np.ndarray,
    *,
    topk_images: int,
    regularization: float,
    iterations: int,
    control_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    image_source = source_features.mean(axis=1)
    image_target = target_features.mean(axis=1)
    similarity = image_source @ image_target.T
    np.fill_diagonal(similarity, -np.inf)
    chosen = np.argsort(-similarity, axis=1)[:, :topk_images]
    paired_source, semantic_target, selected_ids = [], [], []
    for index, candidates in enumerate(chosen):
        features = target_features[candidates].reshape(-1, target_features.shape[-1])
        colours = target_colours[candidates].reshape(-1, 3)
        cost = 1.0 - source_features[index] @ features.T
        weights = _sinkhorn(cost, regularization, iterations)
        paired_source.append(source_colours[index])
        semantic_target.append(weights @ colours)
        selected_ids.extend(np.repeat(candidates, target_features.shape[1]).tolist())
    source = np.concatenate(paired_source)
    semantic = np.concatenate(semantic_target)
    permutation = np.random.default_rng(control_seed).permutation(len(semantic))
    control = semantic[permutation]
    counts = Counter(selected_ids)
    def multiset_sha256(values: np.ndarray) -> str:
        ordered = values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]
        return hashlib.sha256(ordered.astype("<f4").tobytes()).hexdigest()

    return source, semantic, control, {
        "same_identity_matches": int(sum(index in row for index, row in enumerate(chosen))),
        "distinct_target_images": len(counts),
        "largest_target_share": max(counts.values()) / sum(counts.values()),
        "target_colour_multiset_sha256": multiset_sha256(semantic),
        "control_colour_multiset_sha256": multiset_sha256(control),
    }


def _paired_metrics(operator: Any, sources: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    rows = []
    for source, target in zip(sources, targets):
        before = float(np.mean((source - target) ** 2))
        output = operator.apply(source.reshape(-1, 3)).reshape(source.shape)
        after = float(np.mean((output - target) ** 2))
        improvement = 1.0 - after / max(before, 1e-12)
        source_boundary = np.any(
            (source <= 1.0 / 65535.0)
            | (source >= 1.0 - 1.0 / 65535.0),
            axis=-1,
        )
        output_boundary = np.any(
            (output <= 1.0 / 65535.0)
            | (output >= 1.0 - 1.0 / 65535.0),
            axis=-1,
        )
        boundary = float(np.mean(output_boundary & ~source_boundary))
        rows.append({"before_mse": before, "after_mse": after, "improvement_fraction": improvement, "new_boundary_fraction": boundary})
    improvements = np.asarray([row["improvement_fraction"] for row in rows])
    return {
        "rows": rows,
        "median_improvement_fraction": float(np.median(improvements)),
        "worst_improvement_fraction": float(np.min(improvements)),
        "pass_fraction": float(np.mean(improvements >= 0.0)),
        "maximum_new_boundary_fraction": float(max(row["new_boundary_fraction"] for row in rows)),
    }


def evaluate(root: Path, config: dict[str, Any], *, config_sha256: str, software_commit: str) -> dict[str, Any]:
    rows = load_rows(root, config)
    fit_count = int(config["source"]["fit_rows"])
    fit_rows, evaluation_rows = rows[:fit_count], rows[fit_count:]
    feature = config["feature_model"]
    device = feature["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise FiveKSemanticPseudoPairError("frozen CUDA device unavailable")
    model = AutoModel.from_pretrained(root / feature["root"], local_files_only=True).to(device).eval()
    source_features, source_colours = _extract(root, fit_rows, "source_path", model, device=device, batch_size=int(feature["batch_size"]))
    target_features, target_colours = _extract(root, fit_rows, "target_path", model, device=device, batch_size=int(feature["batch_size"]))
    del model
    torch.cuda.empty_cache()
    pseudo_source, semantic_target, control_target, retrieval = build_pseudo_pairs(
        source_features,
        source_colours,
        target_features,
        target_colours,
        topk_images=int(feature["topk_images"]),
        regularization=float(feature["sinkhorn_regularization"]),
        iterations=int(feature["sinkhorn_iterations"]),
        control_seed=int(config["control"]["seed"]),
    )
    spec = config["operator"]
    fit_kwargs = dict(
        axis_size=int(spec["velocity_grid_axis_size"]),
        integration_steps=int(spec["integration_steps"]),
        coefficient_vector_norm_cap=float(spec["coefficient_vector_norm_cap"]),
        steps=int(spec["steps"]),
        learning_rate=float(spec["learning_rate"]),
        coefficient_l2=float(spec["coefficient_l2"]),
        velocity_smoothness_l2=float(spec["velocity_smoothness_l2"]),
        gradient_clip_norm=float(spec["gradient_clip_norm"]),
        seed=int(spec["seed"]),
        device=str(spec["device"]),
        deterministic_algorithms=True,
        optimization_dtype=str(spec["dtype"]),
    )
    semantic_operator, semantic_trace = fit_paired_cube_diffeomorphic_flow(pseudo_source, semantic_target, **fit_kwargs)
    control_operator, control_trace = fit_paired_cube_diffeomorphic_flow(pseudo_source, control_target, **{**fit_kwargs, "seed": int(spec["seed"]) + 1})
    evaluation_sources = np.stack([_load_view(root / row["source_path"]) for row in evaluation_rows])
    evaluation_targets = np.stack([_load_view(root / row["target_path"]) for row in evaluation_rows])
    semantic_metrics = _paired_metrics(semantic_operator, evaluation_sources, evaluation_targets)
    control_metrics = _paired_metrics(control_operator, evaluation_sources, evaluation_targets)
    structure = evaluate_flow_structure(
        [semantic_operator],
        coefficient_cap=float(spec["coefficient_vector_norm_cap"]),
    )
    gates = config["gates"]
    checks = {
        "zero_same_identity": retrieval["same_identity_matches"] == 0,
        "target_diversity": retrieval["distinct_target_images"] >= int(gates["minimum_distinct_target_images"]) and retrieval["largest_target_share"] <= float(gates["maximum_largest_target_share"]),
        "control_multiset": retrieval["target_colour_multiset_sha256"] == retrieval["control_colour_multiset_sha256"],
        "median": semantic_metrics["median_improvement_fraction"] >= float(gates["minimum_semantic_median_improvement_fraction"]),
        "worst": semantic_metrics["worst_improvement_fraction"] >= float(gates["minimum_semantic_worst_improvement_fraction"]),
        "pass_fraction": semantic_metrics["pass_fraction"] >= float(gates["minimum_semantic_pass_fraction"]),
        "control_advantage": semantic_metrics["median_improvement_fraction"] - control_metrics["median_improvement_fraction"] >= float(gates["minimum_semantic_minus_control_median_improvement"]),
        "boundary": semantic_metrics["maximum_new_boundary_fraction"] <= float(gates["maximum_new_boundary_fraction"]),
        "structure": structure["minimum_jacobian_determinant"] > float(gates["minimum_jacobian_determinant_exclusive"]) and structure["maximum_jacobian_spectral_norm"] <= float(gates["maximum_jacobian_spectral_norm"]) and structure["maximum_inverse_error"] <= float(gates["maximum_inverse_error"]) and structure["maximum_replay_error"] <= float(gates["maximum_replay_error"]),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "split": {"fit_ids": [row["cell_id"] for row in fit_rows], "evaluation_ids": [row["cell_id"] for row in evaluation_rows]},
        "retrieval": retrieval,
        "semantic": {"operator": semantic_operator.to_dict(), "fit_trace": semantic_trace, "metrics": semantic_metrics},
        "control": {"operator": control_operator.to_dict(), "fit_trace": control_trace, "metrics": control_metrics},
        "structure": structure,
        "checks": checks,
        "passed": passed,
        "decision": "open_disjoint_semantic_pseudopair_confirmation" if passed else "retain_k1_identity_close_fixed_semantic_pseudopair",
        "claim_ceiling": config["claim_ceiling"],
    }


def report_bytes(report: dict[str, Any]) -> bytes:
    return canonical_json_bytes(report)


__all__ = ["FiveKSemanticPseudoPairError", "build_pseudo_pairs", "evaluate", "load_config", "load_rows", "report_bytes"]
