"""Real-raster FiveK control for the BK22 content-cell appearance gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from src.eval.canonicalizer_consensus_appearance import (
    _grid,
    _normalized_cross_objective_score,
    canonical_json_bytes,
    sha256_file,
)
from src.eval.content_cell_appearance_gate import (
    _canonical_set_sha256,
    _cell_scores,
)
from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from src.roll2film.filmset_recipe_explainability import evaluate_flow_structure
from src.roll2film.unpaired_distribution_flow import (
    DistributionLossAssets,
    fit_unpaired_distribution_flow,
    make_distribution_loss_assets,
)


SCHEMA = "neuro_film.u5_r2bk23_fivek_content_cell_appearance_control_report.v1"


class FiveKContentCellControlError(ValueError):
    """Raised when the frozen BK23 contract or evidence is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_hash(root: Path, item: dict[str, str]) -> None:
    path = root / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise FiveKContentCellControlError(f"evidence hash drift: {item['path']}")


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk23_fivek_content_cell_appearance_control.v1"
        or config.get("status") != "contract_frozen"
    ):
        raise FiveKContentCellControlError("BK23 contract is not frozen")
    parent = config["parent"]
    _validate_hash(
        root,
        {"path": parent["decision"], "sha256": parent["decision_sha256"]},
    )
    if _read_json(root / parent["decision"]).get("decision") != parent[
        "required_decision"
    ]:
        raise FiveKContentCellControlError("BK23 parent decision mismatch")
    source = config["source"]
    _validate_hash(
        root,
        {
            "path": source["development_manifest"],
            "sha256": source["development_manifest_sha256"],
        },
    )
    _validate_hash(
        root,
        {
            "path": source["confirmation_manifest"],
            "sha256": source["confirmation_manifest_sha256"],
        },
    )
    for item in source["licence_files"]:
        _validate_hash(root, item)
    if (
        int(source["development_rows"]) != 64
        or int(source["confirmation_rows"]) != 64
        or int(config["confounded_control"]["rotation"]) % 64 == 0
        or len(config["canonicalizers"]) != 2
    ):
        raise FiveKContentCellControlError("BK23 frozen boundary drift")
    if config["sampling"]["paired_pixel_coordinates_used"]:
        raise FiveKContentCellControlError("paired pixels are forbidden")
    return config


def _licensed_names(root: Path, config: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in config["source"]["licence_files"]:
        if not Path(item["path"]).name.startswith("files"):
            continue
        names.update(
            line.strip()
            for line in (root / item["path"]).read_text(
                encoding="utf-8", errors="strict"
            ).splitlines()
            if line.strip()
        )
    return names


def load_row_inventory(
    root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = config["source"]
    with (root / source["development_manifest"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        raw_development = list(csv.DictReader(handle))
    development = []
    for row in raw_development[: int(source["development_rows"])]:
        development.append(
            {
                "cell_id": row["source_name"],
                "source_path": row[source["development_source_field"]],
                "target_path": row[source["development_target_field"]],
            }
        )

    confirmation_manifest = _read_json(root / source["confirmation_manifest"])
    confirmation = []
    for row in confirmation_manifest["rows"]:
        confirmation.append(
            {
                "cell_id": row["source_name"],
                "source_path": row[source["confirmation_source_field"]],
                "target_path": row[source["confirmation_target_field"]],
                "source_sha256": row["source_sha256"],
                "target_sha256": row["target_sha256"],
            }
        )
    if len(development) != int(source["development_rows"]) or len(
        confirmation
    ) != int(source["confirmation_rows"]):
        raise FiveKContentCellControlError("BK23 row count drift")
    development_ids = {row["cell_id"] for row in development}
    confirmation_ids = {row["cell_id"] for row in confirmation}
    if len(development_ids) != len(development) or len(confirmation_ids) != len(
        confirmation
    ):
        raise FiveKContentCellControlError("duplicate FiveK identity")
    if development_ids & confirmation_ids:
        raise FiveKContentCellControlError("development/confirmation overlap")
    licensed = _licensed_names(root, config)
    missing_licence = sorted((development_ids | confirmation_ids) - licensed)
    if missing_licence:
        raise FiveKContentCellControlError(
            f"FiveK identity absent from licence inventory: {missing_licence[0]}"
        )
    return development, confirmation


def _sample_tiff(
    path: Path,
    *,
    count: int,
    seed: int,
    expected_sha256: str | None,
) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise FiveKContentCellControlError(f"missing raster: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FiveKContentCellControlError(f"raster hash drift: {path}")
    image = tifffile.imread(path)
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 3:
        raise FiveKContentCellControlError(f"invalid FiveK raster: {path}")
    pixels = image.reshape(-1, 3)
    if pixels.shape[0] < count:
        raise FiveKContentCellControlError(f"insufficient raster pixels: {path}")
    indices = np.random.default_rng(seed).choice(
        pixels.shape[0], size=count, replace=False
    )
    sampled = np.asarray(pixels[indices], dtype=np.float64) / 65535.0
    if not np.isfinite(sampled).all():
        raise FiveKContentCellControlError(f"non-finite raster: {path}")
    return sampled, digest


def load_sampled_cells(
    root: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    source_seed: int,
    target_seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    count = int(config["sampling"]["pixels_per_cell"])
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identities = []
    for index, row in enumerate(rows):
        source, source_sha = _sample_tiff(
            root / row["source_path"],
            count=count,
            seed=source_seed + index,
            expected_sha256=row.get("source_sha256"),
        )
        target, target_sha = _sample_tiff(
            root / row["target_path"],
            count=count,
            seed=target_seed + index,
            expected_sha256=row.get("target_sha256"),
        )
        sources.append(source)
        targets.append(target)
        identities.append(
            {
                "cell_id": row["cell_id"],
                "source_sha256": source_sha,
                "target_sha256": target_sha,
            }
        )
    lineage_sha = hashlib.sha256(canonical_json_bytes(identities)).hexdigest()
    return sources, targets, lineage_sha


def _policy_result(
    config: dict[str, Any],
    *,
    disagreement: float,
    pooled_improvement: float,
    cell_improvements: list[float],
) -> tuple[str, list[str]]:
    policy = config["policy"]
    reasons = []
    if disagreement > float(policy["maximum_canonicalizer_operator_rmse"]):
        reasons.append("canonicalizer-disagreement")
    if pooled_improvement < float(
        policy["minimum_pooled_cross_objective_improvement_fraction"]
    ):
        reasons.append("insufficient-pooled-improvement")
    threshold = float(
        policy["minimum_cell_cross_objective_improvement_fraction"]
    )
    pass_fraction = float(np.mean(np.asarray(cell_improvements) >= threshold))
    if pass_fraction < float(policy["minimum_cell_pass_fraction"]):
        reasons.append("content-cell-worst-case")
    return (
        ("identity_fallback", reasons)
        if reasons
        else ("hard_explicit_candidate", [])
    )


def evaluate(
    root: Path,
    config: dict[str, Any],
    *,
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    development_rows, confirmation_rows = load_row_inventory(root, config)
    sampling = config["sampling"]
    development_source, development_target, development_lineage = (
        load_sampled_cells(
            root,
            config,
            development_rows,
            source_seed=int(sampling["development_source_seed"]),
            target_seed=int(sampling["development_target_seed"]),
        )
    )
    confirmation_source, confirmation_target, confirmation_lineage = (
        load_sampled_cells(
            root,
            config,
            confirmation_rows,
            source_seed=int(sampling["confirmation_source_seed"]),
            target_seed=int(sampling["confirmation_target_seed"]),
        )
    )
    pooled_fit_source = np.concatenate(development_source, axis=0)
    pooled_fit_target = np.concatenate(development_target, axis=0)
    pooled_fit_target = pooled_fit_target[
        np.random.default_rng(int(sampling["fit_target_order_seed"])).permutation(
            pooled_fit_target.shape[0]
        )
    ]
    pooled_confirmation_source = np.concatenate(confirmation_source, axis=0)
    pooled_confirmation_target = np.concatenate(confirmation_target, axis=0)

    assets: dict[str, DistributionLossAssets] = {
        spec["id"]: make_distribution_loss_assets(spec)
        for spec in config["canonicalizers"]
    }
    operator_spec = config["operator"]
    optimization = config["optimization"]
    device = str(optimization["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise FiveKContentCellControlError("frozen CUDA device unavailable")
    candidates: dict[str, CubeDiffeomorphicColourFlow] = {}
    candidate_rows: dict[str, Any] = {}
    for index, spec in enumerate(config["canonicalizers"]):
        candidate, trace = fit_unpaired_distribution_flow(
            pooled_fit_source,
            pooled_fit_target,
            loss_assets=assets[spec["id"]],
            axis_size=int(operator_spec["velocity_grid_axis_size"]),
            integration_steps=int(operator_spec["integration_steps"]),
            coefficient_vector_norm_cap=float(
                operator_spec["coefficient_vector_norm_cap"]
            ),
            steps=int(optimization["steps"]),
            learning_rate=float(optimization["learning_rate"]),
            coefficient_l2=float(optimization["coefficient_l2"]),
            velocity_smoothness_l2=float(
                optimization["velocity_smoothness_l2"]
            ),
            gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            seed=int(optimization["seed"]) + index,
            device=device,
            deterministic_algorithms=bool(
                optimization["deterministic_algorithms"]
            ),
            optimization_dtype=str(optimization["dtype"]),
        )
        pooled_ratio, pooled_rows = _normalized_cross_objective_score(
            candidate,
            pooled_confirmation_source,
            pooled_confirmation_target,
            assets,
        )
        candidates[spec["id"]] = candidate
        candidate_rows[spec["id"]] = {
            "pooled_mean_normalized_residual_ratio": pooled_ratio,
            "pooled_improvement_fraction": 1.0 - pooled_ratio,
            "pooled_objectives": pooled_rows,
            "fit_trace": trace,
        }

    points = _grid(int(operator_spec["evaluation_grid_axis_size"]))
    names = tuple(candidates)
    disagreement = float(
        np.sqrt(
            np.mean(
                np.square(
                    candidates[names[0]].apply(points)
                    - candidates[names[1]].apply(points)
                )
            )
        )
    )
    selected_name = min(
        candidate_rows,
        key=lambda name: candidate_rows[name][
            "pooled_mean_normalized_residual_ratio"
        ],
    )
    selected = candidates[selected_name]
    valid_improvements, valid_rows = _cell_scores(
        selected, confirmation_source, confirmation_target, assets
    )
    rotation = int(config["confounded_control"]["rotation"])
    confounded_target = confirmation_target[rotation:] + confirmation_target[:rotation]
    confounded_improvements, confounded_rows = _cell_scores(
        selected, confirmation_source, confounded_target, assets
    )
    pooled_improvement = float(
        candidate_rows[selected_name]["pooled_improvement_fraction"]
    )
    valid_policy, valid_reasons = _policy_result(
        config,
        disagreement=disagreement,
        pooled_improvement=pooled_improvement,
        cell_improvements=valid_improvements,
    )
    confounded_policy, confounded_reasons = _policy_result(
        config,
        disagreement=disagreement,
        pooled_improvement=pooled_improvement,
        cell_improvements=confounded_improvements,
    )
    valid_target_sha = _canonical_set_sha256(confirmation_target)
    confounded_target_sha = _canonical_set_sha256(confounded_target)
    structure = evaluate_flow_structure(
        [selected],
        coefficient_cap=float(operator_spec["coefficient_vector_norm_cap"]),
    )
    gates = config["gates"]
    checks = [
        {
            "name": "development_confirmation_identity_overlap_zero",
            "passed": not (
                {row["cell_id"] for row in development_rows}
                & {row["cell_id"] for row in confirmation_rows}
            ),
        },
        {
            "name": "pooled_target_multiset_exact",
            "passed": valid_target_sha == confounded_target_sha,
        },
        {
            "name": "confounded_correspondence_returns_identity",
            "passed": confounded_policy == "identity_fallback",
        },
        {
            "name": "selected_output_in_cube",
            "passed": structure["minimum_output"] >= float(gates["minimum_output"])
            and structure["maximum_output"] <= float(gates["maximum_output"]),
        },
        {
            "name": "selected_positive_jacobian",
            "passed": structure["minimum_jacobian_determinant"]
            > float(gates["minimum_jacobian_determinant_exclusive"]),
        },
        {
            "name": "selected_bounded_jacobian_norm",
            "passed": structure["maximum_jacobian_spectral_norm"]
            <= float(gates["maximum_jacobian_spectral_norm"]),
        },
        {
            "name": "selected_inverse",
            "passed": structure["maximum_inverse_error"]
            <= float(gates["maximum_inverse_error"]),
        },
        {
            "name": "selected_replay",
            "passed": structure["maximum_replay_error"]
            <= float(gates["maximum_replay_error"]),
        },
    ]
    automatic_pass = all(check["passed"] for check in checks)
    if not automatic_pass:
        decision = "close_invalid_or_unsafe_real_raster_control"
    elif valid_policy == "hard_explicit_candidate":
        decision = "retain_real_raster_content_cell_positive_control"
    else:
        decision = "retain_real_raster_k1_identity_outcome"
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "software_commit": software_commit,
        "device": device,
        "development_cells": len(development_rows),
        "confirmation_cells": len(confirmation_rows),
        "pixels_per_cell": int(sampling["pixels_per_cell"]),
        "development_lineage_sha256": development_lineage,
        "confirmation_lineage_sha256": confirmation_lineage,
        "development_confirmation_identity_overlap": 0,
        "paired_pixel_coordinates_used": False,
        "fit_target_order_destroyed": True,
        "valid_pooled_target_set_sha256": valid_target_sha,
        "confounded_pooled_target_set_sha256": confounded_target_sha,
        "pooled_target_sets_exact": valid_target_sha == confounded_target_sha,
        "canonicalizer_operator_rmse": disagreement,
        "selected_canonicalizer": selected_name,
        "candidate_metrics": candidate_rows,
        "valid_correspondence": {
            "policy_result": valid_policy,
            "fallback_reasons": valid_reasons,
            "cell_improvements": valid_improvements,
            "worst_cell_improvement_fraction": float(min(valid_improvements)),
            "cell_pass_fraction": float(
                np.mean(
                    np.asarray(valid_improvements)
                    >= float(
                        config["policy"][
                            "minimum_cell_cross_objective_improvement_fraction"
                        ]
                    )
                )
            ),
            "cells": valid_rows,
        },
        "confounded_correspondence": {
            "policy_result": confounded_policy,
            "fallback_reasons": confounded_reasons,
            "cell_improvements": confounded_improvements,
            "worst_cell_improvement_fraction": float(
                min(confounded_improvements)
            ),
            "cell_pass_fraction": float(
                np.mean(
                    np.asarray(confounded_improvements)
                    >= float(
                        config["policy"][
                            "minimum_cell_cross_objective_improvement_fraction"
                        ]
                    )
                )
            ),
            "cells": confounded_rows,
        },
        "selected_identity_rmse": float(
            np.sqrt(np.mean(np.square(selected.apply(points) - points)))
        ),
        "selected_structure": structure,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "shared_candidate_supported": valid_policy == "hard_explicit_candidate",
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }


def run(
    root: Path,
    config_path: Path,
    output_path: Path,
    *,
    software_commit: str,
) -> dict[str, Any]:
    config = load_config(root, config_path)
    report = evaluate(
        root,
        config,
        config_sha256=sha256_file(config_path),
        software_commit=software_commit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    return report


__all__ = [
    "FiveKContentCellControlError",
    "evaluate",
    "load_config",
    "load_row_inventory",
    "load_sampled_cells",
    "run",
]
