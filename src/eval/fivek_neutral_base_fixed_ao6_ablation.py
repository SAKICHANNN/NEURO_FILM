"""Fixed-AO6 downstream ablation for the bounded FiveK neutral base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.color import rgb2lab

from scripts.build_fivek_freeze_pack import (
    filtered_target,
    load_expert_icc_srgb,
    load_raw_default,
    resize_to_shape,
)
from src.eval.b0_real_film_residual_frontier import (
    validate_contract as validate_ao6_contract,
)
from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.dual_champion_composition import (
    build_operators,
    compose_rgb,
    validate_contract as validate_base_contract,
)
from src.eval.fivek_neutral_base_parameter_pilot import apply_neutral_base
from src.roll2film.factorized_boundary_guard import (
    apply_factorized_boundary_guard,
)


class FiveKFixedAO6AblationError(ValueError):
    """Raised when the frozen ablation contract or evidence drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(
    root: Path, path: str, expected_sha256: str
) -> dict[str, Any]:
    resolved = root / path
    if not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise FiveKFixedAO6AblationError(f"evidence drift: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKFixedAO6AblationError("ablation contract is not frozen")
    if (
        config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("final_rgb_learning_allowed")
        or config.get("production_integration_allowed")
        or config["fixed_look"].get("operator_refit_allowed")
        or config["fixed_look"].get("strength_change_allowed")
        or config["fixed_look"].get("per_image_style_adjustment_allowed")
    ):
        raise FiveKFixedAO6AblationError("frozen no-fit boundary drift")
    neutral = config["neutral_base"]
    neutral_config = _load_hashed_json(
        root, neutral["config"], neutral["config_sha256"]
    )
    decision = _load_hashed_json(
        root, neutral["decision"], neutral["decision_sha256"]
    )
    report = _load_hashed_json(
        root, neutral["report"], neutral["report_sha256"]
    )
    if (
        report.get("stable_evidence_id")
        != neutral["required_stable_evidence_id"]
        or report.get("automatic_pass")
        is not neutral["required_automatic_pass"]
        or decision.get("automatic_pass") is not True
    ):
        raise FiveKFixedAO6AblationError("neutral-base evidence is not eligible")
    fixed = config["fixed_look"]
    base_config = _load_hashed_json(
        root, fixed["base_config"], fixed["base_config_sha256"]
    )
    ao6_config = _load_hashed_json(
        root, fixed["ao6_config"], fixed["ao6_config_sha256"]
    )
    base_validated = validate_base_contract(root, base_config)
    ao6_validated = validate_ao6_contract(root, ao6_config)
    selected = [
        row
        for row in ao6_validated["candidates"]
        if row["candidate_id"] == fixed["ao6_candidate_id"]
    ]
    if (
        len(selected) != 1
        or selected[0]["tone_strength"] != fixed["tone_strength"]
        or selected[0]["chroma_strength"] != fixed["chroma_strength"]
        or fixed["base_candidate_id"] != ao6_config["base"]["candidate_id"]
    ):
        raise FiveKFixedAO6AblationError("fixed AO6 candidate drift")
    return {
        "neutral_config": neutral_config,
        "neutral_report": report,
        "base_config": base_config,
        "base_validated": base_validated,
        "ao6_config": ao6_config,
        "ao6_validated": ao6_validated,
    }


def build_fixed_ao6_renderer(
    config: Mapping[str, Any], validated: Mapping[str, Any]
) -> Any:
    """Build the unchanged AO6 path, including its frozen RGB8 B0 boundary."""

    apply_anchor, apply_density = build_operators(
        validated["base_config"], validated["base_validated"]
    )
    fixed = config["fixed_look"]
    controls = validated["ao6_config"]["factorization"]
    operator = validated["ao6_validated"]["operator"]

    def render(encoded_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        base = compose_rgb(
            np.asarray(encoded_rgb, dtype=np.float64),
            order=str(fixed["base_order"]),
            density_strength=float(fixed["base_density_strength"]),
            apply_anchor=apply_anchor,
            apply_density=apply_density,
            output_margin=int(
                validated["base_config"]["candidate_bank"][
                    "final_output_margin"
                ]
            ),
        )
        # AO6 was frozen against the retained RGB8 B0 manifest.
        base = np.rint(np.asarray(base, dtype=np.float64) * 255.0) / 255.0
        result = apply_factorized_boundary_guard(
            operator,
            encoded_srgb_to_linear(base),
            tone_strength=float(fixed["tone_strength"]),
            chroma_strength=float(fixed["chroma_strength"]),
            luma_weights=np.asarray(controls["luma_weights"]),
            hard_boundary_epsilon_encoded_srgb=float(
                controls["hard_boundary_epsilon_encoded_srgb"]
            ),
            guard_boundary_epsilon_encoded_srgb=float(
                controls["guard_boundary_epsilon_encoded_srgb"]
            ),
        )
        output = linear_srgb_to_encoded(result.output)
        return np.asarray(output, dtype=np.float64), {
            "tone_limited_fraction": float(
                np.mean(result.tone_scale < 1.0 - 1e-12)
            ),
            "chroma_limited_fraction": float(
                np.mean(result.chroma_scale < 1.0 - 1e-12)
            ),
        }

    return render


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    first_lab = rgb2lab(np.asarray(first).reshape(-1, 1, 3))
    second_lab = rgb2lab(np.asarray(second).reshape(-1, 1, 3))
    return float(
        np.median(
            np.linalg.norm(
                second_lab.reshape(-1, 3) - first_lab.reshape(-1, 3),
                axis=1,
            )
        )
    )


def _new_boundary_fraction(
    source: np.ndarray, output: np.ndarray, epsilon: float
) -> float:
    source_boundary = (source <= epsilon) | (source >= 1.0 - epsilon)
    output_boundary = (output <= epsilon) | (output >= 1.0 - epsilon)
    return float(np.mean(output_boundary & ~source_boundary))


def _bootstrap_lower(
    global_errors: np.ndarray,
    ridge_errors: np.ndarray,
    config: Mapping[str, Any],
) -> float:
    evaluation = config["evaluation"]
    rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    indices = rng.integers(
        0,
        len(global_errors),
        size=(int(evaluation["bootstrap_resamples"]), len(global_errors)),
    )
    values = np.mean(
        global_errors[indices] - ridge_errors[indices], axis=1
    )
    return float(np.quantile(values, 0.025))


def run_ablation(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    neutral_config = validated["neutral_config"]
    report_rows = {
        row["pair_id"]: row for row in validated["neutral_report"]["rows"]
    }
    source_manifest = json.loads(
        (
            root
            / str(neutral_config["source_evidence"]["manifest"])
        ).read_text(encoding="utf-8")
    )
    renderer = build_fixed_ao6_renderer(config, validated)
    target_policy = type(
        "TargetPolicy",
        (),
        {
            "luma_strength": neutral_config["neutral_target"][
                "luma_strength"
            ],
            "chroma_strength": neutral_config["neutral_target"][
                "chroma_strength"
            ],
            "chroma_headroom": neutral_config["neutral_target"][
                "chroma_headroom"
            ],
            "wb_anchor_strength": neutral_config["neutral_target"][
                "white_balance_anchor_strength"
            ],
        },
    )()
    methods = [str(value) for value in config["methods"]]
    errors = {method: [] for method in methods}
    styles = {method: [] for method in methods}
    boundaries = {method: [] for method in methods}
    per_row: list[dict[str, Any]] = []
    epsilon = float(
        validated["ao6_config"]["factorization"][
            "hard_boundary_epsilon_encoded_srgb"
        ]
    )
    for source_row in source_manifest["rows"]:
        pair_id = str(source_row["pair_id"])
        evidence = report_rows.get(pair_id)
        if evidence is None:
            raise FiveKFixedAO6AblationError(
                f"missing neutral-base row: {pair_id}"
            )
        raw, _ = load_raw_default(
            root / source_row["raw_path"],
            int(neutral_config["decode"]["maximum_side"]),
        )
        raw = np.clip(raw, 0.0, 1.0)
        expert = load_expert_icc_srgb(
            root / source_row["expert_path"],
            int(neutral_config["decode"]["maximum_side"]),
        )
        if expert.shape != raw.shape:
            expert = resize_to_shape(expert, raw.shape[:2])
        expert = np.clip(expert, 0.0, 1.0)
        target = filtered_target(raw, expert, target_policy)
        target_look, _ = renderer(target)
        target_style = _median_delta_e76(target, target_look)
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "camera_group_id": source_row["camera_group_id"],
            "target_style_delta_e76": target_style,
        }
        parameter_map = {
            "identity": np.asarray(
                neutral_config["operator"]["identity"], dtype=np.float64
            ),
            "global": np.asarray(
                evidence["global_parameters"], dtype=np.float64
            ),
            "ridge": np.asarray(
                evidence["ridge_parameters"], dtype=np.float64
            ),
            "oracle": np.asarray(
                evidence["fitted_parameters"], dtype=np.float64
            ),
        }
        for method in methods:
            neutral = apply_neutral_base(raw, parameter_map[method])
            output, diagnostics = renderer(neutral)
            error = _rmse(output, target_look)
            style = _median_delta_e76(neutral, output)
            boundary = _new_boundary_fraction(neutral, output, epsilon)
            errors[method].append(error)
            styles[method].append(style)
            boundaries[method].append(boundary)
            record[method] = {
                "rmse_to_fixed_look_target": error,
                "style_delta_e76": style,
                "new_boundary_fraction": boundary,
                **diagnostics,
            }
        per_row.append(record)
    metrics: dict[str, Any] = {}
    for method in methods:
        values = np.asarray(errors[method], dtype=np.float64)
        metrics[method] = {
            "mean_rmse": float(np.mean(values)),
            "median_rmse": float(np.median(values)),
            "p95_rmse": float(np.quantile(values, 0.95)),
            "median_style_delta_e76": float(np.median(styles[method])),
            "maximum_new_boundary_fraction": float(max(boundaries[method])),
        }
    global_errors = np.asarray(errors["global"], dtype=np.float64)
    ridge_errors = np.asarray(errors["ridge"], dtype=np.float64)
    target_style_median = float(
        np.median([row["target_style_delta_e76"] for row in per_row])
    )
    observed = {
        "ridge_mean_improvement_over_global": float(
            (np.mean(global_errors) - np.mean(ridge_errors))
            / max(np.mean(global_errors), 1e-12)
        ),
        "ridge_win_fraction_over_global": float(
            np.mean(ridge_errors < global_errors)
        ),
        "ridge_p95_rmse_ratio_to_global": float(
            np.quantile(ridge_errors, 0.95)
            / max(np.quantile(global_errors, 0.95), 1e-12)
        ),
        "ridge_bootstrap_improvement_lower": _bootstrap_lower(
            global_errors, ridge_errors, config
        ),
        "target_median_style_delta_e76": target_style_median,
        "ridge_to_target_median_style_ratio": float(
            metrics["ridge"]["median_style_delta_e76"]
            / max(target_style_median, 1e-12)
        ),
        "maximum_new_boundary_fraction": float(
            max(
                metrics[method]["maximum_new_boundary_fraction"]
                for method in methods
            )
        ),
    }
    gates_config = config["evaluation"]
    gates = {
        "ridge_mean": observed["ridge_mean_improvement_over_global"]
        >= gates_config["minimum_ridge_mean_improvement_over_global"],
        "ridge_wins": observed["ridge_win_fraction_over_global"]
        >= gates_config["minimum_ridge_win_fraction_over_global"],
        "ridge_tail": observed["ridge_p95_rmse_ratio_to_global"]
        <= gates_config["maximum_ridge_p95_rmse_ratio_to_global"],
        "ridge_bootstrap": observed["ridge_bootstrap_improvement_lower"]
        > gates_config["minimum_ridge_bootstrap_improvement_lower"],
        "target_style": observed["target_median_style_delta_e76"]
        >= gates_config["minimum_target_median_style_delta_e76"],
        "ridge_style": observed["ridge_to_target_median_style_ratio"]
        >= gates_config["minimum_ridge_to_target_median_style_ratio"],
        "new_boundaries": observed["maximum_new_boundary_fraction"]
        <= gates_config["maximum_new_boundary_fraction"],
    }
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "metrics": metrics,
        "observed": observed,
        "gates": gates,
        "rows": per_row,
    }
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": _sha256(config_path),
        "neutral_base_stable_evidence_id": validated["neutral_report"][
            "stable_evidence_id"
        ],
        "row_count": len(per_row),
        "metrics": metrics,
        "observed": observed,
        "gates": gates,
        "automatic_pass": all(gates.values()),
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable_payload)
        ).hexdigest(),
        "rows": per_row,
        "claim_ceiling": config["claim_ceiling"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_bytes(_canonical_bytes(result))
    return {
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "report": result,
    }


__all__ = [
    "FiveKFixedAO6AblationError",
    "build_fixed_ao6_renderer",
    "run_ablation",
    "validate_contract",
]
