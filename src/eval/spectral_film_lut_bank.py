"""U5.R2AE1 evaluator for an isolated external spectral-film operator bank."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.velvia_datasheet_witness import (
    _RGB_TO_XYZ,
    encoded_srgb_to_linear,
    xyz_to_lab,
)
from src.roll2film.baselines import fit_joint_basic_adjustment


MANIFEST_SCHEMA = "u5-r2ae1-external-spectral-bank-manifest-v1"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected AE1 external manifest schema")
    return value


def synthetic_cube(size: int) -> np.ndarray:
    if size < 3:
        raise ValueError("cube size must be at least three")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _lab(encoded: np.ndarray) -> np.ndarray:
    linear = encoded_srgb_to_linear(np.clip(np.asarray(encoded), 0.0, 1.0))
    return xyz_to_lab(linear @ _RGB_TO_XYZ.T)


def median_delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(_lab(left) - _lab(right), axis=-1)))


def _linear_to_encoded(value: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.where(
        array <= 0.0031308,
        12.92 * array,
        1.055 * np.power(array, 1.0 / 2.4) - 0.055,
    )


def matched_basic_output(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_linear = encoded_srgb_to_linear(np.clip(source, 0.0, 1.0))
    target_linear = encoded_srgb_to_linear(np.clip(target, 0.0, 1.0))
    operator = fit_joint_basic_adjustment(
        source_linear.reshape(-1, 3), target_linear.reshape(-1, 3)
    )
    matched = operator.apply(source_linear.reshape(-1, 3)).reshape(source.shape)
    return _linear_to_encoded(matched)


def symmetric_basic_residual(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_to_right = median_delta_e76(matched_basic_output(left, right), right)
    right_to_left = median_delta_e76(matched_basic_output(right, left), left)
    return {
        "left_to_right_delta_e76_median": left_to_right,
        "right_to_left_delta_e76_median": right_to_left,
        "conservative_minimum_delta_e76_median": min(left_to_right, right_to_left),
    }


def local_jacobian_metrics(output: np.ndarray) -> dict[str, float]:
    values = np.asarray(output, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("operator cube must have shape SxSxSx3")
    if not (values.shape[0] == values.shape[1] == values.shape[2]):
        raise ValueError("operator cube axes must have equal length")
    step = 1.0 / (values.shape[0] - 1)
    base = values[:-1, :-1, :-1]
    dr = (values[1:, :-1, :-1] - base) / step
    dg = (values[:-1, 1:, :-1] - base) / step
    db = (values[:-1, :-1, 1:] - base) / step
    jacobian = np.stack((dr, dg, db), axis=-1)
    determinants = np.linalg.det(jacobian)
    spectral_norms = np.linalg.svd(jacobian, compute_uv=False)[..., 0]
    return {
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "negative_jacobian_fraction": float(np.mean(determinants < -1e-5)),
        "maximum_jacobian_spectral_norm": float(np.max(spectral_norms)),
    }


def strength_path_metrics(
    identity: np.ndarray, full: np.ndarray, strength: float
) -> dict[str, float]:
    base = np.asarray(identity, dtype=np.float64)
    effect = np.asarray(full, dtype=np.float64) - base
    target_effect = strength * effect
    denominator = float(np.sum(effect * effect))
    fitted_strength = (
        0.0 if denominator <= 1e-30 else float(np.sum(effect * target_effect) / denominator)
    )
    residual = target_effect - fitted_strength * effect
    target_energy = float(np.sum(target_effect * target_effect))
    residual_energy = float(np.sum(residual * residual))
    explained = (
        1.0
        if target_energy <= 1e-30
        else float(np.clip(1.0 - residual_energy / target_energy, 0.0, 1.0))
    )
    return {
        "fitted_strength": fitted_strength,
        "residual_rgb_rmse": float(np.sqrt(np.mean(residual * residual))),
        "explained_energy_fraction": explained,
    }


def _validate_manifest_contract(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if manifest["external_revision"] != config["external_source"]["revision"]:
        raise ValueError("external revision mismatch")
    if manifest["python_version"] != config["runtime"]["python_version"]:
        raise ValueError("Python version mismatch")
    if manifest["package_versions"] != config["runtime"]["package_versions"]:
        raise ValueError("package version mismatch")
    if manifest["config_sha256"] != canonical_sha256(config):
        raise ValueError("config hash mismatch")
    expected = [row["id"] for row in config["chains"]]
    observed = [row["chain_id"] for row in manifest["records"]]
    if observed != expected:
        raise ValueError("chain population or order mismatch")


def _load_run_arrays(
    manifest: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    neutrals: dict[str, np.ndarray] = {}
    for row in manifest["records"]:
        output_path = root / row["output_path"]
        neutral_path = root / row["neutral_path"]
        if sha256_file(output_path) != row["output_file_sha256"]:
            raise ValueError(f"output file hash mismatch: {row['chain_id']}")
        if sha256_file(neutral_path) != row["neutral_file_sha256"]:
            raise ValueError(f"neutral file hash mismatch: {row['chain_id']}")
        output = np.load(output_path, allow_pickle=False)
        neutral = np.load(neutral_path, allow_pickle=False)
        if array_sha256(output) != row["output_array_sha256"]:
            raise ValueError(f"output array hash mismatch: {row['chain_id']}")
        if array_sha256(neutral) != row["neutral_array_sha256"]:
            raise ValueError(f"neutral array hash mismatch: {row['chain_id']}")
        outputs[row["chain_id"]] = output
        neutrals[row["chain_id"]] = neutral
    duplicate_path = root / manifest["duplicate_control"]["output_path"]
    if sha256_file(duplicate_path) != manifest["duplicate_control"]["output_file_sha256"]:
        raise ValueError("duplicate-control file hash mismatch")
    duplicate = np.load(duplicate_path, allow_pickle=False)
    if array_sha256(duplicate) != manifest["duplicate_control"]["output_array_sha256"]:
        raise ValueError("duplicate-control array hash mismatch")
    return outputs, neutrals, duplicate


def evaluate_manifests(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    _validate_manifest_contract(first, config)
    _validate_manifest_contract(second, config)
    outputs_a, neutrals_a, duplicate_a = _load_run_arrays(first, root=root)
    outputs_b, neutrals_b, duplicate_b = _load_run_arrays(second, root=root)

    chain_ids = [row["id"] for row in config["chains"]]
    exact_replay = all(
        np.array_equal(outputs_a[name], outputs_b[name])
        and np.array_equal(neutrals_a[name], neutrals_b[name])
        for name in chain_ids
    ) and np.array_equal(duplicate_a, duplicate_b)

    size = int(config["synthetic_population"]["cube_size"])
    identity = synthetic_cube(size)
    margin = float(config["synthetic_population"]["interior_clip_margin"])
    interior = np.all((identity > margin) & (identity < 1.0 - margin), axis=-1)
    chain_config = {row["id"]: row for row in config["chains"]}
    gates = config["automatic_gates"]
    records = []
    for chain_id in chain_ids:
        output = np.asarray(outputs_a[chain_id], dtype=np.float64)
        neutral = np.asarray(neutrals_a[chain_id], dtype=np.float64)
        expected_shape = (size, size, size, 3)
        if output.shape != expected_shape:
            raise ValueError(f"unexpected output shape: {chain_id}")
        expected_neutral = int(config["synthetic_population"]["neutral_samples"])
        if neutral.shape != (expected_neutral, 3):
            raise ValueError(f"unexpected neutral shape: {chain_id}")
        jacobian = local_jacobian_metrics(output)
        jacobian["negative_jacobian_fraction"] = float(
            np.mean(
                _jacobian_determinants(output)
                < float(gates["negative_jacobian_threshold"])
            )
        )
        neutral_luma = encoded_srgb_to_linear(np.clip(neutral, 0.0, 1.0)) @ LUMA
        neutral_violation = float(max(0.0, -np.min(np.diff(neutral_luma))))
        interior_output = output[interior]
        hard_clip = (interior_output <= 1e-6) | (interior_output >= 1.0 - 1e-6)
        basic = matched_basic_output(identity, output)
        metrics = {
            "output_array_sha256": array_sha256(outputs_a[chain_id]),
            "output_minimum": float(np.min(output)),
            "output_maximum": float(np.max(output)),
            "finite": bool(np.all(np.isfinite(output))),
            "interior_hard_clip_fraction": float(np.mean(hard_clip)),
            "neutral_luma_monotonic_violation": neutral_violation,
            "style_delta_e76_median_vs_identity": median_delta_e76(identity, output),
            "joint_basic_residual_delta_e76_median_vs_identity": median_delta_e76(
                basic, output
            ),
            **jacobian,
        }
        checks = {
            "finite": metrics["finite"] is gates["all_outputs_finite"],
            "range": metrics["output_minimum"] >= gates["output_minimum"]
            and metrics["output_maximum"] <= gates["output_maximum"],
            "interior_clip": metrics["interior_hard_clip_fraction"]
            <= gates["maximum_interior_hard_clip_fraction"],
            "neutral_monotonicity": metrics["neutral_luma_monotonic_violation"]
            <= gates["maximum_neutral_luma_monotonic_violation"],
            "jacobian_orientation": metrics["negative_jacobian_fraction"]
            <= gates["maximum_negative_jacobian_fraction"],
            "jacobian_amplification": metrics["maximum_jacobian_spectral_norm"]
            <= gates["maximum_jacobian_spectral_norm"],
            "style": metrics["style_delta_e76_median_vs_identity"]
            >= gates["minimum_style_delta_e76_median_vs_identity"],
            "non_basic": metrics["joint_basic_residual_delta_e76_median_vs_identity"]
            >= gates["minimum_joint_basic_residual_delta_e76_median_vs_identity"],
        }
        records.append(
            {
                "chain_id": chain_id,
                "family": chain_config[chain_id]["family"],
                "metrics": metrics,
                "checks": checks,
            }
        )

    pairwise = []
    family_distinct: dict[str, bool] = {}
    for family in sorted({row["family"] for row in config["chains"]}):
        names = [row["id"] for row in config["chains"] if row["family"] == family]
        family_rows = []
        for left, right in combinations(names, 2):
            residual = symmetric_basic_residual(outputs_a[left], outputs_a[right])
            distinct = (
                residual["conservative_minimum_delta_e76_median"]
                >= gates["minimum_within_family_pair_residual_delta_e76_median"]
            )
            row = {
                "family": family,
                "left": left,
                "right": right,
                "distinct_after_basic": distinct,
                **residual,
            }
            pairwise.append(row)
            family_rows.append(row)
        family_distinct[family] = any(
            row["distinct_after_basic"] for row in family_rows
        )

    duplicate_id = config["controls"]["exact_duplicate_chain_id"]
    duplicate_exact = np.array_equal(outputs_a[duplicate_id], duplicate_a)
    strength = strength_path_metrics(
        identity,
        outputs_a[config["controls"]["strength_path_chain_id"]],
        float(config["controls"]["strength_path_value"]),
    )
    strength_pass = (
        strength["residual_rgb_rmse"]
        <= config["controls"]["strength_fit_residual_rgb_rmse_max"]
        and strength["explained_energy_fraction"]
        >= config["controls"]["strength_fit_explained_energy_min"]
    )
    safety_keys = (
        "finite",
        "range",
        "interior_clip",
        "neutral_monotonicity",
        "jacobian_orientation",
        "jacobian_amplification",
    )
    safety_pass = all(
        all(row["checks"][key] for key in safety_keys) for row in records
    )
    styled = sum(row["checks"]["style"] for row in records)
    non_basic = sum(row["checks"]["non_basic"] for row in records)
    distinct_families = sum(family_distinct.values())
    checks = {
        "chain_count": len(records) == gates["exact_chain_count"],
        "exact_complete_run_replay": exact_replay
        is gates["exact_complete_run_replay_required"],
        "all_structurally_safe": safety_pass,
        "minimum_stylized_chains": styled >= gates["minimum_stylized_chains"],
        "minimum_non_basic_chains": non_basic >= gates["minimum_non_basic_chains"],
        "minimum_distinct_families": distinct_families
        >= gates["minimum_distinct_families"],
        "duplicate_control": duplicate_exact
        is gates["duplicate_control_requires_exact_equality"],
        "strength_path_control": strength_pass
        is gates["strength_path_control_required"],
    }
    if not checks["chain_count"]:
        decision = "close_population_mismatch"
    elif not checks["exact_complete_run_replay"]:
        decision = "close_runtime_or_replay_failure"
    elif not checks["duplicate_control"] or not checks["strength_path_control"]:
        decision = "close_negative_control_failure"
    elif not checks["all_structurally_safe"]:
        decision = "close_range_fold_or_monotonicity_failure"
    elif not (
        checks["minimum_stylized_chains"]
        and checks["minimum_non_basic_chains"]
        and checks["minimum_distinct_families"]
    ):
        decision = "close_basic_only_or_insufficient_family_diversity"
    else:
        decision = "retain_external_structural_comparison_bank_only"
    return {
        "schema_version": "u5-r2ae1-spectral-film-lut-structural-bank-report-v1",
        "experiment_id": config["experiment_id"],
        "decision": decision,
        "summary": {
            "chain_count": len(records),
            "exact_complete_run_replay": exact_replay,
            "structurally_safe_chains": sum(
                all(row["checks"][key] for key in safety_keys) for row in records
            ),
            "stylized_chains": styled,
            "non_basic_chains": non_basic,
            "distinct_family_count": distinct_families,
            "family_distinct_after_basic": family_distinct,
            "duplicate_control_exact": duplicate_exact,
            "strength_path": strength,
        },
        "gate_checks": checks,
        "records": records,
        "within_family_pairwise": pairwise,
        "visual_review_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def _jacobian_determinants(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float64)
    step = 1.0 / (values.shape[0] - 1)
    base = values[:-1, :-1, :-1]
    jacobian = np.stack(
        (
            (values[1:, :-1, :-1] - base) / step,
            (values[:-1, 1:, :-1] - base) / step,
            (values[:-1, :-1, 1:] - base) / step,
        ),
        axis=-1,
    )
    return np.linalg.det(jacobian)
