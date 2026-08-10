"""Synthetic capacity and identifiability test for chromaticity-conditioned LUTs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class ChromaticityLUTError(ValueError):
    """Raised when the frozen experiment or explicit operator is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "neuro_film.u5_r2bx0_c2lut_chromaticity_identifiability_contract.v1",
        "experiment_id": "U5.R2BX0",
        "status": "contract_frozen_implementation_ready",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ChromaticityLUTError("contract identity drift")
    fixture = payload.get("fixture", {})
    gates = payload.get("gates", {})
    if (
        fixture.get("lut_grid_size") != 9
        or fixture.get("evaluation_grid_size") != 17
        or fixture.get("low_rank_basis_rank") != 4
        or fixture.get("anchor_r") != [0.24, 0.33, 0.42]
        or fixture.get("anchor_g") != [0.24, 0.33, 0.42]
        or gates.get("minimum_injective_median_improvement_over_global") != 0.75
        or gates.get("minimum_metamer_truth_pair_difference") != 0.02
        or payload.get("clean_room_scope", {}).get("learned_final_rgb_allowed")
        or payload.get("clean_room_scope", {}).get("paper_code_or_weights_used")
        or payload.get("clean_room_scope", {}).get("paper_data_used")
    ):
        raise ChromaticityLUTError("contract boundary drift")
    return payload


def _grid(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _truth_lut(
    grid_size: int,
    r: float,
    g: float,
    hidden: float,
    *,
    visible_scale: float,
    hidden_scale: float,
) -> np.ndarray:
    rgb = _grid(grid_size)
    u = (float(r) - 0.33) / 0.09
    v = (float(g) - 0.33) / 0.09
    centered = rgb - 0.5
    envelope = 4.0 * rgb * (1.0 - rgb)
    visible = np.empty_like(rgb)
    visible[..., 0] = (
        0.72 * u * centered[..., 1]
        - 0.48 * v * centered[..., 2]
        + 0.36 * u * v * centered[..., 1] * centered[..., 2]
    )
    visible[..., 1] = (
        -0.55 * u * centered[..., 0]
        + 0.63 * v * centered[..., 2]
        - 0.32 * u * v * centered[..., 0] * centered[..., 2]
    )
    visible[..., 2] = (
        0.46 * u * centered[..., 0]
        - 0.68 * v * centered[..., 1]
        + 0.40 * u * v * centered[..., 0] * centered[..., 1]
    )
    spectral = np.empty_like(rgb)
    spectral[..., 0] = 0.85 * centered[..., 1] - 0.55 * centered[..., 2]
    spectral[..., 1] = 0.75 * centered[..., 2] - 0.45 * centered[..., 0]
    spectral[..., 2] = 0.65 * centered[..., 0] - 0.70 * centered[..., 1]
    output = rgb + envelope * (
        visible_scale * visible + hidden_scale * float(hidden) * spectral
    )
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise ChromaticityLUTError("synthetic truth escaped the bounded cube")
    return output


def _trilinear_apply(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    table = np.asarray(lut, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or table.ndim != 4
        or table.shape[3] != 3
        or table.shape[:3] != (table.shape[0],) * 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ChromaticityLUTError("invalid trilinear input")
    size = table.shape[0]
    scaled = values * (size - 1)
    lower = np.minimum(np.floor(scaled).astype(np.int64), size - 2)
    fraction = scaled - lower
    output = np.zeros_like(values)
    for dr in (0, 1):
        wr = fraction[:, 0] if dr else 1.0 - fraction[:, 0]
        for dg in (0, 1):
            wg = fraction[:, 1] if dg else 1.0 - fraction[:, 1]
            for db in (0, 1):
                wb = fraction[:, 2] if db else 1.0 - fraction[:, 2]
                output += (wr * wg * wb)[:, None] * table[
                    lower[:, 0] + dr,
                    lower[:, 1] + dg,
                    lower[:, 2] + db,
                ]
    return output


def _axis_weights(value: float, anchors: np.ndarray) -> np.ndarray:
    if value < anchors[0] or value > anchors[-1]:
        raise ChromaticityLUTError("descriptor outside convex anchor hull")
    upper = int(np.searchsorted(anchors, value, side="right"))
    upper = min(max(upper, 1), len(anchors) - 1)
    lower = upper - 1
    fraction = (value - anchors[lower]) / (anchors[upper] - anchors[lower])
    weights = np.zeros(len(anchors), dtype=np.float64)
    weights[lower] = 1.0 - fraction
    weights[upper] = fraction
    return weights


def _convex_weights(r: float, g: float, anchors_r: np.ndarray, anchors_g: np.ndarray) -> np.ndarray:
    weights = np.outer(_axis_weights(r, anchors_r), _axis_weights(g, anchors_g))
    if np.any(weights < 0.0) or abs(float(np.sum(weights)) - 1.0) > 1e-12:
        raise ChromaticityLUTError("invalid convex weights")
    return weights.reshape(-1)


def _fit_affine(rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.concatenate([rgb.reshape(-1, 3), np.ones((rgb.size // 3, 1))], axis=1)
    y = target.reshape(-1, 3)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _apply_affine(rgb: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    x = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    return x @ coefficients


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def _improvement(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ChromaticityLUTError("invalid zero-error baseline")
    return 1.0 - candidate / baseline


def _boundary_fraction(values: np.ndarray, epsilon: float) -> float:
    outside = np.any((values < -epsilon) | (values > 1.0 + epsilon), axis=1)
    return float(np.mean(outside))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def evaluate_contract(contract_path: Path) -> dict[str, Any]:
    config = load_contract(contract_path)
    fixture = config["fixture"]
    gates = config["gates"]
    anchors_r = np.asarray(fixture["anchor_r"], dtype=np.float64)
    anchors_g = np.asarray(fixture["anchor_g"], dtype=np.float64)
    lut_size = int(fixture["lut_grid_size"])
    visible_scale = float(fixture["visible_residual_scale"])
    hidden_scale = float(fixture["hidden_spectral_residual_scale"])

    anchor_luts = np.stack(
        [
            _truth_lut(
                lut_size,
                r,
                g,
                0.0,
                visible_scale=visible_scale,
                hidden_scale=hidden_scale,
            )
            for r in anchors_r
            for g in anchors_g
        ]
    )
    identity_lut = _grid(lut_size)
    residuals = anchor_luts - identity_lut
    flat = residuals.reshape(len(residuals), -1)
    u, singular, vt = np.linalg.svd(flat, full_matrices=False)
    rank = int(fixture["low_rank_basis_rank"])
    rank_flat = (u[:, :rank] * singular[:rank]) @ vt[:rank]
    rank_luts = identity_lut + rank_flat.reshape(residuals.shape)
    global_lut = np.mean(anchor_luts, axis=0)
    anchor_rgb = _grid(lut_size)
    affine_bank = np.stack([_fit_affine(anchor_rgb, lut) for lut in anchor_luts])
    evaluation_rgb = _grid(int(fixture["evaluation_grid_size"])).reshape(-1, 3)

    rows: list[dict[str, Any]] = []
    for r in fixture["held_r"]:
        for g in fixture["held_g"]:
            weights = _convex_weights(float(r), float(g), anchors_r, anchors_g)
            truth_lut = _truth_lut(
                lut_size,
                float(r),
                float(g),
                0.0,
                visible_scale=visible_scale,
                hidden_scale=hidden_scale,
            )
            target = _trilinear_apply(evaluation_rgb, truth_lut)
            identity = evaluation_rgb
            global_output = _trilinear_apply(evaluation_rgb, global_lut)
            affine_output = _apply_affine(
                evaluation_rgb, np.tensordot(weights, affine_bank, axes=1)
            )
            convex_output = _trilinear_apply(
                evaluation_rgb, np.tensordot(weights, anchor_luts, axes=1)
            )
            rank_output = _trilinear_apply(
                evaluation_rgb, np.tensordot(weights, rank_luts, axes=1)
            )
            errors = {
                "identity": _rmse(identity, target),
                "global_explicit_lut": _rmse(global_output, target),
                "chromaticity_conditioned_affine": _rmse(affine_output, target),
                "chromaticity_conditioned_convex_lut": _rmse(convex_output, target),
                "chromaticity_conditioned_rank4_lut": _rmse(rank_output, target),
            }
            rows.append(
                {
                    "r": float(r),
                    "g": float(g),
                    "weight_sum": float(np.sum(weights)),
                    "minimum_weight": float(np.min(weights)),
                    "errors": errors,
                    "rank4_improvement_over_global": _improvement(
                        errors["chromaticity_conditioned_rank4_lut"],
                        errors["global_explicit_lut"],
                    ),
                    "rank4_improvement_over_affine": _improvement(
                        errors["chromaticity_conditioned_rank4_lut"],
                        errors["chromaticity_conditioned_affine"],
                    ),
                    "rank4_new_boundary_fraction": _boundary_fraction(
                        rank_output, float(fixture["boundary_epsilon"])
                    ),
                }
            )

    metamer_rows: list[dict[str, Any]] = []
    hidden_values = [float(value) for value in fixture["metamer_hidden_coordinates"]]
    for descriptor in fixture["metamer_descriptors"]:
        r, g = (float(value) for value in descriptor)
        weights = _convex_weights(r, g, anchors_r, anchors_g)
        predicted_lut = np.tensordot(weights, rank_luts, axes=1)
        predicted = _trilinear_apply(evaluation_rgb, predicted_lut)
        targets = []
        oracles = []
        predicted_errors = []
        oracle_errors = []
        for hidden in hidden_values:
            truth_lut = _truth_lut(
                lut_size,
                r,
                g,
                hidden,
                visible_scale=visible_scale,
                hidden_scale=hidden_scale,
            )
            target = _trilinear_apply(evaluation_rgb, truth_lut)
            oracle = _trilinear_apply(evaluation_rgb, truth_lut)
            targets.append(target)
            oracles.append(oracle)
            predicted_errors.append(_rmse(predicted, target))
            oracle_errors.append(_rmse(oracle, target))
        metamer_rows.append(
            {
                "r": r,
                "g": g,
                "hidden_coordinates": hidden_values,
                "truth_pair_rmse": _rmse(targets[0], targets[1]),
                "predicted_pair_rmse": _rmse(predicted, predicted.copy()),
                "rank4_rmse": _summary(predicted_errors),
                "oracle_rmse": _summary(oracle_errors),
            }
        )

    global_improvements = [row["rank4_improvement_over_global"] for row in rows]
    affine_improvements = [row["rank4_improvement_over_affine"] for row in rows]
    capacity_gates = {
        "convex_weights": all(
            abs(row["weight_sum"] - 1.0) <= gates["maximum_convex_weight_error"]
            and row["minimum_weight"] >= 0.0
            for row in rows
        ),
        "rank4_error": max(
            row["errors"]["chromaticity_conditioned_rank4_lut"] for row in rows
        )
        <= gates["maximum_injective_rank4_rmse"],
        "median_over_global": float(np.median(global_improvements))
        >= gates["minimum_injective_median_improvement_over_global"],
        "worst_over_global": min(global_improvements)
        >= gates["minimum_injective_worst_improvement_over_global"],
        "median_over_affine": float(np.median(affine_improvements))
        >= gates["minimum_injective_median_improvement_over_affine"],
        "boundary": max(row["rank4_new_boundary_fraction"] for row in rows)
        <= gates["maximum_new_boundary_fraction"],
    }
    identifiability_gates = {
        "same_descriptor_same_prediction": max(
            row["predicted_pair_rmse"] for row in metamer_rows
        )
        <= gates["maximum_metamer_predicted_pair_difference"],
        "different_spectra_different_truth": min(
            row["truth_pair_rmse"] for row in metamer_rows
        )
        >= gates["minimum_metamer_truth_pair_difference"],
        "descriptor_only_error_small": max(
            row["rank4_rmse"]["maximum"] for row in metamer_rows
        )
        < gates["minimum_metamer_rank4_rmse"],
        "hidden_oracle_exact": max(
            row["oracle_rmse"]["maximum"] for row in metamer_rows
        )
        <= gates["maximum_metamer_oracle_rmse"],
    }
    capacity_pass = all(capacity_gates.values())
    descriptor_sufficiency_pass = all(identifiability_gates.values())
    if capacity_pass and not identifiability_gates["different_spectra_different_truth"]:
        decision = "retain_capacity_close_underpowered_metamer_fixture"
    elif capacity_pass and not descriptor_sufficiency_pass:
        decision = "retain_synthetic_capacity_close_chromaticity_sufficiency"
    else:
        decision = "unexpected_contract_outcome"
    report: dict[str, Any] = {
        "schema": "neuro_film.u5_r2bx0_c2lut_chromaticity_identifiability_result.v1",
        "experiment_id": config["experiment_id"],
        "contract": str(contract_path.as_posix()),
        "contract_sha256": _sha256(contract_path),
        "primary_source": config["primary_source"],
        "fixture": fixture,
        "rank_singular_values": [float(value) for value in singular],
        "injective_rows": rows,
        "injective_summary": {
            "rank4_improvement_over_global": _summary(global_improvements),
            "rank4_improvement_over_affine": _summary(affine_improvements),
            "maximum_rank4_rmse": max(
                row["errors"]["chromaticity_conditioned_rank4_lut"] for row in rows
            ),
            "maximum_new_boundary_fraction": max(
                row["rank4_new_boundary_fraction"] for row in rows
            ),
        },
        "metamer_rows": metamer_rows,
        "capacity_gates": capacity_gates,
        "identifiability_gates": identifiability_gates,
        "capacity_pass": capacity_pass,
        "descriptor_sufficiency_pass": descriptor_sufficiency_pass,
        "decision": decision,
        "claim_ceiling": config["claim_ceiling"],
    }
    identity_payload = dict(report)
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
