"""Grouped AO4P Ektachrome display-proxy operator audit.

The input is one author-rendered stable-pigment palette, so the evaluator
deliberately limits itself to complete-row cross-validation and operator
direction diagnostics.  It does not identify an Ektachrome stock response.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.real_film.velvia_chart_explainability import (
    _fit_affine,
    _fit_record,
    _metrics,
)
from src.real_film.velvia_cross_domain import _fit_options, _to_linear
from src.roll2film.positive_film import positive_film_operator_from_config
from src.roll2film.positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def validate_contract(root: Path, config: dict[str, Any]) -> None:
    """Validate all frozen parent and source identities before fitting."""

    if (
        config.get("schema_version") != 1
        or config.get("experiment_id")
        != "u5.r2ap0-ektachrome-palette-operator-v1"
        or config.get("controls")
        != [
            "identity",
            "per_channel_affine",
            "full_affine",
            "one_matrix",
        ]
        or config["cross_validation"].get("scheme")
        != "leave_one_complete_palette_row_out"
        or config["cross_validation"].get("fold_count") != 6
        or config["fit"].get("model") != "one_matrix"
        or not config.get("operator_fitting_allowed")
        or config.get("training_allowed")
        or config.get("image_rendering_allowed")
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
    ):
        raise ValueError("AP0 frozen contract mismatch")

    expected_parent_decisions = {
        "ao8d_split_instability_decision": "close_case_selector_as_split_unstable",
        "ao9_capacity_decision": (
            "retain_bounded_one_matrix_global_proxy_operator_and_close_"
            "extra_two_matrix_capacity_on_this_pool"
        ),
    }
    for name, expected_decision in expected_parent_decisions.items():
        record = config["parents"][name]
        raw = (root / record["path"]).read_bytes()
        if _sha256(raw) != record["sha256"]:
            raise ValueError(f"AP0 parent hash mismatch: {name}")
        if json.loads(raw).get("decision") != expected_decision:
            raise ValueError(f"AP0 parent decision mismatch: {name}")

    source = config["source"]
    palette_contract_raw = (root / source["palette_contract"]).read_bytes()
    if _sha256(palette_contract_raw) != source["palette_contract_sha256"]:
        raise ValueError("AP0 palette contract hash mismatch")
    palette_contract = json.loads(palette_contract_raw)
    groups = {
        row["group_id"]: row for row in palette_contract.get("groups", [])
    }
    expected_group = groups.get(source["group_id"])
    if (
        palette_contract.get("operator_fitting_allowed")
        or expected_group is None
        or expected_group.get("row_cell_counts") != source["row_cell_counts"]
        or expected_group.get("expected_pair_count")
        != source["expected_pair_count"]
        or expected_group.get("expected_paired_u8_sha256")
        != source["paired_u8_sha256"]
    ):
        raise ValueError("AP0 palette contract lineage mismatch")

    direction = config["velvia_direction_control"]
    operator_raw = (root / direction["operator_config"]).read_bytes()
    if _sha256(operator_raw) != direction["operator_config_sha256"]:
        raise ValueError("AP0 Velvia operator config hash mismatch")
    operator_config = json.loads(operator_raw)
    if direction["operator_witness_id"] not in operator_config.get(
        "witnesses", {}
    ):
        raise ValueError("AP0 Velvia operator witness missing")


def load_ektachrome_pairs(
    path: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Load exact source-reference -> Ektachrome target RGB pairs."""

    source = config["source"]
    raw = path.read_bytes()
    if _sha256(raw) != source["paired_palettes_sha256"]:
        raise ValueError("AP0 paired palette file hash mismatch")
    payload = json.loads(raw)
    if (
        payload.get("asset_sha256") != source["asset_sha256"]
        or payload.get("paired_channel_order")
        != ["film_rgb", "reference_rgb"]
        or payload.get("automatic_pass") is not True
    ):
        raise ValueError("AP0 paired palette lineage mismatch")
    groups = {row["group_id"]: row for row in payload.get("groups", [])}
    group = groups.get(source["group_id"])
    if (
        group is None
        or group.get("pair_count") != source["expected_pair_count"]
        or group.get("paired_u8_sha256") != source["paired_u8_sha256"]
    ):
        raise ValueError("AP0 Ektachrome group mismatch")
    count = int(source["expected_pair_count"])
    film_u8 = np.asarray(group["film_rgb_u8"])
    reference_u8 = np.asarray(group["reference_rgb_u8"])
    if (
        film_u8.shape != (count, 3)
        or reference_u8.shape != (count, 3)
        or film_u8.dtype.kind not in "iu"
        or reference_u8.dtype.kind not in "iu"
        or np.any(film_u8 < 0)
        or np.any(film_u8 > 255)
        or np.any(reference_u8 < 0)
        or np.any(reference_u8 > 255)
    ):
        raise ValueError("AP0 Ektachrome RGB8 rows are invalid")
    paired = np.concatenate(
        (film_u8.astype(np.uint8), reference_u8.astype(np.uint8)), axis=1
    )
    if _sha256(paired.tobytes()) != source["paired_u8_sha256"]:
        raise ValueError("AP0 Ektachrome paired byte identity mismatch")
    return (
        _to_linear(reference_u8, expected_rows=count),
        _to_linear(film_u8, expected_rows=count),
    )


def complete_row_folds(row_counts: list[int], *, total: int) -> list[np.ndarray]:
    if (
        not row_counts
        or any(not isinstance(value, int) or value <= 0 for value in row_counts)
        or sum(row_counts) != total
    ):
        raise ValueError("AP0 row counts do not partition the paired rows")
    folds: list[np.ndarray] = []
    cursor = 0
    for count in row_counts:
        folds.append(np.arange(cursor, cursor + count, dtype=np.int64))
        cursor += count
    return folds


def _predict(
    source: np.ndarray,
    *,
    per_channel: dict[str, Any],
    full_affine: dict[str, Any],
    one_matrix: PositiveFilmFitResult,
) -> dict[str, np.ndarray]:
    return {
        "identity": source.copy(),
        "per_channel_affine": (
            source @ np.asarray(per_channel["matrix"]).T
            + np.asarray(per_channel["bias"])
        ),
        "full_affine": (
            source @ np.asarray(full_affine["matrix"]).T
            + np.asarray(full_affine["bias"])
        ),
        "one_matrix": one_matrix.operator.apply(source),
    }


def operator_direction_metrics(
    candidate: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    candidate_delta = np.asarray(candidate, dtype=np.float64).reshape(-1)
    control_delta = np.asarray(control, dtype=np.float64).reshape(-1)
    candidate_norm = float(np.linalg.norm(candidate_delta))
    control_norm = float(np.linalg.norm(control_delta))
    if (
        candidate_delta.shape != control_delta.shape
        or candidate_norm <= 0.0
        or control_norm <= 0.0
        or not np.all(np.isfinite(candidate_delta))
        or not np.all(np.isfinite(control_delta))
    ):
        raise ValueError("AP0 operator directions must be finite and nonzero")
    dot = float(np.dot(candidate_delta, control_delta))
    cosine = dot / (candidate_norm * control_norm)
    scalar = dot / float(np.dot(control_delta, control_delta))
    residual = candidate_delta - scalar * control_delta
    return {
        "absolute_cosine": float(abs(cosine)),
        "signed_cosine": float(cosine),
        "best_scalar_alignment": float(scalar),
        "residual_fraction_after_best_scalar_alignment": float(
            np.linalg.norm(residual) / candidate_norm
        ),
    }


def evaluate_ektachrome_palette_operator(
    source: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
    *,
    velvia_operator_payload: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one bounded global operator under frozen complete-row CV."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    expected = int(config["source"]["expected_pair_count"])
    if (
        source.shape != (expected, 3)
        or target.shape != source.shape
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(target < 0.0)
        or np.any(target > 1.0)
    ):
        raise ValueError("AP0 requires exact finite paired linear RGB rows")

    fold_indices = complete_row_folds(
        list(config["source"]["row_cell_counts"]), total=expected
    )
    model_names = tuple(config["controls"])
    out_of_fold = {
        name: np.empty_like(target, dtype=np.float64) for name in model_names
    }
    folds: list[dict[str, Any]] = []
    all_converged = True
    wins_identity = 0
    wins_full_affine = 0
    all_indices = np.arange(expected)
    for row_index, confirmation_indices in enumerate(fold_indices):
        development_indices = np.setdiff1d(all_indices, confirmation_indices)
        development_source = source[development_indices]
        development_target = target[development_indices]
        _, per_channel = _fit_affine(
            development_source, development_target, per_channel=True
        )
        _, full_affine = _fit_affine(
            development_source, development_target, per_channel=False
        )
        one_matrix = fit_positive_film_response_operator(
            development_source,
            development_target,
            model="one_matrix",
            **_fit_options(config),
        )
        predictions = _predict(
            source[confirmation_indices],
            per_channel=per_channel,
            full_affine=full_affine,
            one_matrix=one_matrix,
        )
        metrics = {
            name: _metrics(prediction, target[confirmation_indices])
            for name, prediction in predictions.items()
        }
        for name, prediction in predictions.items():
            out_of_fold[name][confirmation_indices] = prediction
        one_rmse = metrics["one_matrix"]["rgb_rmse"]
        identity_win = one_rmse < metrics["identity"]["rgb_rmse"]
        affine_win = one_rmse < metrics["full_affine"]["rgb_rmse"]
        wins_identity += int(identity_win)
        wins_full_affine += int(affine_win)
        all_converged = all_converged and one_matrix.converged
        folds.append(
            {
                "held_palette_row": row_index,
                "development_indices": development_indices.tolist(),
                "confirmation_indices": confirmation_indices.tolist(),
                "models": metrics,
                "one_matrix_fit": _fit_record(one_matrix),
                "one_matrix_wins_identity": identity_win,
                "one_matrix_wins_full_affine": affine_win,
            }
        )

    aggregate = {
        name: _metrics(prediction, target)
        for name, prediction in out_of_fold.items()
    }
    one_rmse = aggregate["one_matrix"]["rgb_rmse"]
    identity_rmse = aggregate["identity"]["rgb_rmse"]
    affine_rmse = aggregate["full_affine"]["rgb_rmse"]
    gain_identity = 1.0 - one_rmse / identity_rmse
    gain_affine = 1.0 - one_rmse / affine_rmse

    global_fit = fit_positive_film_response_operator(
        source, target, model="one_matrix", **_fit_options(config)
    )
    direction_config = config["velvia_direction_control"]
    velvia_operator = positive_film_operator_from_config(
        velvia_operator_payload,
        exposure_floor=float(config["fit"]["exposure_floor"]),
        matrix_minimum_determinant=float(
            config["fit"]["matrix_minimum_determinant"]
        ),
        minimum_endpoint_span=float(config["fit"]["minimum_endpoint_span"]),
    )
    grid_size = int(direction_config["grid_size"])
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    candidate_output = global_fit.operator.apply(grid)
    control_output = velvia_operator.apply(grid)
    direction = operator_direction_metrics(
        candidate_output - grid, control_output - grid
    )
    cube_metrics = _metrics(candidate_output, grid)

    gates = config["gates"]
    checks = [
        {
            "name": "all_rows_tested_exactly_once",
            "passed": sorted(
                index
                for fold in fold_indices
                for index in fold.astype(int).tolist()
            )
            == list(range(expected)),
        },
        {
            "name": "all_structural_fits_converged",
            "passed": all_converged and global_fit.converged,
        },
        {
            "name": "cross_validated_one_matrix_rgb_rmse",
            "passed": one_rmse
            <= float(gates["maximum_cross_validated_one_matrix_rgb_rmse"]),
        },
        {
            "name": "gain_over_identity",
            "passed": gain_identity
            >= float(gates["minimum_gain_over_identity"]),
        },
        {
            "name": "gain_over_full_affine",
            "passed": gain_affine
            >= float(gates["minimum_gain_over_full_affine"]),
        },
        {
            "name": "fold_wins_over_identity",
            "passed": wins_identity
            >= int(gates["minimum_fold_wins_over_identity"]),
        },
        {
            "name": "fold_wins_over_full_affine",
            "passed": wins_full_affine
            >= int(gates["minimum_fold_wins_over_full_affine"]),
        },
        {
            "name": "raw_out_of_cube_fraction",
            "passed": cube_metrics["raw_out_of_cube_fraction"]
            <= float(gates["maximum_raw_out_of_cube_fraction"]),
        },
        {
            "name": "velvia_direction_absolute_cosine",
            "passed": direction["absolute_cosine"]
            <= float(direction_config["maximum_absolute_direction_cosine"]),
        },
        {
            "name": "velvia_direction_residual_fraction",
            "passed": direction[
                "residual_fraction_after_best_scalar_alignment"
            ]
            >= float(
                direction_config[
                    "minimum_residual_fraction_after_best_scalar_alignment"
                ]
            ),
        },
    ]
    automatic_pass = all(bool(check["passed"]) for check in checks)
    result = {
        "folds": folds,
        "cross_validated_metrics": aggregate,
        "one_matrix_gain_over_identity": gain_identity,
        "one_matrix_gain_over_full_affine": gain_affine,
        "fold_wins_over_identity": wins_identity,
        "fold_wins_over_full_affine": wins_full_affine,
        "global_one_matrix_fit": _fit_record(global_fit),
        "grid_audit": {
            "grid_size": grid_size,
            "sample_count": len(grid),
            "candidate_output_metrics_against_identity": cube_metrics,
        },
        "velvia_direction_control": direction,
        "automatic_checks": checks,
        "automatic_pass": automatic_pass,
        "selected_model": "one_matrix" if automatic_pass else "none",
        "decision": (
            config["decision_if_pass"]
            if automatic_pass
            else config["decision_if_fail"]
        ),
    }
    result["stable_evidence_id"] = _sha256(_canonical_json(result))
    return result


__all__ = [
    "complete_row_folds",
    "evaluate_ektachrome_palette_operator",
    "load_ektachrome_pairs",
    "operator_direction_metrics",
    "validate_contract",
]
