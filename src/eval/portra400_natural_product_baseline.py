"""Evaluate the fixed Portra product look on registered natural observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pipeline_color_baseline import load_guardrail_config
from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.eval.portra400_chart_operator_d1 import _affine_apply, _load_samples, _rmse
from src.eval.portra400_local_correspondence_transfer_d1 import _mutual_inliers
from src.eval.portra400_same_scene_registration import _raster_rgb, _raw_rgb
from src.inference import load_render_profile, render_three_stock_look_rgb
from src.real_film.velvia_chart_explainability import _fit_affine

SCHEMA = "neuro-film.rf3-d9-portra400-natural-product-baseline-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-d9-portra400-natural-product-baseline-result.v1"


class Portra400NaturalProductBaselineError(ValueError):
    """Raised when a frozen RF3.D9 input or execution contract drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _linear_rows(encoded: list[np.ndarray]) -> np.ndarray:
    values = np.asarray(encoded, dtype=np.float64)
    return np.asarray(encoded_srgb_to_linear(values[:, None, :]), dtype=np.float64)[
        :, 0, :
    ]


def _registered_patch_rows(
    source: np.ndarray,
    target: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    radius = int(spec["patch_radius_pixels"])
    low, high = float(spec["minimum_code"]), float(spec["maximum_code"])
    maximum_mad = float(spec["maximum_patch_channel_mad"])
    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    candidate_rows: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    folds: list[int] = []
    for source_point, target_point in zip(source_xy, target_xy, strict=True):
        sx, sy = (round(float(value)) for value in source_point)
        tx, ty = (round(float(value)) for value in target_point)
        if (
            sx - radius < 0
            or sy - radius < 0
            or tx - radius < 0
            or ty - radius < 0
            or sx + radius >= source.shape[1]
            or sy + radius >= source.shape[0]
            or tx + radius >= target.shape[1]
            or ty + radius >= target.shape[0]
        ):
            continue
        source_patch = (
            source[sy - radius : sy + radius + 1, sx - radius : sx + radius + 1]
            .astype(np.float64)
            / 255.0
        )
        target_patch = (
            target[ty - radius : ty + radius + 1, tx - radius : tx + radius + 1]
            .astype(np.float64)
            / 255.0
        )
        source_median = np.median(source_patch.reshape(-1, 3), axis=0)
        target_median = np.median(target_patch.reshape(-1, 3), axis=0)
        source_mad = np.median(
            np.abs(source_patch.reshape(-1, 3) - source_median), axis=0
        )
        target_mad = np.median(
            np.abs(target_patch.reshape(-1, 3) - target_median), axis=0
        )
        if (
            np.any(source_median < low)
            or np.any(source_median > high)
            or np.any(target_median < low)
            or np.any(target_median > high)
            or np.any(source_mad > maximum_mad)
            or np.any(target_mad > maximum_mad)
        ):
            continue
        source_rows.append(source_median)
        target_rows.append(target_median)
        for name, candidate in candidates.items():
            patch = candidate[
                sy - radius : sy + radius + 1,
                sx - radius : sx + radius + 1,
            ]
            candidate_rows[name].append(np.median(patch.reshape(-1, 3), axis=0))
        folds.append(
            2 * int(ty >= target.shape[0] / 2) + int(tx >= target.shape[1] / 2)
        )
    if not source_rows:
        raise Portra400NaturalProductBaselineError("no eligible natural patches")
    return (
        _linear_rows(source_rows),
        _linear_rows(target_rows),
        {name: _linear_rows(rows) for name, rows in candidate_rows.items()},
        np.asarray(folds, dtype=np.int64),
    )


def _fold_improvements(
    candidate: np.ndarray,
    identity: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    fold_count: int,
) -> list[float]:
    values: list[float] = []
    for fold in range(fold_count):
        held = folds == fold
        if np.any(held):
            values.append(
                1.0 - _rmse(candidate[held], target[held]) / _rmse(identity[held], target[held])
            )
    return values


def evaluate(*, root: Path, contract_path: Path, reverse: bool = False) -> dict[str, Any]:
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    if contract.get("schema") != SCHEMA or contract.get("experiment_id") != "RF3.D9":
        raise Portra400NaturalProductBaselineError("unsupported RF3.D9 contract")
    for binding in contract["sources"].values():
        path = root / binding["path"]
        if _sha256_file(path) != binding["sha256"]:
            raise Portra400NaturalProductBaselineError(f"source identity drift: {path}")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise Portra400NaturalProductBaselineError("parent decision drift")

    cham10 = json.loads(
        (root / contract["sources"]["cham10_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    cham6 = json.loads(
        (root / cham10["parents"]["cham6_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    chart_source, chart_target, _blocks, _dataset = _load_samples(cham6, root)
    _, chart_affine = _fit_affine(chart_source, chart_target, per_channel=False)

    data_root = root / "data/quarantine/spektrafilm_portra400_same_scene_v1"
    source_u8, source_metadata = _raw_rgb(
        data_root / "RAW.RW2",
        {
            "raw_use_camera_wb": True,
            "raw_no_auto_bright": False,
            "raw_half_size": True,
        },
    )
    encoded_source = np.ascontiguousarray(source_u8.astype(np.float32) / 255.0)
    profile = load_render_profile(
        root / contract["sources"]["render_profile"]["path"], root=root
    )
    style_statistics = json.loads(
        (root / contract["sources"]["style_statistics"]["path"]).read_text(
            encoding="utf-8"
        )
    )["styles"]
    arm_specs = [
        ("current_portra_400_look_approximation", "kodak_portra_400", "portra_400"),
        ("current_ektar_100_wrong_stock_control", "kodak_ektar_100", "ektar_100"),
    ]
    if reverse:
        arm_specs.reverse()
    candidates: dict[str, np.ndarray] = {}
    output_facts: dict[str, dict[str, Any]] = {}
    for arm_id, stock_id, style in arm_specs:
        kwargs = {
            "profile": profile,
            "film_stock_id": stock_id,
            "look_amount": float(contract["execution"]["look_amount"]),
            "style_statistics": style_statistics[style],
            "guardrails": load_guardrail_config(
                root / contract["sources"]["guardrails"]["path"], style
            ),
            "seed": int(contract["execution"]["seed"]),
            "tile_size": int(contract["execution"]["tile_size"]),
        }
        first = render_three_stock_look_rgb(encoded_source, **kwargs)
        second = render_three_stock_look_rgb(encoded_source, **kwargs)
        candidates[arm_id] = first
        output_facts[arm_id] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "maximum_repeat_error": float(np.max(np.abs(first - second))),
            "boundary_fraction": float(np.mean((first <= 0.0) | (first >= 1.0))),
        }

    role_paths = {
        "same_scene_portra400_fuji_dpii_ra4_epson_scan": data_root / "RA4 Print.jpg",
        "same_scene_portra400_noritsu_scan": data_root / "Film Scan.jpg",
    }
    pair_specs = list(cham10["natural_pairs"])
    if reverse:
        pair_specs.reverse()
    rows: list[dict[str, Any]] = []
    fold_count = int(cham10["correspondence"]["spatial_fold_count"])
    for pair in pair_specs:
        target, target_metadata = _raster_rgb(role_paths[pair["target_role"]])
        source_xy, target_xy, registration = _mutual_inliers(
            source_u8, target, cham10["correspondence"]
        )
        identity, natural_target, sampled, folds = _registered_patch_rows(
            source_u8,
            target,
            candidates,
            source_xy,
            target_xy,
            cham10["correspondence"],
        )
        sampled["identity"] = identity
        sampled["frozen_cham10_chart_affine"] = _affine_apply(chart_affine, identity)
        rmse = {name: _rmse(values, natural_target) for name, values in sampled.items()}
        wrong_target = np.roll(
            natural_target,
            int(cham10["correspondence"]["wrong_target_roll"]),
            axis=0,
        )
        portra = sampled["current_portra_400_look_approximation"]
        fold_improvements = _fold_improvements(
            portra, identity, natural_target, folds, fold_count
        )
        rows.append(
            {
                "id": pair["id"],
                "primary": bool(pair["primary"]),
                "source_decoded_sha256": source_metadata["decoded_sha256"],
                "target_decoded_sha256": target_metadata["decoded_sha256"],
                "registration": registration,
                "eligible_correspondences": len(natural_target),
                "occupied_spatial_folds": len(fold_improvements),
                "arm_rmse": dict(sorted(rmse.items())),
                "metrics": {
                    "portra_improvement_over_identity_fraction": 1.0
                    - rmse["current_portra_400_look_approximation"] / rmse["identity"],
                    "portra_improvement_over_wrong_stock_fraction": 1.0
                    - rmse["current_portra_400_look_approximation"]
                    / rmse["current_ektar_100_wrong_stock_control"],
                    "portra_improvement_over_wrong_pairing_fraction": 1.0
                    - _rmse(portra, natural_target) / _rmse(portra, wrong_target),
                    "portra_improvement_over_frozen_cham10_fraction": 1.0
                    - rmse["current_portra_400_look_approximation"]
                    / rmse["frozen_cham10_chart_affine"],
                    "portra_fold_win_fraction_over_identity": float(
                        np.mean(np.asarray(fold_improvements) > 0.0)
                    ),
                    "portra_worst_fold_improvement_over_identity_fraction": min(
                        fold_improvements
                    ),
                },
            }
        )
    rows.sort(key=lambda row: str(row["id"]))
    primary = next(row for row in rows if row["primary"])
    gates_cfg = contract["gates"]
    identity_wins = sum(
        row["metrics"]["portra_improvement_over_identity_fraction"]
        >= gates_cfg["minimum_portra_improvement_over_identity_fraction"]
        for row in rows
    )
    stock_wins = sum(
        row["metrics"]["portra_improvement_over_wrong_stock_fraction"]
        >= gates_cfg["minimum_portra_improvement_over_wrong_stock_fraction"]
        for row in rows
    )
    pairing_wins = sum(
        row["metrics"]["portra_improvement_over_wrong_pairing_fraction"]
        >= gates_cfg["minimum_portra_improvement_over_wrong_pairing_fraction"]
        for row in rows
    )
    gates = {
        "minimum_support": all(
            row["eligible_correspondences"]
            >= gates_cfg["minimum_eligible_correspondences_per_interpretation"]
            for row in rows
        ),
        "minimum_interpretations_improving_identity": identity_wins
        >= gates_cfg["minimum_interpretations_improving_identity"],
        "minimum_interpretations_beating_wrong_stock": stock_wins
        >= gates_cfg["minimum_interpretations_beating_wrong_stock"],
        "minimum_interpretations_beating_wrong_pairing": pairing_wins
        >= gates_cfg["minimum_interpretations_beating_wrong_pairing"],
        "minimum_primary_fold_win_fraction_over_identity": primary["metrics"][
            "portra_fold_win_fraction_over_identity"
        ]
        >= gates_cfg["minimum_primary_fold_win_fraction_over_identity"],
        "minimum_primary_worst_fold_improvement_over_identity": primary["metrics"][
            "portra_worst_fold_improvement_over_identity_fraction"
        ]
        >= gates_cfg["minimum_primary_worst_fold_improvement_over_identity_fraction"],
        "portra_beats_frozen_cham10_on_both_interpretations": all(
            row["metrics"]["portra_improvement_over_frozen_cham10_fraction"] > 0.0
            for row in rows
        ),
        "maximum_current_output_boundary_fraction": max(
            fact["boundary_fraction"] for fact in output_facts.values()
        )
        <= gates_cfg["maximum_current_output_boundary_fraction"],
        "maximum_repeat_error": max(
            fact["maximum_repeat_error"] for fact in output_facts.values()
        )
        <= gates_cfg["maximum_repeat_error"],
    }
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "RF3.D9",
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "arm_output_facts": dict(sorted(output_facts.items())),
        "rows": rows,
        "summary": {
            "interpretations_improving_identity": identity_wins,
            "interpretations_beating_wrong_stock": stock_wins,
            "interpretations_beating_wrong_pairing": pairing_wins,
        },
        "gates": gates,
        "decision": (
            contract["decision_if_pass"]
            if all(gates.values())
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    result["stable_id"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


__all__ = ["Portra400NaturalProductBaselineError", "evaluate"]
