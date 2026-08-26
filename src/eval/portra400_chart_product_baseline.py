"""Evaluate the unchanged product Portra look on one registered film chart."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pipeline_color_baseline import load_guardrail_config
from src.eval.portra400_chart_operator_d1 import _load_samples, _rmse
from src.eval.portra400_same_scene_registration import _raw_rgb
from src.eval.portra400_spektrafilm_author_baseline_d1 import (
    _registered_candidate_samples,
)
from src.inference import load_render_profile, render_three_stock_look_rgb

SCHEMA = "neuro-film.rf3-d8-portra400-chart-product-baseline-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-d8-portra400-chart-product-baseline-result.v1"


class Portra400ChartProductBaselineError(ValueError):
    """Raised when the frozen RF3.D8 sources or execution contract drift."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _fold_rmse(
    candidate: np.ndarray,
    target: np.ndarray,
    blocks: np.ndarray,
    fold_count: int,
) -> list[float]:
    return [
        _rmse(candidate[blocks % fold_count == fold], target[blocks % fold_count == fold])
        for fold in range(fold_count)
    ]


def evaluate(*, root: Path, contract_path: Path, reverse: bool = False) -> dict[str, Any]:
    contract_payload = contract_path.read_bytes()
    contract = json.loads(contract_payload)
    if contract.get("schema") != SCHEMA or contract.get("experiment_id") != "RF3.D8":
        raise Portra400ChartProductBaselineError("unsupported RF3.D8 contract")
    for binding in contract["sources"].values():
        path = root / binding["path"]
        if _sha256_file(path) != binding["sha256"]:
            raise Portra400ChartProductBaselineError(f"source identity drift: {path}")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise Portra400ChartProductBaselineError("parent decision drift")

    cham6_path = root / contract["sources"]["cham6_contract"]["path"]
    cham6 = json.loads(cham6_path.read_text(encoding="utf-8"))
    source_samples, target_samples, blocks, dataset = _load_samples(cham6, root)
    data_root = root / "data/quarantine/spektrafilm_portra400_same_scene_v1"
    source_u8, source_metadata = _raw_rgb(
        data_root / "Digital Lumix S5ii Color Chart.RW2",
        {
            "raw_use_camera_wb": True,
            "raw_no_auto_bright": False,
            "raw_half_size": True,
        },
    )
    source = np.ascontiguousarray(source_u8.astype(np.float32) / 255.0)
    profile = load_render_profile(
        root / contract["sources"]["render_profile"]["path"], root=root
    )
    style_statistics = json.loads(
        (root / contract["sources"]["style_statistics"]["path"]).read_text(
            encoding="utf-8"
        )
    )["styles"]
    arms = [
        ("current_portra_400_look_approximation", "kodak_portra_400", "portra_400"),
        ("current_ektar_100_wrong_stock_control", "kodak_ektar_100", "ektar_100"),
    ]
    if reverse:
        arms.reverse()
    sampled: dict[str, np.ndarray] = {"identity": source_samples}
    output_facts: dict[str, dict[str, Any]] = {}
    sampler_contract = {
        "parents": {
            "cham6_contract": contract["sources"]["cham6_contract"],
        }
    }
    for arm_id, stock_id, style in arms:
        first = render_three_stock_look_rgb(
            source,
            profile=profile,
            film_stock_id=stock_id,
            look_amount=float(contract["execution"]["look_amount"]),
            style_statistics=style_statistics[style],
            guardrails=load_guardrail_config(
                root / contract["sources"]["guardrails"]["path"], style
            ),
            seed=int(contract["execution"]["seed"]),
            tile_size=int(contract["execution"]["tile_size"]),
        )
        second = render_three_stock_look_rgb(
            source,
            profile=profile,
            film_stock_id=stock_id,
            look_amount=float(contract["execution"]["look_amount"]),
            style_statistics=style_statistics[style],
            guardrails=load_guardrail_config(
                root / contract["sources"]["guardrails"]["path"], style
            ),
            seed=int(contract["execution"]["seed"]),
            tile_size=int(contract["execution"]["tile_size"]),
        )
        repeat_error = float(np.max(np.abs(first - second)))
        sampled[arm_id] = _registered_candidate_samples(sampler_contract, root, first)
        output_facts[arm_id] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "maximum_repeat_error": repeat_error,
            "boundary_fraction": float(np.mean((first <= 0.0) | (first >= 1.0))),
        }

    fold_count = int(cham6["sampling"]["fold_count"])
    rmse = {name: _rmse(values, target_samples) for name, values in sampled.items()}
    folds = {
        name: _fold_rmse(values, target_samples, blocks, fold_count)
        for name, values in sampled.items()
    }
    identity_folds = folds["identity"]
    portra_folds = folds["current_portra_400_look_approximation"]
    fold_improvements = [
        1.0 - candidate / identity
        for candidate, identity in zip(portra_folds, identity_folds, strict=True)
    ]
    publisher = json.loads(
        (root / contract["sources"]["cham9_evidence"]["path"]).read_text(
            encoding="utf-8"
        )
    )["observations"]["author_candidate_rmse"]
    metrics = {
        "row_count": len(target_samples),
        "identity_rmse": rmse["identity"],
        "current_portra_400_rmse": rmse["current_portra_400_look_approximation"],
        "current_ektar_100_wrong_stock_rmse": rmse[
            "current_ektar_100_wrong_stock_control"
        ],
        "frozen_publisher_spektrafilm_portra_400_rmse": float(publisher),
        "portra_improvement_over_identity_fraction": 1.0
        - rmse["current_portra_400_look_approximation"] / rmse["identity"],
        "portra_improvement_over_wrong_stock_fraction": 1.0
        - rmse["current_portra_400_look_approximation"]
        / rmse["current_ektar_100_wrong_stock_control"],
        "portra_fold_win_fraction_over_identity": float(
            np.mean(np.asarray(portra_folds) < np.asarray(identity_folds))
        ),
        "portra_worst_fold_improvement_over_identity_fraction": min(
            fold_improvements
        ),
        "maximum_current_output_boundary_fraction": max(
            value["boundary_fraction"] for value in output_facts.values()
        ),
        "maximum_repeat_error": max(
            value["maximum_repeat_error"] for value in output_facts.values()
        ),
    }
    gates_cfg = contract["gates"]
    gates = {
        "minimum_portra_improvement_over_identity": metrics[
            "portra_improvement_over_identity_fraction"
        ]
        >= gates_cfg["minimum_portra_improvement_over_identity_fraction"],
        "minimum_portra_improvement_over_wrong_stock": metrics[
            "portra_improvement_over_wrong_stock_fraction"
        ]
        >= gates_cfg["minimum_portra_improvement_over_wrong_stock_fraction"],
        "minimum_fold_win_fraction_over_identity": metrics[
            "portra_fold_win_fraction_over_identity"
        ]
        >= gates_cfg["minimum_fold_win_fraction_over_identity"],
        "minimum_worst_fold_improvement_over_identity": metrics[
            "portra_worst_fold_improvement_over_identity_fraction"
        ]
        >= gates_cfg["minimum_worst_fold_improvement_over_identity_fraction"],
        "portra_beats_frozen_publisher_baseline": metrics[
            "current_portra_400_rmse"
        ]
        < metrics["frozen_publisher_spektrafilm_portra_400_rmse"],
        "maximum_current_output_boundary_fraction": metrics[
            "maximum_current_output_boundary_fraction"
        ]
        <= gates_cfg["maximum_current_output_boundary_fraction"],
        "maximum_repeat_error": metrics["maximum_repeat_error"]
        <= gates_cfg["maximum_repeat_error"],
    }
    scientific: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "RF3.D8",
        "contract_sha256": hashlib.sha256(contract_payload).hexdigest(),
        "dataset": {**dataset, "source_decode_sha256": source_metadata["decoded_sha256"]},
        "arm_output_facts": dict(sorted(output_facts.items())),
        "arm_rmse": dict(sorted(rmse.items())),
        "fold_rmse": {name: folds[name] for name in sorted(folds)},
        "metrics": metrics,
        "gates": gates,
        "decision": (
            contract["decision_if_pass"]
            if all(gates.values())
            else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["stable_id"] = _stable_id(scientific)
    return scientific


__all__ = ["Portra400ChartProductBaselineError", "evaluate"]
