"""Evaluate fixed Velvia product and AO6 looks on the BALICA chart proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pipeline_color_baseline import load_guardrail_config
from src.film_physics.display_look import build_source_context_display_look
from src.inference import load_render_profile, render_three_stock_look_rgb

SCHEMA = "neuro-film.rf3-d10-velvia50-chart-product-baseline-contract.v1"
REPORT_SCHEMA = "neuro-film.rf3-d10-velvia50-chart-product-baseline-result.v1"


class Velvia50ChartProductBaselineError(ValueError):
    """Raised when a frozen RF3.D10 source or execution fact drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _stable_id(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mosaic(patches: np.ndarray, rows: int, columns: int, patch_size: int) -> np.ndarray:
    values = np.asarray(patches, dtype=np.float32)
    if values.shape != (rows * columns, 3) or patch_size <= 0:
        raise Velvia50ChartProductBaselineError("patch mosaic geometry drift")
    return np.ascontiguousarray(
        np.repeat(
            np.repeat(values.reshape(rows, columns, 3), patch_size, axis=0),
            patch_size,
            axis=1,
        )
    )


def _sample_mosaic(
    image: np.ndarray, rows: int, columns: int, patch_size: int
) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (rows * patch_size, columns * patch_size, 3):
        raise Velvia50ChartProductBaselineError("rendered mosaic geometry drift")
    return np.asarray(
        [
            np.median(
                value[
                    row * patch_size : (row + 1) * patch_size,
                    column * patch_size : (column + 1) * patch_size,
                ],
                axis=(0, 1),
            )
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.float64,
    )


def _rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def _fold_rmse(candidate: np.ndarray, target: np.ndarray, rows: int) -> list[float]:
    if candidate.shape != target.shape or candidate.shape[0] % rows:
        raise Velvia50ChartProductBaselineError("fold geometry drift")
    columns = candidate.shape[0] // rows
    return [
        _rmse(
            candidate[row * columns : (row + 1) * columns],
            target[row * columns : (row + 1) * columns],
        )
        for row in range(rows)
    ]


def _load_bound_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _sha256_file(path) != binding["sha256"]:
        raise Velvia50ChartProductBaselineError(f"bound source drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Velvia50ChartProductBaselineError(f"expected JSON object: {path}")
    return value


def evaluate(*, root: Path, contract_path: Path, reverse: bool = False) -> dict[str, Any]:
    contract_payload = contract_path.read_bytes()
    contract = json.loads(contract_payload)
    if contract.get("schema") != SCHEMA or contract.get("experiment_id") != "RF3.D10":
        raise Velvia50ChartProductBaselineError("unsupported RF3.D10 contract")
    for binding in contract["sources"].values():
        path = root / binding["path"]
        if not path.is_file() or _sha256_file(path) != binding["sha256"]:
            raise Velvia50ChartProductBaselineError(f"source identity drift: {path}")

    paired = _load_bound_json(root, contract["sources"]["paired_patches"])
    source_config = _load_bound_json(root, contract["sources"]["source_contract"])
    source_u8 = np.asarray(paired["reference_patch_rgb_u8"], dtype=np.uint8)
    target_u8 = np.asarray(paired["film_patch_rgb_u8"], dtype=np.uint8)
    pair_bytes = np.concatenate((target_u8, source_u8), axis=1).tobytes()
    expected_pair_sha = source_config["extraction"]["expected_paired_patch_u8_sha256"]
    if (
        source_u8.shape != (24, 3)
        or target_u8.shape != (24, 3)
        or paired["paired_patch_u8_sha256"] != expected_pair_sha
        or hashlib.sha256(pair_bytes).hexdigest() != expected_pair_sha
    ):
        raise Velvia50ChartProductBaselineError("paired patch identity drift")

    execution = contract["execution"]
    rows = int(execution["chart_rows"])
    columns = int(execution["chart_columns"])
    patch_size = int(execution["patch_size"])
    source_patches = source_u8.astype(np.float32) / 255.0
    target_patches = target_u8.astype(np.float64) / 255.0
    source = _mosaic(source_patches, rows, columns, patch_size)

    profile = load_render_profile(
        root / contract["sources"]["render_profile"]["path"], root=root
    )
    style_statistics = _load_bound_json(
        root, contract["sources"]["style_statistics"]
    )["styles"]
    product_arms = [
        ("current_velvia_50_look_approximation", "fujifilm_velvia_50", "velvia_50"),
        ("current_portra_400_wrong_stock_control", "kodak_portra_400", "portra_400"),
        ("current_ektar_100_wrong_stock_control", "kodak_ektar_100", "ektar_100"),
    ]
    if reverse:
        product_arms.reverse()

    sampled: dict[str, np.ndarray] = {"identity": source_patches.astype(np.float64)}
    output_facts: dict[str, dict[str, Any]] = {}
    for arm_id, stock_id, style in product_arms:
        kwargs = {
            "profile": profile,
            "film_stock_id": stock_id,
            "look_amount": float(execution["look_amount"]),
            "style_statistics": style_statistics[style],
            "guardrails": load_guardrail_config(
                root / contract["sources"]["guardrails"]["path"], style
            ),
            "seed": int(execution["seed"]),
            "tile_size": int(execution["tile_size"]),
        }
        first = render_three_stock_look_rgb(source, **kwargs)
        second = render_three_stock_look_rgb(source, **kwargs)
        sampled[arm_id] = _sample_mosaic(first, rows, columns, patch_size)
        output_facts[arm_id] = {
            "output_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
            "maximum_repeat_error": float(np.max(np.abs(first - second))),
            "boundary_fraction": float(np.mean((first <= 0.0) | (first >= 1.0))),
        }

    frozen_report = _load_bound_json(root, contract["sources"]["ao6_frozen_report"])
    if frozen_report.get("artifact_canonical_sha256") != contract["sources"][
        "ao6_frozen_report"
    ]["artifact_canonical_sha256"]:
        raise Velvia50ChartProductBaselineError("AO6 artifact identity drift")
    ao6_payload = frozen_report["artifact"]["component_payloads"][
        "ao6-source-context-display-look"
    ]
    if _stable_id(ao6_payload) != contract["sources"]["ao6_frozen_report"][
        "display_payload_sha256"
    ]:
        raise Velvia50ChartProductBaselineError("AO6 display payload drift")
    ao6_first = build_source_context_display_look(ao6_payload, source)(source)
    ao6_second = build_source_context_display_look(ao6_payload, source)(source)
    ao6_id = "ao6_velvia_50_display_proxy_baseline"
    sampled[ao6_id] = _sample_mosaic(ao6_first, rows, columns, patch_size)
    output_facts[ao6_id] = {
        "output_sha256": hashlib.sha256(ao6_first.tobytes()).hexdigest(),
        "maximum_repeat_error": float(np.max(np.abs(ao6_first - ao6_second))),
        "boundary_fraction": float(np.mean((ao6_first <= 0.0) | (ao6_first >= 1.0))),
    }

    rmse = {name: _rmse(value, target_patches) for name, value in sampled.items()}
    folds = {
        name: _fold_rmse(value, target_patches, rows) for name, value in sampled.items()
    }
    velvia_id = "current_velvia_50_look_approximation"
    wrong_ids = [
        "current_portra_400_wrong_stock_control",
        "current_ektar_100_wrong_stock_control",
    ]
    fold_improvements = [
        1.0 - candidate / identity
        for candidate, identity in zip(folds[velvia_id], folds["identity"], strict=True)
    ]
    metrics = {
        "patch_count": int(source_u8.shape[0]),
        "identity_rmse": rmse["identity"],
        "current_velvia_50_rmse": rmse[velvia_id],
        "ao6_velvia_50_rmse": rmse[ao6_id],
        "current_portra_400_wrong_stock_rmse": rmse[wrong_ids[0]],
        "current_ektar_100_wrong_stock_rmse": rmse[wrong_ids[1]],
        "velvia_improvement_over_identity_fraction": 1.0
        - rmse[velvia_id] / rmse["identity"],
        "velvia_improvement_over_portra_fraction": 1.0
        - rmse[velvia_id] / rmse[wrong_ids[0]],
        "velvia_improvement_over_ektar_fraction": 1.0
        - rmse[velvia_id] / rmse[wrong_ids[1]],
        "velvia_improvement_over_ao6_fraction": 1.0 - rmse[velvia_id] / rmse[ao6_id],
        "velvia_fold_win_fraction_over_identity": float(
            np.mean(np.asarray(folds[velvia_id]) < np.asarray(folds["identity"]))
        ),
        "velvia_worst_fold_improvement_over_identity_fraction": min(
            fold_improvements
        ),
        "maximum_current_output_boundary_fraction": max(
            output_facts[name]["boundary_fraction"] for name, _, _ in product_arms
        ),
        "maximum_repeat_error": max(
            value["maximum_repeat_error"] for value in output_facts.values()
        ),
    }
    gate_config = contract["gates"]
    gates = {
        "minimum_velvia_improvement_over_identity": metrics[
            "velvia_improvement_over_identity_fraction"
        ]
        >= gate_config["minimum_velvia_improvement_over_identity_fraction"],
        "minimum_velvia_improvement_over_each_wrong_stock": min(
            metrics["velvia_improvement_over_portra_fraction"],
            metrics["velvia_improvement_over_ektar_fraction"],
        )
        >= gate_config["minimum_velvia_improvement_over_each_wrong_stock_fraction"],
        "minimum_fold_win_fraction_over_identity": metrics[
            "velvia_fold_win_fraction_over_identity"
        ]
        >= gate_config["minimum_fold_win_fraction_over_identity"],
        "minimum_worst_fold_improvement_over_identity": metrics[
            "velvia_worst_fold_improvement_over_identity_fraction"
        ]
        >= gate_config["minimum_worst_fold_improvement_over_identity_fraction"],
        "maximum_current_output_boundary_fraction": metrics[
            "maximum_current_output_boundary_fraction"
        ]
        <= gate_config["maximum_current_output_boundary_fraction"],
        "maximum_repeat_error": metrics["maximum_repeat_error"]
        <= gate_config["maximum_repeat_error"],
    }
    scientific: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": "RF3.D10",
        "contract_sha256": hashlib.sha256(contract_payload).hexdigest(),
        "source": {
            "paired_patch_sha256": expected_pair_sha,
            "rows": rows,
            "columns": columns,
            "comparison_domain": "author-rendered-encoded-srgb",
        },
        "arm_output_facts": dict(sorted(output_facts.items())),
        "arm_rmse": dict(sorted(rmse.items())),
        "fold_rmse": {name: folds[name] for name in sorted(folds)},
        "metrics": metrics,
        "gates": gates,
        "decision": (
            contract["decision_if_pass"] if all(gates.values()) else contract["decision_if_fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["stable_id"] = _stable_id(scientific)
    return scientific


__all__ = [
    "Velvia50ChartProductBaselineError",
    "_fold_rmse",
    "_mosaic",
    "_sample_mosaic",
    "_stable_id",
    "evaluate",
]
