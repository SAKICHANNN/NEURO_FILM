"""Fit and score one bounded operator on the exact AO0+AO4P Velvia pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.roll2film.adaptive_density_strength import srgb8_to_linear
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)
from src.real_film.velvia_chart_explainability import (
    _fit_affine,
    _fit_record,
    _metrics,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _linear(rows: Any, count: int) -> np.ndarray:
    values = np.asarray(rows)
    if (
        values.shape != (count, 3)
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values > 255)
    ):
        raise ValueError("AO5 RGB8 pair rows are invalid")
    return srgb8_to_linear(values.astype(np.uint8).reshape(-1, 1, 3)).reshape(
        -1, 3
    )


def load_combined_velvia_pairs(
    chart_path: Path,
    palette_path: Path,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load exact chart and stable-pigment Velvia pairs."""

    inputs = config["inputs"]
    chart = json.loads(chart_path.read_bytes())
    chart_film = np.asarray(chart["film_patch_rgb_u8"], dtype=np.uint8)
    chart_reference = np.asarray(
        chart["reference_patch_rgb_u8"], dtype=np.uint8
    )
    chart_paired = np.concatenate((chart_film, chart_reference), axis=1)
    if (
        chart.get("paired_patch_channel_order")
        != ["film_rgb", "reference_rgb"]
        or _sha256(chart_paired.tobytes())
        != inputs["chart_paired_u8_sha256"]
    ):
        raise ValueError("AO5 chart lineage mismatch")

    palette = json.loads(palette_path.read_bytes())
    if (
        not palette.get("automatic_pass")
        or palette.get("asset_sha256") != inputs["palette_asset_sha256"]
        or palette.get("paired_channel_order")
        != ["film_rgb", "reference_rgb"]
    ):
        raise ValueError("AO5 palette lineage mismatch")
    groups = {group["group_id"]: group for group in palette["groups"]}
    group = groups["yoda_velvia_halogen_well_exposed"]
    palette_film = np.asarray(group["film_rgb_u8"], dtype=np.uint8)
    palette_reference = np.asarray(group["reference_rgb_u8"], dtype=np.uint8)
    palette_paired = np.concatenate(
        (palette_film, palette_reference), axis=1
    )
    if (
        group["paired_u8_sha256"]
        != inputs["velvia_palette_paired_u8_sha256"]
        or _sha256(palette_paired.tobytes())
        != inputs["velvia_palette_paired_u8_sha256"]
    ):
        raise ValueError("AO5 palette pair identity mismatch")

    chart_source = _linear(chart_reference, int(inputs["chart_rows"]))
    chart_target = _linear(chart_film, int(inputs["chart_rows"]))
    palette_source = _linear(
        palette_reference, int(inputs["palette_rows"])
    )
    palette_target = _linear(palette_film, int(inputs["palette_rows"]))
    return {
        "velvia_chart": (chart_source, chart_target),
        "velvia_palette": (palette_source, palette_target),
        "combined": (
            np.concatenate((chart_source, palette_source), axis=0),
            np.concatenate((chart_target, palette_target), axis=0),
        ),
    }


def evaluate_combined_velvia_operator(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit the frozen model once on 71 rows and score all declared domains."""

    if set(datasets) != {"velvia_chart", "velvia_palette", "combined"}:
        raise ValueError("AO5 requires exact chart, palette and combined domains")
    source, target = datasets["combined"]
    fit = config["fit"]
    result = fit_positive_film_response_operator(
        source,
        target,
        model="one_matrix",
        identity_mixture=float(fit["identity_mixture"]),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(fit["maximum_function_evaluations"]),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    )
    _, per_channel = _fit_affine(source, target, per_channel=True)
    _, full_affine = _fit_affine(source, target, per_channel=False)
    scores: dict[str, Any] = {}
    for domain in config["score_domains"]:
        domain_source, domain_target = datasets[domain]
        predictions = {
            "identity": domain_source,
            "per_channel_affine": (
                domain_source @ np.asarray(per_channel["matrix"]).T
                + np.asarray(per_channel["bias"])
            ),
            "full_affine": (
                domain_source @ np.asarray(full_affine["matrix"]).T
                + np.asarray(full_affine["bias"])
            ),
            "one_matrix": result.operator.apply(domain_source),
        }
        metrics = {
            name: _metrics(prediction, domain_target)
            for name, prediction in predictions.items()
        }
        one_rmse = metrics["one_matrix"]["rgb_rmse"]
        scores[domain] = {
            "metrics": metrics,
            "one_matrix_gain_over_identity": (
                1.0 - one_rmse / metrics["identity"]["rgb_rmse"]
            ),
            "one_matrix_gain_over_full_affine": (
                1.0 - one_rmse / metrics["full_affine"]["rgb_rmse"]
            ),
        }

    gates = config["gates"]
    checks = [
        {"name": "fit_converged", "passed": result.converged},
        {
            "name": "combined_one_matrix_rgb_rmse",
            "passed": scores["combined"]["metrics"]["one_matrix"]["rgb_rmse"]
            <= float(gates["maximum_combined_one_matrix_rgb_rmse"]),
        },
        {
            "name": "each_domain_one_matrix_rgb_rmse",
            "passed": all(
                scores[name]["metrics"]["one_matrix"]["rgb_rmse"]
                <= float(gates["maximum_each_domain_one_matrix_rgb_rmse"])
                for name in ("velvia_chart", "velvia_palette")
            ),
        },
        {
            "name": "combined_gain_over_identity",
            "passed": scores["combined"]["one_matrix_gain_over_identity"]
            >= float(gates["minimum_combined_gain_over_identity"]),
        },
        {
            "name": "combined_gain_over_full_affine",
            "passed": scores["combined"]["one_matrix_gain_over_full_affine"]
            >= float(gates["minimum_combined_gain_over_full_affine"]),
        },
        {
            "name": "each_domain_gain_over_full_affine",
            "passed": all(
                scores[name]["one_matrix_gain_over_full_affine"]
                >= float(gates["minimum_each_domain_gain_over_full_affine"])
                for name in ("velvia_chart", "velvia_palette")
            ),
        },
        {
            "name": "one_matrix_in_cube",
            "passed": all(
                scores[name]["metrics"]["one_matrix"][
                    "raw_out_of_cube_fraction"
                ]
                <= float(
                    gates["one_matrix_raw_out_of_cube_fraction_maximum"]
                )
                for name in scores
            ),
        },
    ]
    automatic_pass = all(bool(check["passed"]) for check in checks)
    return {
        "fit": _fit_record(result),
        "affine_parameters": {
            "per_channel_affine": per_channel,
            "full_affine": full_affine,
        },
        "scores": scores,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_pass"]
            if automatic_pass
            else config["decision_if_fail"]
        ),
    }


__all__ = [
    "evaluate_combined_velvia_operator",
    "load_combined_velvia_pairs",
]
