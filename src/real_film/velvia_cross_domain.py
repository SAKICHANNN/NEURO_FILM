"""Cross-domain validation for the AO0/AO4P Velvia display proxies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.roll2film.adaptive_density_strength import srgb8_to_linear
from src.roll2film.positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)
from src.real_film.velvia_chart_explainability import (
    _fit_affine,
    _fit_record,
    _metrics,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _to_linear(rows: Any, *, expected_rows: int) -> np.ndarray:
    values = np.asarray(rows)
    if (
        values.shape != (expected_rows, 3)
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values > 255)
    ):
        raise ValueError("paired proxy RGB8 rows have invalid shape or range")
    return srgb8_to_linear(values.astype(np.uint8).reshape(-1, 1, 3)).reshape(
        -1, 3
    )


def load_cross_domain_pairs(
    chart_path: Path,
    palette_path: Path,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load exact chart and palette pairs as source-reference -> target-film."""

    inputs = config["inputs"]
    chart = json.loads(chart_path.read_bytes())
    if (
        chart.get("paired_patch_u8_sha256")
        != inputs["chart_paired_u8_sha256"]
        or chart.get("paired_patch_channel_order")
        != ["film_rgb", "reference_rgb"]
    ):
        raise ValueError("AO4C chart lineage mismatch")
    chart_film_u8 = np.asarray(chart["film_patch_rgb_u8"], dtype=np.uint8)
    chart_reference_u8 = np.asarray(
        chart["reference_patch_rgb_u8"], dtype=np.uint8
    )
    if (
        _sha256(
            np.concatenate((chart_film_u8, chart_reference_u8), axis=1).tobytes()
        )
        != inputs["chart_paired_u8_sha256"]
    ):
        raise ValueError("AO4C chart paired bytes mismatch")

    palette = json.loads(palette_path.read_bytes())
    if (
        palette.get("asset_sha256") != inputs["palette_asset_sha256"]
        or palette.get("paired_channel_order") != ["film_rgb", "reference_rgb"]
        or not palette.get("automatic_pass")
    ):
        raise ValueError("AO4C palette lineage mismatch")
    groups = {group["group_id"]: group for group in palette["groups"]}
    expected = {
        "yoda_velvia_halogen_well_exposed": (
            47,
            inputs["velvia_palette_paired_u8_sha256"],
        ),
        "cactus_ektachrome_halogen_well_exposed": (
            56,
            inputs["ektachrome_palette_paired_u8_sha256"],
        ),
    }
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "velvia_chart": (
            _to_linear(chart_reference_u8, expected_rows=24),
            _to_linear(chart_film_u8, expected_rows=24),
        )
    }
    for output_name, group_id in (
        ("velvia_palette", "yoda_velvia_halogen_well_exposed"),
        ("ektachrome_palette", "cactus_ektachrome_halogen_well_exposed"),
    ):
        group = groups[group_id]
        count, expected_hash = expected[group_id]
        film_u8 = np.asarray(group["film_rgb_u8"], dtype=np.uint8)
        reference_u8 = np.asarray(group["reference_rgb_u8"], dtype=np.uint8)
        paired = np.concatenate((film_u8, reference_u8), axis=1)
        if (
            group["paired_u8_sha256"] != expected_hash
            or _sha256(paired.tobytes()) != expected_hash
        ):
            raise ValueError("AO4C palette paired bytes mismatch")
        output[output_name] = (
            _to_linear(reference_u8, expected_rows=count),
            _to_linear(film_u8, expected_rows=count),
        )
    return output


def _fit_options(config: dict[str, Any]) -> dict[str, Any]:
    fit = config["fit"]
    return {
        "identity_mixture": float(fit["identity_mixture"]),
        "restart_count": int(fit["restart_count"]),
        "maximum_function_evaluations": int(
            fit["maximum_function_evaluations"]
        ),
        "function_tolerance": float(fit["function_tolerance"]),
        "parameter_tolerance": float(fit["parameter_tolerance"]),
        "gradient_tolerance": float(fit["gradient_tolerance"]),
        "loss": str(fit["loss"]),
        "loss_scale": float(fit["loss_scale"]),
        "seed": int(fit["seed"]),
    }


def _fit_bundle(
    source: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    _, per_channel = _fit_affine(source, target, per_channel=True)
    _, full_affine = _fit_affine(source, target, per_channel=False)
    one = fit_positive_film_response_operator(
        source, target, model="one_matrix", **_fit_options(config)
    )
    return {
        "per_channel_affine": per_channel,
        "full_affine": full_affine,
        "one_matrix_result": one,
    }


def _predict_bundle(
    source: np.ndarray,
    fitted: dict[str, Any],
) -> dict[str, np.ndarray]:
    predictions = {"identity": source}
    for name in ("per_channel_affine", "full_affine"):
        parameters = fitted[name]
        predictions[name] = (
            source @ np.asarray(parameters["matrix"]).T
            + np.asarray(parameters["bias"])
        )
    result: PositiveFilmFitResult = fitted["one_matrix_result"]
    predictions["one_matrix"] = result.operator.apply(source)
    return predictions


def _score_direction(
    fit_name: str,
    confirm_name: str,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    fitted: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    source, target = datasets[confirm_name]
    predictions = _predict_bundle(source, fitted)
    metrics = {
        name: _metrics(prediction, target)
        for name, prediction in predictions.items()
    }
    one_rmse = metrics["one_matrix"]["rgb_rmse"]
    identity_rmse = metrics["identity"]["rgb_rmse"]
    affine_rmse = metrics["full_affine"]["rgb_rmse"]
    gain_identity = 1.0 - one_rmse / identity_rmse
    gain_affine = 1.0 - one_rmse / affine_rmse
    checks = [
        {
            "name": "one_matrix_rgb_rmse",
            "passed": one_rmse
            <= float(gates["maximum_one_matrix_rgb_rmse_each_direction"]),
        },
        {
            "name": "gain_over_identity",
            "passed": gain_identity
            >= float(
                gates["minimum_one_matrix_gain_over_identity_each_direction"]
            ),
        },
        {
            "name": "gain_over_full_affine",
            "passed": gain_affine
            >= float(
                gates[
                    "minimum_one_matrix_gain_over_full_affine_each_direction"
                ]
            ),
        },
    ]
    result: PositiveFilmFitResult = fitted["one_matrix_result"]
    return {
        "fit_domain": fit_name,
        "confirmation_domain": confirm_name,
        "fit": _fit_record(result),
        "affine_parameters": {
            "per_channel_affine": fitted["per_channel_affine"],
            "full_affine": fitted["full_affine"],
        },
        "confirmation_metrics": metrics,
        "one_matrix_gain_over_identity": gain_identity,
        "one_matrix_gain_over_full_affine": gain_affine,
        "checks": checks,
        "automatic_pass": result.converged
        and all(bool(check["passed"]) for check in checks),
    }


def evaluate_cross_domain(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit on each proxy and evaluate bidirectional held-domain transfer."""

    required = {"velvia_chart", "velvia_palette", "ektachrome_palette"}
    if set(datasets) != required:
        raise ValueError("AO4C requires all three exact proxy domains")
    fitted = {
        name: _fit_bundle(*datasets[name], config)
        for name in sorted(required)
    }
    gates = config["gates"]
    directions = [
        _score_direction(
            row["fit"], row["confirm"], datasets, fitted[row["fit"]], gates
        )
        for row in config["positive_cross_domain_directions"]
    ]

    velvia_palette_source, velvia_palette_target = datasets["velvia_palette"]
    ektachrome_source, ektachrome_target = datasets["ektachrome_palette"]
    diagnostics = {
        "chart_fit_on_ektachrome_palette": {
            name: _metrics(prediction, ektachrome_target)
            for name, prediction in _predict_bundle(
                ektachrome_source, fitted["velvia_chart"]
            ).items()
        },
        "ektachrome_fit_on_velvia_palette": {
            name: _metrics(prediction, velvia_palette_target)
            for name, prediction in _predict_bundle(
                velvia_palette_source, fitted["ektachrome_palette"]
            ).items()
        },
        "claim": config["diagnostic_wrong_stock"]["claim"],
    }
    automatic_pass = all(direction["automatic_pass"] for direction in directions)
    return {
        "directions": directions,
        "wrong_stock_diagnostics": diagnostics,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_pass"]
            if automatic_pass
            else config["decision_if_fail"]
        ),
    }


__all__ = ["evaluate_cross_domain", "load_cross_domain_pairs"]
