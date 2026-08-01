"""U6.P5J development-only positive-PSF compiler and held-frequency audit."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

from src.eval.physical_mtf_source import load_contract as load_source_contract
from src.eval.physical_mtf_source import load_trace as load_development_trace
from src.film_physics.measured_mtf import (
    BUNDLE_SCHEMA,
    ChannelPsf,
    PositivePsfComponent,
)

SCHEMA = "neuro_film.u6_p5j_kodak_250d_positive_psf_compiler_contract.v1"
CONFIRMATION_SCHEMA = "neuro_film.kodak_250d_mtf_confirmation_pixels.v1"
REPORT_SCHEMA = "neuro_film.u6_p5j_kodak_250d_positive_psf_compiler_report.v1"
CHANNELS = ("blue", "green", "red")
FAMILIES = ("single_gaussian", "delta_plus_gaussian", "two_gaussian")


class MeasuredMtfError(RuntimeError):
    """Raised when the P5J contract, evidence or compiler drifts."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MeasuredMtfError("P5J paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent", {})
    development = payload.get("development", {})
    confirmation = payload.get("confirmation", {})
    constraints = payload.get("operator_constraints", {})
    if (
        payload.get("schema") != SCHEMA
        or parent.get("report_sha256")
        != "b32d20f21abc12926d73b56a56e1516e41b35320663aa665e14f6758572f6164"
        or parent.get("stable_evidence_id")
        != "63413b079953e7045a2fde674ebdbf1931cb5c66d0188c6770e5f44dec723a53"
        or development.get("selection") != "leave-one-frequency-out"
        or development.get("candidate_order") != list(FAMILIES)
        or development.get("single_to_delta_min_relative_rmse_improvement") != 0.20
        or development.get("delta_to_two_min_relative_rmse_improvement") != 0.25
        or development.get("sigma_um_bounds") != [0.05, 30.0]
        or development.get("two_gaussian_min_sigma_separation_um") != 1.0
        or development.get("two_gaussian_min_component_weight") != 0.10
        or confirmation.get("frequencies_cycles_per_mm")
        != [27.0, 33.0, 42.0, 52.0, 58.0, 63.0]
        or confirmation.get("all_channel_rmse_ratio_max") != 0.75
        or confirmation.get("per_channel_rmse_ratio_max") != 1.05
        or confirmation.get("green_relative_rmse_improvement_min") != 0.30
        or confirmation.get("red_relative_rmse_improvement_min") != 0.30
        or confirmation.get("selected_max_absolute_error_max") != 0.04
        or confirmation.get("wrong_channel_mean_rmse_ratio_min") != 1.05
        or constraints.get("measured_frequency_interval_cycles_per_mm") != [25.0, 65.0]
        or not constraints.get("positive_spatial_psf")
        or not constraints.get("ringing_forbidden")
        or not constraints.get("extrapolation_forbidden")
    ):
        raise MeasuredMtfError("P5J frozen contract drift")
    for value in (parent.get("report", ""), parent.get("trace", ""), confirmation.get("trace", "")):
        _relative_path(value)
    return payload


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    return float(10.0 ** (math.log10(value0) + fraction * (math.log10(value1) - math.log10(value0))))


def _load_confirmation(config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = root / _relative_path(str(config["confirmation"]["trace"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    frequencies = config["confirmation"]["frequencies_cycles_per_mm"]
    if (
        payload.get("schema") != CONFIRMATION_SCHEMA
        or payload.get("frequencies_cycles_per_mm") != frequencies
        or tuple(payload.get("curves", {})) != CHANNELS
        or any(len(payload["curves"][channel]) != len(frequencies) for channel in CHANNELS)
    ):
        raise MeasuredMtfError("P5J confirmation trace drift")
    development_trace = load_development_trace(root / config["parent"]["trace"])
    axes = development_trace["graph_axes"]
    source_contract = load_source_contract(root / "configs/u6_p5i_kodak_250d_mtf_source_v1.json")
    graph_path = root / source_contract["graph"]["path"]
    if _hash_file(graph_path) != source_contract["graph"]["sha256"]:
        raise MeasuredMtfError("P5J source graph drift")
    image = np.asarray(Image.open(graph_path).convert("L"))
    responses: dict[str, np.ndarray] = {}
    expected = np.asarray(frequencies, dtype=np.float64)
    for channel in CHANNELS:
        coordinates = np.asarray(payload["curves"][channel], dtype=np.int64)
        mapped = np.asarray(
            [_log_value_from_pixel(float(x), axes["x_value_pixels"]) for x in coordinates[:, 0]]
        )
        if np.max(np.abs(mapped - expected)) > 0.35:
            raise MeasuredMtfError("P5J confirmation frequency trace drift")
        for x, y in coordinates:
            if image[int(y), int(x)] >= 128:
                raise MeasuredMtfError("P5J confirmation point is not on source ink")
        responses[channel] = np.asarray(
            [_log_value_from_pixel(float(y), axes["y_value_pixels"]) / 100.0 for y in coordinates[:, 1]],
            dtype=np.float64,
        )
    return payload, responses


def _parameterization(family: str) -> tuple[list[list[float]], tuple[list[float], list[float]]]:
    if family == "single_gaussian":
        return [[math.log(4.0)]], ([math.log(0.05)], [math.log(30.0)])
    if family == "delta_plus_gaussian":
        return (
            [[math.log(4.0), 0.0], [math.log(6.0), 1.0], [math.log(8.0), -1.0]],
            ([math.log(0.05), -6.0], [math.log(30.0), 6.0]),
        )
    return (
        [
            [math.log(0.5), math.log(4.0), 0.0],
            [math.log(2.0), math.log(4.0), 0.0],
            [math.log(0.1), math.log(8.0), 1.0],
            [math.log(3.0), math.log(8.0), -1.0],
        ],
        ([math.log(0.05), math.log(0.05), -6.0], [math.log(30.0), math.log(30.0), 6.0]),
    )


def _decode(family: str, parameters: np.ndarray) -> ChannelPsf:
    if family == "single_gaussian":
        return ChannelPsf(family, (PositivePsfComponent(1.0, float(np.exp(parameters[0]))),))
    weight = float(1.0 / (1.0 + np.exp(-parameters[-1])))
    if family == "delta_plus_gaussian":
        return ChannelPsf(
            family,
            (
                PositivePsfComponent(weight, 0.0),
                PositivePsfComponent(1.0 - weight, float(np.exp(parameters[0]))),
            ),
        )
    sigma1 = float(np.exp(parameters[0]))
    sigma2 = sigma1 + float(np.exp(parameters[1]))
    return ChannelPsf(
        family,
        (
            PositivePsfComponent(weight, sigma1),
            PositivePsfComponent(1.0 - weight, sigma2),
        ),
    )


def _fit(family: str, frequencies: np.ndarray, responses: np.ndarray) -> ChannelPsf:
    starts, bounds = _parameterization(family)
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        result = least_squares(
            lambda value: _decode(family, value).response(frequencies) - responses,
            np.asarray(start, dtype=np.float64),
            bounds=bounds,
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=20_000,
        )
        score = float(np.sum(np.square(result.fun)))
        if best is None or score < best[0]:
            best = (score, result.x)
    if best is None:
        raise MeasuredMtfError("P5J optimizer produced no candidate")
    return _decode(family, best[1])


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def _loocv(family: str, frequencies: np.ndarray, responses: np.ndarray) -> dict[str, float]:
    predictions = []
    for index in range(len(frequencies)):
        keep = np.arange(len(frequencies)) != index
        model = _fit(family, frequencies[keep], responses[keep])
        predictions.append(float(model.response([frequencies[index]])[0]))
    return _metrics(np.asarray(predictions), responses)


def _select(
    config: Mapping[str, Any], candidates: Mapping[str, Mapping[str, Any]]
) -> str:
    development = config["development"]
    selected = "single_gaussian"
    single = candidates[selected]["loocv"]
    delta = candidates["delta_plus_gaussian"]["loocv"]
    delta_gain = 1.0 - delta["rmse"] / single["rmse"]
    if (
        delta_gain >= float(development["single_to_delta_min_relative_rmse_improvement"])
        and delta["max_absolute_error"] <= single["max_absolute_error"]
    ):
        selected = "delta_plus_gaussian"
    if selected == "delta_plus_gaussian":
        two = candidates["two_gaussian"]
        two_cv = two["loocv"]
        two_gain = 1.0 - two_cv["rmse"] / delta["rmse"]
        components = two["model"].components
        valid_components = (
            components[1].sigma_um - components[0].sigma_um
            >= float(development["two_gaussian_min_sigma_separation_um"])
            and min(row.weight for row in components)
            >= float(development["two_gaussian_min_component_weight"])
        )
        if (
            two_gain >= float(development["delta_to_two_min_relative_rmse_improvement"])
            and two_cv["max_absolute_error"] <= delta["max_absolute_error"]
            and valid_components
        ):
            selected = "two_gaussian"
    return selected


def _draw_overlay(
    graph_path: Path,
    confirmation: Mapping[str, Any],
    output: Path,
) -> str:
    image = Image.open(graph_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"blue": (0, 92, 255), "green": (0, 180, 80), "red": (240, 40, 40)}
    for channel in CHANNELS:
        for x, y in confirmation["curves"][channel]:
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline=colors[channel], width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return _hash_file(output)


def compile_and_evaluate(
    config: Mapping[str, Any], root: Path, *, overlay_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_path = root / _relative_path(str(config["parent"]["report"]))
    if _hash_file(parent_path) != config["parent"]["report_sha256"]:
        raise MeasuredMtfError("P5J parent report hash mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("stable_evidence_id") != config["parent"]["stable_evidence_id"] or not parent.get("source_pass"):
        raise MeasuredMtfError("P5J parent evidence mismatch")
    confirmation_trace, confirmation_responses = _load_confirmation(config, root)
    confirmation_frequencies = np.asarray(
        config["confirmation"]["frequencies_cycles_per_mm"], dtype=np.float64
    )

    channel_results: dict[str, Any] = {}
    selected_models: dict[str, ChannelPsf] = {}
    baseline_errors: list[float] = []
    selected_errors: list[float] = []
    for channel in CHANNELS:
        frequencies = np.asarray(parent["channels"][channel]["frequencies_cycles_per_mm"])
        responses = np.asarray(parent["channels"][channel]["responses_percent"]) / 100.0
        candidates: dict[str, Any] = {}
        for family in FAMILIES:
            model = _fit(family, frequencies, responses)
            candidates[family] = {
                "model": model,
                "development": _metrics(model.response(frequencies), responses),
                "loocv": _loocv(family, frequencies, responses),
            }
        selected_family = _select(config, candidates)
        selected = candidates[selected_family]["model"]
        baseline = candidates["single_gaussian"]["model"]
        target = confirmation_responses[channel]
        baseline_prediction = baseline.response(confirmation_frequencies)
        selected_prediction = selected.response(confirmation_frequencies)
        baseline_metric = _metrics(baseline_prediction, target)
        selected_metric = _metrics(selected_prediction, target)
        baseline_errors.extend((baseline_prediction - target).tolist())
        selected_errors.extend((selected_prediction - target).tolist())
        selected_models[channel] = selected
        channel_results[channel] = {
            "candidates": {
                family: {
                    "model": row["model"].to_json(),
                    "development": row["development"],
                    "loocv": row["loocv"],
                }
                for family, row in candidates.items()
            },
            "selected_family": selected_family,
            "confirmation_target": target.tolist(),
            "confirmation_baseline_prediction": baseline_prediction.tolist(),
            "confirmation_selected_prediction": selected_prediction.tolist(),
            "confirmation_baseline": baseline_metric,
            "confirmation_selected": selected_metric,
            "confirmation_rmse_ratio": selected_metric["rmse"] / baseline_metric["rmse"],
            "confirmation_relative_rmse_improvement": 1.0
            - selected_metric["rmse"] / baseline_metric["rmse"],
        }

    right_rmse = [channel_results[channel]["confirmation_selected"]["rmse"] for channel in CHANNELS]
    wrong_rmse = []
    for target_channel in CHANNELS:
        for model_channel in CHANNELS:
            if target_channel != model_channel:
                wrong_rmse.append(
                    _metrics(
                        selected_models[model_channel].response(confirmation_frequencies),
                        confirmation_responses[target_channel],
                    )["rmse"]
                )
    all_ratio = float(
        np.sqrt(np.mean(np.square(selected_errors)))
        / np.sqrt(np.mean(np.square(baseline_errors)))
    )
    wrong_ratio = float(np.mean(wrong_rmse) / np.mean(right_rmse))
    confirmation = config["confirmation"]
    gate_results = {
        "development_selects_simplest_supported_family": [
            channel_results[channel]["selected_family"] for channel in CHANNELS
        ]
        == ["single_gaussian", "delta_plus_gaussian", "two_gaussian"],
        "all_channel_confirmation_rmse": all_ratio
        <= float(confirmation["all_channel_rmse_ratio_max"]),
        "each_channel_confirmation_noninferior": all(
            channel_results[channel]["confirmation_rmse_ratio"]
            <= float(confirmation["per_channel_rmse_ratio_max"])
            for channel in CHANNELS
        ),
        "green_confirmation_gain": channel_results["green"][
            "confirmation_relative_rmse_improvement"
        ]
        >= float(confirmation["green_relative_rmse_improvement_min"]),
        "red_confirmation_gain": channel_results["red"][
            "confirmation_relative_rmse_improvement"
        ]
        >= float(confirmation["red_relative_rmse_improvement_min"]),
        "confirmation_tail": max(
            channel_results[channel]["confirmation_selected"]["max_absolute_error"]
            for channel in CHANNELS
        )
        <= float(confirmation["selected_max_absolute_error_max"]),
        "wrong_channel_negative_control": wrong_ratio
        >= float(confirmation["wrong_channel_mean_rmse_ratio_min"]),
        "positive_dc_monotone_operator": all(
            abs(model.response([0.0])[0] - 1.0) <= 1e-12
            and np.all(np.diff(model.response(np.linspace(0.0, 65.0, 1025))) <= 1e-12)
            and all(component.weight > 0.0 and component.sigma_um >= 0.0 for component in model.components)
            for model in selected_models.values()
        ),
        "no_extrapolation": True,
    }
    bundle_core = {
        "schema": BUNDLE_SCHEMA,
        "source": {
            "film": "KODAK VISION3 250D 5207/7207",
            "measurement_context": parent["measurement_context"],
            "source_sha256": parent["source_sha256"],
            "graph_sha256": parent["graph_sha256"],
            "parent_evidence_id": parent["stable_evidence_id"],
        },
        "measured_frequency_interval_cycles_per_mm": [25.0, 65.0],
        "channels": {channel: selected_models[channel].to_json() for channel in CHANNELS},
        "claim_ceiling": config["claim_ceiling"],
    }
    bundle = {
        **bundle_core,
        "bundle_id": hashlib.sha256(_canonical_json(bundle_core)).hexdigest(),
    }
    overlay_sha = _draw_overlay(
        root / "outputs/u5_r2aa0_source_audit/extracted/250d_mtf.png",
        confirmation_trace,
        overlay_path,
    )
    stable = {
        "experiment_id": config["experiment_id"],
        "config_sha256": _hash_file(root / "configs/u6_p5j_kodak_250d_positive_psf_compiler_v1.json"),
        "parent_report_sha256": _hash_file(parent_path),
        "confirmation_trace_sha256": _hash_file(root / config["confirmation"]["trace"]),
        "confirmation_overlay_sha256": overlay_sha,
        "bundle_id": bundle["bundle_id"],
        "confirmation_frequencies_cycles_per_mm": confirmation_frequencies.tolist(),
        "channel_results": channel_results,
        "all_channel_confirmation_rmse_ratio": all_ratio,
        "wrong_channel_mean_rmse_ratio": wrong_ratio,
        "gate_results": gate_results,
    }
    passed = all(gate_results.values())
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": "open_u6_p5k_synthetic_kernel_audit" if passed else "close_positive_psf_compiler",
        "claim_ceiling": config["claim_ceiling"],
    }
    return bundle, report


__all__ = [
    "CONFIRMATION_SCHEMA",
    "REPORT_SCHEMA",
    "SCHEMA",
    "MeasuredMtfError",
    "compile_and_evaluate",
    "load_contract",
]
