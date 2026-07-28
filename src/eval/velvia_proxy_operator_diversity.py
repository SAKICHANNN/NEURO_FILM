"""AO8 identifiability audit for two frozen Velvia display-proxy operators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.color import rgb2lab

from src.real_film.gold_matrix_transplant import style_and_basic_residual
from src.roll2film.positive_film import PositiveFilmResponseOperator


class VelviaOperatorDiversityError(ValueError):
    """Raised when AO8 inputs or its frozen contract drift."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _encoded(linear: np.ndarray) -> np.ndarray:
    values = np.asarray(linear, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < -1e-12)
        or np.any(values > 1.0 + 1e-12)
    ):
        raise VelviaOperatorDiversityError("operator output escaped finite RGB cube")
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def residual_alignment(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    minimum_scale: float,
    maximum_scale: float,
) -> dict[str, float]:
    """Fit one bounded non-negative scalar and report unexplained energy."""

    first = np.asarray(reference, dtype=np.float64).reshape(-1)
    second = np.asarray(target, dtype=np.float64).reshape(-1)
    if (
        first.shape != second.shape
        or first.size == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or not 0.0 <= minimum_scale <= maximum_scale
    ):
        raise VelviaOperatorDiversityError("residual alignment inputs are invalid")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise VelviaOperatorDiversityError("residual alignment requires nonzero directions")
    scale = float(
        np.clip(
            np.dot(first, second) / np.dot(first, first),
            minimum_scale,
            maximum_scale,
        )
    )
    fraction = float(np.linalg.norm(second - scale * first) / second_norm)
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return {
        "scale": scale,
        "target_residual_fraction": fraction,
        "cosine_similarity": cosine,
    }


def _pair_metrics(
    source: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    minimum_scale: float,
    maximum_scale: float,
) -> dict[str, Any]:
    first_residual = first - source
    second_residual = second - source
    forward = residual_alignment(
        first_residual,
        second_residual,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )
    reverse = residual_alignment(
        second_residual,
        first_residual,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )
    first_encoded = _encoded(first)
    second_encoded = _encoded(second)
    first_lab = rgb2lab(first_encoded.reshape(-1, 1, 3)).reshape(-1, 3)
    second_lab = rgb2lab(second_encoded.reshape(-1, 1, 3)).reshape(-1, 3)
    delta = np.linalg.norm(first_lab - second_lab, axis=1)
    style, nonbasic = style_and_basic_residual(first_encoded, second_encoded)
    return {
        "residual_cosine_similarity": forward["cosine_similarity"],
        "first_to_second_alignment": forward,
        "second_to_first_alignment": reverse,
        "minimum_symmetric_strength_aligned_residual_fraction": min(
            forward["target_residual_fraction"],
            reverse["target_residual_fraction"],
        ),
        "median_delta_e76": float(np.median(delta)),
        "p95_delta_e76": float(np.quantile(delta, 0.95)),
        "maximum_delta_e76": float(np.max(delta)),
        "style_delta_e76": style,
        "nonbasic_delta_e76": nonbasic,
    }


def _load_exact_json(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(spec["path"])
    if _sha256(path) != spec["sha256"]:
        raise VelviaOperatorDiversityError(f"lineage hash mismatch: {spec['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, PositiveFilmResponseOperator]:
    """Validate all frozen lineage and return the three explicit operators."""

    expected = {
        "experiment_id": "u5.r2ao8-velvia-proxy-operator-diversity-v1",
        "axis_size": 17,
        "minimum": 0.03125,
        "maximum": 0.96875,
        "expected_rows": 4913,
        "minimum_delta": 2.0,
        "minimum_nonbasic": 1.0,
        "minimum_aligned": 0.2,
        "maximum_cosine": 0.98,
    }
    grid = config["evaluation_grid"]
    gates = config["gates"]
    if (
        config.get("experiment_id") != expected["experiment_id"]
        or int(grid["axis_size"]) != expected["axis_size"]
        or float(grid["minimum"]) != expected["minimum"]
        or float(grid["maximum"]) != expected["maximum"]
        or int(grid["expected_rows"]) != expected["expected_rows"]
        or float(gates["minimum_median_chart_palette_delta_e76"])
        != expected["minimum_delta"]
        or float(gates["minimum_nonbasic_chart_palette_delta_e76"])
        != expected["minimum_nonbasic"]
        or float(
            gates["minimum_symmetric_strength_aligned_residual_fraction"]
        )
        != expected["minimum_aligned"]
        or float(gates["maximum_absolute_residual_cosine_similarity"])
        != expected["maximum_cosine"]
        or config["operator_refitting_allowed"]
        or config["training_allowed"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
        or config["latent_stock_mode_claim_allowed"]
        or config["stock_response_claim_allowed"]
    ):
        raise VelviaOperatorDiversityError("AO8 frozen contract mismatch")

    parent = _load_exact_json(root, config["parent_decision"])
    if parent.get("decision") != config["parent_decision"]["required_decision"]:
        raise VelviaOperatorDiversityError("AO8 parent decision mismatch")

    inputs = config["inputs"]
    ao4_config = _load_exact_json(root, inputs["ao4_config"])
    ao4_report = _load_exact_json(root, inputs["ao4_report"])
    ao4_decision = _load_exact_json(root, inputs["ao4_decision"])
    ao5 = _load_exact_json(root, inputs["ao5_combined_operator"])
    if (
        not ao4_report.get("automatic_pass")
        or ao4_report.get("decision")
        != inputs["ao4_report"]["required_decision"]
        or ao4_decision.get("decision")
        != inputs["ao4_report"]["required_decision"]
        or ao4_report.get("config_sha256") != inputs["ao4_config"]["sha256"]
        or ao4_decision.get("config_sha256") != inputs["ao4_config"]["sha256"]
        or ao4_config.get("operator_fitting_allowed") is not True
    ):
        raise VelviaOperatorDiversityError("AO4 retained evidence mismatch")

    directions = ao4_report.get("directions", [])
    by_domain = {row.get("fit_domain"): row for row in directions}
    required = inputs["ao4_report"]["required_fit_domains"]
    if (
        list(required) != ["velvia_chart", "velvia_palette"]
        or set(by_domain) != set(required)
        or not all(by_domain[name].get("automatic_pass") for name in required)
    ):
        raise VelviaOperatorDiversityError("AO4 operator directions mismatch")
    decision_operators = {
        row["fit"]: row["operator_sha256"]
        for row in ao4_decision.get("directions", [])
    }
    for name in required:
        fit = by_domain[name]["fit"]
        if fit.get("operator_sha256") != decision_operators.get(name):
            raise VelviaOperatorDiversityError("AO4 operator identity mismatch")
        if hashlib.sha256(_canonical_json(fit["operator"])).hexdigest() != fit[
            "operator_sha256"
        ]:
            raise VelviaOperatorDiversityError("AO4 serialized operator mismatch")

    combined_payload = ao5["witnesses"]["velvia_combined_one_matrix"]
    combined_hash = hashlib.sha256(_canonical_json(combined_payload)).hexdigest()
    if (
        ao5.get("operator_sha256")
        != inputs["ao5_combined_operator"]["operator_sha256"]
        or combined_hash != ao5["operator_sha256"]
    ):
        raise VelviaOperatorDiversityError("AO5 combined operator mismatch")
    return {
        "chart": PositiveFilmResponseOperator.from_dict(
            by_domain["velvia_chart"]["fit"]["operator"]
        ),
        "palette": PositiveFilmResponseOperator.from_dict(
            by_domain["velvia_palette"]["fit"]["operator"]
        ),
        "combined": PositiveFilmResponseOperator.from_dict(combined_payload),
    }


def evaluate_operator_diversity(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate frozen operator directions without fitting or rendering images."""

    operators = validate_contract(root, config)
    grid_config = config["evaluation_grid"]
    axis = np.linspace(
        float(grid_config["minimum"]),
        float(grid_config["maximum"]),
        int(grid_config["axis_size"]),
        dtype=np.float64,
    )
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    samples = np.stack((rr, gg, bb), axis=-1).reshape(-1, 3)
    if len(samples) != int(grid_config["expected_rows"]):
        raise VelviaOperatorDiversityError("AO8 grid row count mismatch")
    outputs = {
        name: operator.apply(
            samples, strength=float(config["analysis"]["operator_strength"])
        )
        for name, operator in operators.items()
    }
    finite_in_cube = all(
        np.all(np.isfinite(values))
        and np.all(values >= -1e-12)
        and np.all(values <= 1.0 + 1e-12)
        for values in outputs.values()
    )
    alignment = config["analysis"]
    pair = _pair_metrics(
        samples,
        outputs["chart"],
        outputs["palette"],
        minimum_scale=float(alignment["minimum_alignment_scale"]),
        maximum_scale=float(alignment["maximum_alignment_scale"]),
    )
    context = {
        "chart_vs_combined": _pair_metrics(
            samples,
            outputs["chart"],
            outputs["combined"],
            minimum_scale=float(alignment["minimum_alignment_scale"]),
            maximum_scale=float(alignment["maximum_alignment_scale"]),
        ),
        "palette_vs_combined": _pair_metrics(
            samples,
            outputs["palette"],
            outputs["combined"],
            minimum_scale=float(alignment["minimum_alignment_scale"]),
            maximum_scale=float(alignment["maximum_alignment_scale"]),
        ),
    }

    chart_residual = outputs["chart"] - samples
    controls = []
    negative = config["negative_controls"]
    for scale in negative["strength_scales"]:
        scaled = float(scale) * chart_residual
        forward = residual_alignment(
            chart_residual,
            scaled,
            minimum_scale=float(alignment["minimum_alignment_scale"]),
            maximum_scale=float(alignment["maximum_alignment_scale"]),
        )
        reverse = residual_alignment(
            scaled,
            chart_residual,
            minimum_scale=float(alignment["minimum_alignment_scale"]),
            maximum_scale=float(alignment["maximum_alignment_scale"]),
        )
        symmetric_fraction = min(
            forward["target_residual_fraction"],
            reverse["target_residual_fraction"],
        )
        passed = (
            forward["cosine_similarity"]
            >= float(negative["minimum_cosine_similarity"])
            and symmetric_fraction
            <= float(negative["maximum_strength_aligned_residual_fraction"])
        )
        controls.append(
            {
                "strength_scale": float(scale),
                "passed": bool(passed),
                "residual_cosine_similarity": forward["cosine_similarity"],
                "first_to_second_alignment": forward,
                "second_to_first_alignment": reverse,
                "minimum_symmetric_strength_aligned_residual_fraction": (
                    symmetric_fraction
                ),
            }
        )

    gates = config["gates"]
    checks = {
        "ao4_cross_domain_evidence": True,
        "finite_and_in_cube": bool(finite_in_cube),
        "strength_only_negative_controls": all(row["passed"] for row in controls),
        "median_perceptual_disagreement": pair["median_delta_e76"]
        >= float(gates["minimum_median_chart_palette_delta_e76"]),
        "nonbasic_disagreement": pair["nonbasic_delta_e76"]
        >= float(gates["minimum_nonbasic_chart_palette_delta_e76"]),
        "strength_aligned_direction_diversity": pair[
            "minimum_symmetric_strength_aligned_residual_fraction"
        ]
        >= float(
            gates["minimum_symmetric_strength_aligned_residual_fraction"]
        ),
        "cosine_direction_diversity": abs(pair["residual_cosine_similarity"])
        <= float(gates["maximum_absolute_residual_cosine_similarity"]),
    }
    automatic_pass = all(checks.values())
    return {
        "schema": "neuro-film.u5.r2ao8.velvia-proxy-operator-diversity-report.v1",
        "experiment_id": config["experiment_id"],
        "grid_rows": len(samples),
        "chart_palette": pair,
        "combined_operator_context": context,
        "strength_only_negative_controls": controls,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_pass"]
            if automatic_pass
            else config["decision_if_fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "VelviaOperatorDiversityError",
    "evaluate_operator_diversity",
    "residual_alignment",
    "validate_contract",
]
