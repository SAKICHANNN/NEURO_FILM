"""U6.P4CD Poisson-binomial density-granularity compatibility baseline."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro_film.u6_p4cd_poisson_binomial_granularity_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cd_poisson_binomial_granularity_report.v1"
CHANNELS = ("blue", "green", "red")


class PoissonBinomialGranularityError(RuntimeError):
    """Raised when a frozen P4CD input or contract drifts."""


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


def _relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PoissonBinomialGranularityError(
            "P4CD paths must be repository-relative"
        )
    return root / relative


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    split = payload.get("split", {})
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != SCHEMA
        or model.get("uniformity") != 0.98
        or model.get("post_result_parameter_or_family_change_allowed")
        or split.get("development")
        != "eligible sorted index modulo 3 is 0 or 2"
        or split.get("confirmation")
        != "eligible sorted index modulo 3 is 1"
        or split.get("minimum_development_rows_per_channel") != 10
        or split.get("minimum_confirmation_rows_per_channel") != 5
        or gates.get("minimum_improvement_vs_channel_constant") != 0.1
        or gates.get("minimum_improvement_vs_no_competition") != 0.05
        or gates.get("minimum_improvement_vs_characteristic_slope") != 0.05
        or gates.get("maximum_confirmation_median_log_sigma_error") != 0.25
        or gates.get("maximum_confirmation_p95_log_sigma_error") != 0.7
        or gates.get("maximum_worst_channel_median_log_sigma_error") != 0.4
        or gates.get("minimum_median_channel_spearman") != 0.35
        or gates.get("minimum_channel_spearman") != 0.0
        or not gates.get("require_positive_shared_amplitude")
        or not gates.get("require_finite_nonnegative_prediction")
        or not gates.get("two_byte_identical_reports")
    ):
        raise PoissonBinomialGranularityError("P4CD frozen contract drift")
    return payload


def _load_json_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> tuple[dict[str, Any], str]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    actual = _hash_file(path) if path.is_file() else ""
    if actual != parents[f"{stem}_sha256"]:
        raise PoissonBinomialGranularityError(
            f"P4CD parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8")), actual


def _validate_inputs(
    contract: Mapping[str, Any], root: Path
) -> tuple[
    dict[str, Any],
    ManufacturerCharacteristicPrior,
    dict[str, Any],
    dict[str, str],
]:
    source = contract["primary_source"]
    report_path = _relative(root, str(source["report_path"]))
    if (
        not report_path.is_file()
        or report_path.stat().st_size != source["report_bytes"]
        or _hash_file(report_path) != source["report_sha256"]
    ):
        raise PoissonBinomialGranularityError("P4CD primary source drift")

    parents = contract["parents"]
    payloads: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {"primary_report": _hash_file(report_path)}
    for stem in (
        "p4av_contract",
        "p4av_report",
        "p2q_contract",
        "p2q_bundle",
        "p4aw_decision",
        "p4aw_report",
    ):
        payloads[stem], identities[stem] = _load_json_parent(parents, root, stem)

    p4av = payloads["p4av_report"]
    p4aw_decision = payloads["p4aw_decision"]
    p4aw_report = payloads["p4aw_report"]
    if (
        p4av.get("source_pass") is not True
        or p4aw_decision.get("decision")
        != "close_effective_same_sheet_joint_family_without_rescue"
        or p4aw_decision.get("report_sha256") != identities["p4aw_report"]
        or p4aw_report.get("stable_evidence_id")
        != p4aw_decision.get("stable_evidence_id")
    ):
        raise PoissonBinomialGranularityError("P4CD parent decision mismatch")
    prior = ManufacturerCharacteristicPrior.from_dict(payloads["p2q_bundle"]["prior"])
    return p4av, prior, p4aw_report, identities


def _build_rows(
    p4av: Mapping[str, Any], prior: ManufacturerCharacteristicPrior
) -> list[dict[str, Any]]:
    curves = {curve.layer: curve for curve in prior.curves}
    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        curve = curves[channel]
        lower, upper = curve.density_bounds
        span = upper - lower
        if not math.isfinite(span) or span <= 0.0:
            raise PoissonBinomialGranularityError("P4CD invalid density span")
        eligible: list[dict[str, Any]] = []
        source = p4av["channels"][channel]
        for exposure_value, sigma_value in zip(
            source["log_relative_exposure"], source["sigma_d"], strict=True
        ):
            exposure = float(exposure_value)
            sigma = float(sigma_value)
            if curve.domain[0] <= exposure <= curve.domain[1]:
                density = float(curve.apply(np.asarray([exposure]))[0])
                probability = min(1.0, max(0.0, (density - lower) / span))
                eligible.append(
                    {
                        "channel": channel,
                        "log_exposure": exposure,
                        "sigma_d": sigma,
                        "variance": sigma * sigma,
                        "density": density,
                        "normalized_density": probability,
                    }
                )
        eligible.sort(key=lambda item: item["log_exposure"])
        for index, row in enumerate(eligible):
            row["eligible_index"] = index
            row["role"] = "confirmation" if index % 3 == 1 else "development"
            rows.append(row)
    return rows


def _design(
    rows: Sequence[Mapping[str, Any]], *, uniformity: float
) -> np.ndarray:
    matrix = np.zeros((len(rows), 4), dtype=np.float64)
    for index, row in enumerate(rows):
        channel = str(row["channel"])
        matrix[index, CHANNELS.index(channel)] = 1.0
        probability = float(row["normalized_density"])
        matrix[index, 3] = probability * (1.0 - uniformity * probability)
    return matrix


def _fit_predict(
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
    *,
    uniformity: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray([float(row["variance"]) for row in development])
    parameters, residual_norm = nnls(_design(development, uniformity=uniformity), target)
    variance = _design(confirmation, uniformity=uniformity) @ parameters
    prediction = np.sqrt(np.maximum(variance, np.finfo(np.float64).tiny))
    return prediction, {
        "uniformity": uniformity,
        "channel_floor_variance": {
            channel: float(parameters[index]) for index, channel in enumerate(CHANNELS)
        },
        "shared_amplitude": float(parameters[3]),
        "development_residual_norm": float(residual_norm),
    }


def _constant_predict(
    development: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    floor = {
        channel: float(
            np.mean(
                [
                    float(row["variance"])
                    for row in development
                    if row["channel"] == channel
                ]
            )
        )
        for channel in CHANNELS
    }
    return (
        np.asarray([math.sqrt(floor[str(row["channel"])]) for row in confirmation]),
        {"channel_floor_variance": floor, "shared_amplitude": 0.0},
    )


def _score(
    confirmation: Sequence[Mapping[str, Any]], prediction: np.ndarray
) -> dict[str, Any]:
    observed = np.asarray([float(row["sigma_d"]) for row in confirmation])
    error = np.abs(np.log(prediction / observed))
    per_channel: dict[str, Any] = {}
    correlations: list[float] = []
    for channel in CHANNELS:
        indices = [
            index for index, row in enumerate(confirmation) if row["channel"] == channel
        ]
        channel_observed = observed[indices]
        channel_prediction = prediction[indices]
        if (
            np.unique(channel_observed).size < 2
            or np.unique(channel_prediction).size < 2
        ):
            correlation = -1.0
        else:
            correlation = float(
                spearmanr(channel_observed, channel_prediction).statistic
            )
            if not math.isfinite(correlation):
                correlation = -1.0
        correlations.append(correlation)
        channel_error = error[indices]
        per_channel[channel] = {
            "row_count": len(indices),
            "median_log_sigma_error": float(np.median(channel_error)),
            "p95_log_sigma_error": float(np.percentile(channel_error, 95.0)),
            "spearman": correlation,
        }
    return {
        "median_log_sigma_error": float(np.median(error)),
        "p95_log_sigma_error": float(np.percentile(error, 95.0)),
        "maximum_log_sigma_error": float(np.max(error)),
        "median_channel_spearman": float(np.median(correlations)),
        "minimum_channel_spearman": float(np.min(correlations)),
        "per_channel": per_channel,
        "predicted_sigma_d": prediction.tolist(),
        "point_log_sigma_error": error.tolist(),
    }


def evaluate_compatibility(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    p4av, prior, p4aw, parent_sha = _validate_inputs(contract, root)
    rows = _build_rows(p4av, prior)
    development = [row for row in rows if row["role"] == "development"]
    confirmation = [row for row in rows if row["role"] == "confirmation"]
    split_counts = {
        channel: {
            "development": sum(row["channel"] == channel for row in development),
            "confirmation": sum(row["channel"] == channel for row in confirmation),
        }
        for channel in CHANNELS
    }
    split = contract["split"]
    if any(
        counts["development"] < split["minimum_development_rows_per_channel"]
        or counts["confirmation"] < split["minimum_confirmation_rows_per_channel"]
        for counts in split_counts.values()
    ):
        raise PoissonBinomialGranularityError("P4CD eligible split is undersized")

    p4aw_rows = p4aw["confirmation_rows"]
    row_identity = [
        (row["channel"], row["eligible_index"], row["log_exposure"], row["sigma_d"])
        for row in confirmation
    ]
    p4aw_identity = [
        (row["channel"], row["eligible_index"], row["log_exposure"], row["sigma_d"])
        for row in p4aw_rows
    ]
    if row_identity != p4aw_identity:
        raise PoissonBinomialGranularityError("P4CD/P4AW confirmation split drift")

    predictions: dict[str, np.ndarray] = {}
    parameters: dict[str, Any] = {}
    predictions["channel_constant"], parameters["channel_constant"] = (
        _constant_predict(development, confirmation)
    )
    predictions["no_competition_poisson"], parameters["no_competition_poisson"] = (
        _fit_predict(development, confirmation, uniformity=0.0)
    )
    predictions["candidate"], parameters["candidate"] = _fit_predict(
        development,
        confirmation,
        uniformity=float(contract["model"]["uniformity"]),
    )
    predictions["characteristic_slope"] = np.asarray(
        p4aw["scores"]["characteristic_slope_without_p5j"]["predicted_sigma_d"],
        dtype=np.float64,
    )
    parameters["characteristic_slope"] = p4aw["parameters"][
        "characteristic_slope_without_p5j"
    ]

    scores = {name: _score(confirmation, value) for name, value in predictions.items()}
    candidate_median = float(scores["candidate"]["median_log_sigma_error"])
    improvements = {
        name: (float(scores[name]["median_log_sigma_error"]) - candidate_median)
        / float(scores[name]["median_log_sigma_error"])
        for name in (
            "channel_constant",
            "no_competition_poisson",
            "characteristic_slope",
        )
    }
    worst_channel_median = max(
        float(item["median_log_sigma_error"])
        for item in scores["candidate"]["per_channel"].values()
    )
    prediction = predictions["candidate"]
    gates = contract["gates"]
    gate_results = {
        "parent_identity": True,
        "split_identity": True,
        "improvement_vs_channel_constant": improvements["channel_constant"]
        >= float(gates["minimum_improvement_vs_channel_constant"]),
        "improvement_vs_no_competition": improvements["no_competition_poisson"]
        >= float(gates["minimum_improvement_vs_no_competition"]),
        "improvement_vs_characteristic_slope": improvements["characteristic_slope"]
        >= float(gates["minimum_improvement_vs_characteristic_slope"]),
        "confirmation_median_error": candidate_median
        <= float(gates["maximum_confirmation_median_log_sigma_error"]),
        "confirmation_p95_error": float(scores["candidate"]["p95_log_sigma_error"])
        <= float(gates["maximum_confirmation_p95_log_sigma_error"]),
        "worst_channel_median_error": worst_channel_median
        <= float(gates["maximum_worst_channel_median_log_sigma_error"]),
        "median_channel_spearman": float(
            scores["candidate"]["median_channel_spearman"]
        )
        >= float(gates["minimum_median_channel_spearman"]),
        "minimum_channel_spearman": float(
            scores["candidate"]["minimum_channel_spearman"]
        )
        >= float(gates["minimum_channel_spearman"]),
        "positive_shared_amplitude": float(parameters["candidate"]["shared_amplitude"])
        > 0.0,
        "finite_nonnegative_prediction": bool(
            np.all(np.isfinite(prediction)) and np.all(prediction >= 0.0)
        ),
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "parent_sha256": parent_sha,
        "p2q_prior_identity": prior.identity(),
        "split_counts": split_counts,
        "parameters": parameters,
        "scores": scores,
        "control_improvement": improvements,
        "worst_channel_median_log_sigma_error": worst_channel_median,
        "confirmation_rows": confirmation,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_poisson_binomial_density_marginal_baseline"
            if automatic_pass
            else "close_poisson_binomial_density_marginal_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "PoissonBinomialGranularityError",
    "evaluate_compatibility",
    "load_contract",
    "write_report",
]
