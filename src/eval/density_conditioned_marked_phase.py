"""Current-density-conditioned exact-spectrum marked grain confirmation."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy import ndimage

from src.eval.absolute_cinema_archival_grain import _decode_scalar
from src.eval.compound_thomas_archival_confirmation import _feature_matrix
from src.eval.hermite_phase_coupling_confirmation import (
    _acquire_confirmation,
    _model_distance,
    _source_rows_from_scratch,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from src.eval.marked_poisson_phase_confirmation import (
    _marked_fields,
    _p4cm_statistics,
    _projection_errors,
)

SCHEMA = "neuro_film.u6_p4cp_density_conditioned_marked_phase_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cp_density_conditioned_marked_phase_report.v1"


class DensityConditionedMarkedPhaseError(RuntimeError):
    """Raised when a frozen density-conditioned invariant drifts."""


def _validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    parent = contract["parent"]
    candidate = contract["candidate"]
    observation = contract["patch_observation"]
    if (
        contract.get("schema") != SCHEMA
        or candidate.get("expected_parent_count_grid") != [4, 8, 16, 32, 64, 128]
        or candidate.get("cluster_sigma_pixels") != parent["fixed_cluster_sigma_pixels"]
        or candidate.get("offspring_per_parent") != parent["fixed_offspring_per_parent"]
        or observation.get("density_split") != 0.5
    ):
        raise DensityConditionedMarkedPhaseError("P4CP frozen contract drift")
    evidence_path = root / str(parent["p4co_evidence_path"])
    if not evidence_path.is_file() or sha256_file(evidence_path) != str(
        parent["p4co_evidence_sha256"]
    ):
        raise DensityConditionedMarkedPhaseError("P4CO parent identity drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("decision") != parent["required_decision"]:
        raise DensityConditionedMarkedPhaseError("P4CO parent decision drift")


def _select_patch_observations(
    scalar: np.ndarray, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rule = contract["patch_observation"]
    size = int(rule["patch_size_pixels"])
    border = int(rule["border_pixels"])
    low = ndimage.gaussian_filter(
        scalar, sigma=float(rule["lowpass_gaussian_sigma_pixels"]), mode="reflect"
    )
    gy, gx = np.gradient(low)
    minimum, maximum = map(float, rule["accepted_patch_mean_interval"])
    candidates: list[tuple[float, int, int, float]] = []
    for y in range(border, scalar.shape[0] - border - size + 1, size):
        for x in range(border, scalar.shape[1] - border - size + 1, size):
            low_patch = low[y : y + size, x : x + size]
            mean = float(np.mean(low_patch, dtype=np.float64))
            if minimum <= mean <= maximum:
                score = math.sqrt(
                    float(
                        np.mean(
                            np.square(gy[y : y + size, x : x + size])
                            + np.square(gx[y : y + size, x : x + size]),
                            dtype=np.float64,
                        )
                    )
                )
                candidates.append((score, y, x, mean))
    candidates.sort()
    count = int(rule["patches_per_frame"])
    if len(candidates) < count:
        raise DensityConditionedMarkedPhaseError("insufficient flat patches")
    output: list[dict[str, Any]] = []
    for _score, y, x, mean in candidates[:count]:
        residual = scalar[y : y + size, x : x + size] - low[
            y : y + size, x : x + size
        ]
        residual = residual - float(np.mean(residual, dtype=np.float64))
        rms = math.sqrt(float(np.mean(np.square(residual), dtype=np.float64)))
        if rms < 1e-6:
            raise DensityConditionedMarkedPhaseError("selected residual is degenerate")
        output.append(
            {"mean": mean, "residual": np.ascontiguousarray(residual / rms)}
        )
    return output


def _decode_observations(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].extend(
            _select_patch_observations(_decode_scalar(Path(row["path"]), contract), contract)
        )
    expected = int(contract["patch_observation"]["patches_per_frame"]) * int(
        contract["confirmation"]["frames_per_source"]
    )
    if any(len(values) != expected for values in grouped.values()):
        raise DensityConditionedMarkedPhaseError("patch count drift")
    return dict(grouped)


def _flatten_observations(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    means: list[float] = []
    residuals: list[np.ndarray] = []
    sources: list[str] = []
    for source in sorted(grouped):
        for row in grouped[source]:
            means.append(float(row["mean"]))
            residuals.append(np.asarray(row["residual"]))
            sources.append(source)
    return (
        np.asarray(means, dtype=np.float64),
        _feature_matrix(np.asarray(residuals)),
        sources,
    )


def _model_bank(
    contract: Mapping[str, Any], *, seed_offset: int
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, dict[int, dict[str, float]]]:
    candidate = contract["candidate"]
    count = int(candidate["evaluation_sample_count_per_count"])
    bases, _gaussian, scale, p4cm_median = _p4cm_statistics(
        contract, count, seed_offset=seed_offset
    )
    medians: dict[int, np.ndarray] = {}
    projections: dict[int, dict[str, float]] = {}
    for parents in candidate["expected_parent_count_grid"]:
        fields = _marked_fields(
            contract,
            count,
            int(parents),
            float(candidate["cluster_sigma_pixels"]),
            seed_offset=seed_offset,
        )
        medians[int(parents)] = np.median(_feature_matrix(fields), axis=0)
        projections[int(parents)] = _projection_errors(contract, bases, fields)
    return medians, scale, p4cm_median, projections


def _fit_counts(
    contract: Mapping[str, Any], means: np.ndarray, features: np.ndarray
) -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    split = float(contract["patch_observation"]["density_split"])
    minimum = int(contract["patch_observation"]["minimum_development_patches_per_bin"])
    medians, scale, _p4cm, _projections = _model_bank(contract, seed_offset=0)
    masks = {"low": means < split, "high": means >= split}
    support = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    if any(value < minimum for value in support.values()):
        raise DensityConditionedMarkedPhaseError("development density support gate failed")
    rows: list[dict[str, Any]] = []
    selected: dict[str, int] = {}
    for name, mask in masks.items():
        for parents in contract["candidate"]["expected_parent_count_grid"]:
            distances = _model_distance(features[mask], medians[int(parents)], scale)
            rows.append(
                {
                    "density_bin": name,
                    "expected_parent_count": int(parents),
                    "median_patch_distance": float(np.median(distances)),
                }
            )
        selected[name] = min(
            (
                row
                for row in rows
                if row["density_bin"] == name
            ),
            key=lambda row: (row["median_patch_distance"], row["expected_parent_count"]),
        )["expected_parent_count"]
    return selected, support, rows


def _enumerate_confirmation(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    confirmation = contract["confirmation"]
    params: dict[str, Any] = {"pageSize": 200}
    rows: list[dict[str, Any]] = []
    session = requests.Session()
    for _ in range(16):
        for attempt in range(3):
            try:
                response = session.get(
                    str(confirmation["api_url"]),
                    params=params,
                    headers={"User-Agent": "neuro-film-u6-p4cp/1.0"},
                    timeout=60,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(float(attempt + 1))
        payload = response.json()
        rows.extend(payload.get("datasetFiles", []))
        token = payload.get("nextPageTokenNullable")
        if not token:
            break
        params = {"pageSize": 200, "pageToken": token}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        name = str(row.get("name", ""))
        parts = name.split("/")
        if len(parts) == 3 and name.endswith(".png"):
            grouped[parts[0]][parts[1]].append(row)
    eligible: list[tuple[str, str, list[dict[str, Any]]]] = []
    for source, clips in grouped.items():
        clip = min(clips)
        frames = sorted(clips[clip], key=lambda item: str(item["name"]))
        if len(frames) == int(confirmation["frames_per_source"]):
            eligible.append((source, clip, frames))
    chosen = sorted(eligible, key=lambda item: item[0])[24:28]
    if [source for source, _clip, _frames in chosen] != confirmation["source_names"]:
        raise DensityConditionedMarkedPhaseError("confirmation source identity drift")
    lock = {
        "dataset_ref": confirmation["dataset_ref"],
        "version": confirmation["dataset_version"],
        "selected": [
            {
                "source": source,
                "clip": clip,
                "files": [
                    {"name": str(row["name"]), "bytes": int(row["totalBytes"])}
                    for row in frames
                ],
            }
            for source, clip, frames in chosen
        ],
    }
    compact = (json.dumps(lock, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if sha256_bytes(compact) != confirmation["metadata_lock_sha256"]:
        raise DensityConditionedMarkedPhaseError("confirmation metadata lock drift")
    return [row for _source, _clip, frames in chosen for row in frames]


def _score_confirmation(
    contract: Mapping[str, Any],
    observations: Mapping[str, Sequence[Mapping[str, Any]]],
    selected: Mapping[str, int],
    medians: Mapping[int, np.ndarray],
    scale: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split = float(contract["patch_observation"]["density_split"])
    fixed = medians[64]
    rows: list[dict[str, Any]] = []
    support = {"low": 0, "high": 0}
    for source in sorted(observations):
        features = _feature_matrix(
            np.asarray([np.asarray(row["residual"]) for row in observations[source]])
        )
        means = np.asarray([float(row["mean"]) for row in observations[source]])
        bins = np.where(means < split, "low", "high")
        for name in bins:
            support[str(name)] += 1
        fixed_distances = _model_distance(features, fixed, scale)
        conditioned = np.asarray(
            [
                _model_distance(features[index : index + 1], medians[selected[str(name)]], scale)[0]
                for index, name in enumerate(bins)
            ],
            dtype=np.float64,
        )
        fixed_median = float(np.median(fixed_distances))
        conditioned_median = float(np.median(conditioned))
        ratio = conditioned_median / max(fixed_median, 1e-12)
        rows.append(
            {
                "source_id": source,
                "patch_count": len(features),
                "low_patch_count": int(np.count_nonzero(bins == "low")),
                "high_patch_count": int(np.count_nonzero(bins == "high")),
                "fixed_p4co_patch_distance_median": fixed_median,
                "conditioned_patch_distance_median": conditioned_median,
                "conditioned_to_fixed_distance_ratio": ratio,
                "improvement_over_fixed_p4co": 1.0 - ratio,
            }
        )
    return rows, support


def evaluate_density_conditioned_marked_phase(
    root: Path, contract_path: Path, confirmation_scratch: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(root, contract)
    development_rows = _source_rows_from_scratch(
        contract,
        Path(contract["development"]["source_scratch"]),
        contract["development"]["source_names"],
    )
    development = _decode_observations(development_rows, contract)
    means, features, _sources = _flatten_observations(development)
    selected, development_support, fit_rows = _fit_counts(contract, means, features)
    medians, scale, p4cm_median, projections = _model_bank(
        contract, seed_offset=100000
    )
    freeze: dict[str, Any] = {
        "selected_parent_counts": selected,
        "development_support": development_support,
        "gaussian_feature_mad": scale.tolist(),
        "p4cm_feature_median": p4cm_median.tolist(),
        "marked_feature_medians": {
            str(key): value.tolist() for key, value in medians.items()
        },
        "selected_projection": {
            name: projections[count] for name, count in selected.items()
        },
    }
    freeze["model_freeze_id"] = sha256_bytes(canonical_json(freeze))
    confirmation_decode_count_at_freeze = 0
    metadata = _enumerate_confirmation(contract)
    acquired = _acquire_confirmation(contract, metadata, confirmation_scratch)
    confirmation = _decode_observations(acquired, contract)
    scores, confirmation_support = _score_confirmation(
        contract, confirmation, selected, medians, scale
    )
    ratios = np.asarray(
        [row["conditioned_to_fixed_distance_ratio"] for row in scores],
        dtype=np.float64,
    )
    selected_projection = [projections[count] for count in selected.values()]
    max_power = max(
        row["maximum_per_sample_power_relative_error"] for row in selected_projection
    )
    max_acf = max(
        row["maximum_per_sample_acf_absolute_error"] for row in selected_projection
    )
    gate_spec = contract["confirmation_gates"]
    minimum_support = int(gate_spec["minimum_confirmation_patches_per_bin"])
    gates = {
        "source_wins": int(np.count_nonzero(ratios < 1.0))
        >= int(gate_spec["minimum_sources_beating_fixed_p4co"]),
        "median_source_improvement": float(np.median(1.0 - ratios))
        >= float(gate_spec["minimum_median_source_improvement_over_fixed_p4co"]),
        "worst_source_distance": float(np.max(ratios))
        <= float(gate_spec["maximum_worst_source_distance_ratio_to_fixed_p4co"]),
        "confirmation_bin_support": all(
            value >= minimum_support for value in confirmation_support.values()
        ),
        "exact_power_projection": max_power
        <= float(contract["second_order"]["maximum_per_sample_power_relative_error"]),
        "exact_acf_projection": max_acf
        <= float(contract["second_order"]["maximum_per_sample_acf_absolute_error"]),
        "finite_fields": all(math.isfinite(float(value)) for value in ratios),
        "confirmation_pixels_unread_at_freeze": confirmation_decode_count_at_freeze
        == 0,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_bytes(canonical_json(contract)),
        "development": {
            "sources": sorted(development),
            "selected_parent_counts": selected,
            "bin_support": development_support,
            "fit_rows": fit_rows,
        },
        "model_freeze": freeze,
        "confirmation_metadata_lock_sha256": contract["confirmation"][
            "metadata_lock_sha256"
        ],
        "confirmation_total_bytes": sum(int(row["bytes"]) for row in acquired),
        "confirmation_source_sha256": [row["sha256"] for row in acquired],
        "confirmation_decode_count_at_model_freeze": confirmation_decode_count_at_freeze,
        "confirmation_bin_support": confirmation_support,
        "confirmation_scores": scores,
        "summary": {
            "sources_beating_fixed_p4co": int(np.count_nonzero(ratios < 1.0)),
            "median_source_improvement_over_fixed_p4co": float(np.median(1.0 - ratios)),
            "worst_source_distance_ratio_to_fixed_p4co": float(np.max(ratios)),
        },
        "projection": {
            "maximum_per_sample_power_relative_error": max_power,
            "maximum_per_sample_acf_absolute_error": max_acf,
        },
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "PASS_DENSITY_CONDITIONED_MARKED_PHASE_DEVELOPMENT"
            if automatic_pass
            else "FAIL_CLOSED_DENSITY_CONDITIONED_MARKED_PHASE_TRANSFER"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report
