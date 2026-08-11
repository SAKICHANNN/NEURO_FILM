"""Exact-spectrum marked-Poisson excursion topology confirmation."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.eval.compound_thomas_archival_confirmation import (
    _base_and_scale_fields,
    _feature_matrix,
    _normalized,
    _thomas_power,
)
from src.eval.hermite_phase_coupling_confirmation import (
    _acquire_confirmation,
    _decode_source_patches,
    _model_distance,
    _observed_medians,
    _source_rows_from_scratch,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from src.eval.hermite_phase_coupling_confirmation import (
    _projected_fields as _hermite_projected_fields,
)
from src.eval.phase_projected_thomas_confirmation import _projection_errors

SCHEMA = "neuro_film.u6_p4co_marked_poisson_phase_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4co_marked_poisson_phase_confirmation_report.v1"


class MarkedPoissonPhaseError(RuntimeError):
    """Raised when frozen identities or topology invariants drift."""


def _validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    candidate = contract["candidate"]
    parent = contract["parent"]
    if (
        contract.get("schema") != SCHEMA
        or candidate.get("expected_parent_count_grid") != [4, 8, 16, 32, 64, 128]
        or candidate.get("cluster_sigma_grid_pixels") != [1.0, 2.0, 4.0, 8.0]
        or candidate.get("offspring_per_parent") != 8
        or contract["confirmation"].get("source_count") != 4
    ):
        raise MarkedPoissonPhaseError("P4CO frozen contract drift")
    evidence_path = root / str(parent["p4cn_evidence_path"])
    if not evidence_path.is_file() or sha256_file(evidence_path) != str(
        parent["p4cn_evidence_sha256"]
    ):
        raise MarkedPoissonPhaseError("P4CN parent identity drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("decision") != parent["required_decision"]:
        raise MarkedPoissonPhaseError("P4CN parent decision drift")


def _p4cm_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(contract)
    candidate = result["candidate"]
    candidate["scale_sigma_pixels"] = candidate["p4cm_scale_sigma_pixels"]
    candidate["scale_bounds"] = candidate["p4cm_scale_bounds"]
    candidate["log_scale_std"] = candidate["p4cm_log_scale_std"]
    return result


def _marked_fields(
    contract: Mapping[str, Any],
    count: int,
    expected_parents: int,
    cluster_sigma: float,
    *,
    seed_offset: int,
) -> np.ndarray:
    candidate = contract["candidate"]
    size = int(contract["patch_observation"]["patch_size_pixels"])
    offspring = int(candidate["offspring_per_parent"])
    power = _thomas_power(contract, size)
    output = np.empty((count, size, size), dtype=np.float64)
    for index in range(count):
        rng = np.random.default_rng(int(candidate["candidate_seed"]) + seed_offset + index)
        parent_count = 0
        while parent_count == 0:
            parent_count = int(rng.poisson(expected_parents))
        centers = rng.uniform(0.0, float(size), size=(parent_count, 2))
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=parent_count)
        skeleton = np.zeros((size, size), dtype=np.float64)
        offsets = rng.normal(0.0, cluster_sigma, size=(parent_count, offspring, 2))
        positions = np.rint(centers[:, None, :] + offsets).astype(np.int64) % size
        for parent in range(parent_count):
            np.add.at(
                skeleton,
                (positions[parent, :, 0], positions[parent, :, 1]),
                signs[parent],
            )
        spectrum = np.fft.rfft2(skeleton)
        magnitude = np.abs(spectrum)
        phase = np.divide(
            spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 1e-12
        )
        for y in (0, size // 2):
            for x in (0, size // 2):
                phase[y, x] = complex(np.sign(phase[y, x].real) or 1.0, 0.0)
        phase[0, 0] = 0.0
        output[index] = _normalized(
            np.fft.irfft2(
                phase * np.sqrt(power[:, : size // 2 + 1]),
                s=(size, size),
            )
        )
    return output


def _p4cm_statistics(
    contract: Mapping[str, Any], count: int, *, seed_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compat = _p4cm_contract(contract)
    bases, scales = _base_and_scale_fields(compat, count, seed_offset=seed_offset)
    p4cm = _hermite_projected_fields(compat, bases, scales, 0.0)
    gaussian_features = _feature_matrix(bases)
    gaussian_median = np.median(gaussian_features, axis=0)
    gaussian_mad = np.maximum(
        np.median(np.abs(gaussian_features - gaussian_median), axis=0), 1e-6
    )
    return bases, gaussian_median, gaussian_mad, np.median(_feature_matrix(p4cm), axis=0)


def _fit_topology(
    contract: Mapping[str, Any], observed: np.ndarray
) -> tuple[int, float, list[dict[str, Any]]]:
    candidate = contract["candidate"]
    count = int(candidate["fit_sample_count"])
    _bases, _gaussian, scale, _p4cm = _p4cm_statistics(
        contract, count, seed_offset=0
    )
    rows: list[dict[str, Any]] = []
    for parents in candidate["expected_parent_count_grid"]:
        for sigma in candidate["cluster_sigma_grid_pixels"]:
            fields = _marked_fields(
                contract,
                count,
                int(parents),
                float(sigma),
                seed_offset=0,
            )
            model_median = np.median(_feature_matrix(fields), axis=0)
            distances = _model_distance(observed, model_median, scale)
            rows.append(
                {
                    "expected_parent_count": int(parents),
                    "cluster_sigma_pixels": float(sigma),
                    "model_feature_median": model_median.tolist(),
                    "median_development_distance": float(np.median(distances)),
                    "source_distances": distances.tolist(),
                }
            )
    best = min(row["median_development_distance"] for row in rows)
    eligible = [
        row for row in rows if row["median_development_distance"] <= best + 1e-12
    ]
    selected = min(
        eligible,
        key=lambda row: (row["expected_parent_count"], row["cluster_sigma_pixels"]),
    )
    return (
        int(selected["expected_parent_count"]),
        float(selected["cluster_sigma_pixels"]),
        rows,
    )


def _freeze_models(
    contract: Mapping[str, Any], expected_parents: int, cluster_sigma: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    count = int(contract["candidate"]["evaluation_sample_count"])
    bases, gaussian_median, gaussian_mad, p4cm_median = _p4cm_statistics(
        contract, count, seed_offset=100000
    )
    marked = _marked_fields(
        contract,
        count,
        expected_parents,
        cluster_sigma,
        seed_offset=100000,
    )
    marked_median = np.median(_feature_matrix(marked), axis=0)
    projection = _projection_errors(contract, bases, marked)
    freeze: dict[str, Any] = {
        "expected_parent_count": expected_parents,
        "cluster_sigma_pixels": cluster_sigma,
        "gaussian_feature_median": gaussian_median.tolist(),
        "gaussian_feature_mad": gaussian_mad.tolist(),
        "p4cm_feature_median": p4cm_median.tolist(),
        "marked_feature_median": marked_median.tolist(),
        "projection": projection,
    }
    freeze["model_freeze_id"] = sha256_bytes(canonical_json(freeze))
    return freeze, gaussian_mad, p4cm_median, marked_median


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
                    headers={"User-Agent": "neuro-film-u6-p4co/1.0"},
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
    chosen = sorted(eligible, key=lambda item: item[0])[20:24]
    if [source for source, _clip, _frames in chosen] != confirmation["source_names"]:
        raise MarkedPoissonPhaseError("confirmation source identity drift")
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
        raise MarkedPoissonPhaseError("confirmation metadata lock drift")
    return [row for _source, _clip, frames in chosen for row in frames]


def _score_confirmation(
    patches: Mapping[str, list[np.ndarray]],
    scale: np.ndarray,
    p4cm_median: np.ndarray,
    marked_median: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(patches):
        observed = np.median(_feature_matrix(np.asarray(patches[source])), axis=0)
        baseline = float(_model_distance(observed[None, :], p4cm_median, scale)[0])
        marked = float(_model_distance(observed[None, :], marked_median, scale)[0])
        ratio = marked / max(baseline, 1e-12)
        rows.append(
            {
                "source_id": source,
                "patch_count": len(patches[source]),
                "observed_feature_median": observed.tolist(),
                "p4cm_feature_distance": baseline,
                "marked_feature_distance": marked,
                "marked_to_p4cm_distance_ratio": ratio,
                "improvement_over_p4cm": 1.0 - ratio,
            }
        )
    return rows


def evaluate_marked_poisson_phase(
    root: Path, contract_path: Path, confirmation_scratch: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(root, contract)
    development_rows = _source_rows_from_scratch(
        contract,
        Path(contract["development"]["source_scratch"]),
        contract["development"]["source_names"],
    )
    development_patches = _decode_source_patches(development_rows, contract)
    observed = _observed_medians(development_patches)
    expected_parents, cluster_sigma, fit_rows = _fit_topology(contract, observed)
    freeze, scale, p4cm_median, marked_median = _freeze_models(
        contract, expected_parents, cluster_sigma
    )
    confirmation_decode_count_at_freeze = 0
    metadata = _enumerate_confirmation(contract)
    acquired = _acquire_confirmation(contract, metadata, confirmation_scratch)
    patches = _decode_source_patches(acquired, contract)
    scores = _score_confirmation(patches, scale, p4cm_median, marked_median)
    ratios = np.asarray(
        [row["marked_to_p4cm_distance_ratio"] for row in scores], dtype=np.float64
    )
    gates_spec = contract["confirmation_gates"]
    second = contract["second_order"]
    projection = freeze["projection"]
    gates = {
        "source_wins": int(np.count_nonzero(ratios < 1.0))
        >= int(gates_spec["minimum_sources_beating_p4cm_feature_distance"]),
        "median_feature_distance_improvement": float(np.median(1.0 - ratios))
        >= float(gates_spec["minimum_median_feature_distance_improvement_over_p4cm"]),
        "worst_feature_distance": float(np.max(ratios))
        <= float(gates_spec["maximum_worst_feature_distance_ratio_to_p4cm"]),
        "exact_power_projection": projection["maximum_per_sample_power_relative_error"]
        <= float(second["maximum_per_sample_power_relative_error"]),
        "exact_acf_projection": projection["maximum_per_sample_acf_absolute_error"]
        <= float(second["maximum_per_sample_acf_absolute_error"]),
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
            "sources": sorted(development_patches),
            "selected_expected_parent_count": expected_parents,
            "selected_cluster_sigma_pixels": cluster_sigma,
            "fit_rows": fit_rows,
        },
        "model_freeze": freeze,
        "confirmation_metadata_lock_sha256": contract["confirmation"][
            "metadata_lock_sha256"
        ],
        "confirmation_total_bytes": sum(int(row["bytes"]) for row in acquired),
        "confirmation_source_sha256": [row["sha256"] for row in acquired],
        "confirmation_decode_count_at_model_freeze": confirmation_decode_count_at_freeze,
        "confirmation_scores": scores,
        "summary": {
            "sources_beating_p4cm": int(np.count_nonzero(ratios < 1.0)),
            "median_feature_distance_improvement_over_p4cm": float(
                np.median(1.0 - ratios)
            ),
            "worst_feature_distance_ratio_to_p4cm": float(np.max(ratios)),
        },
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "PASS_MARKED_POISSON_PHASE_HIGH_ORDER_DEVELOPMENT"
            if automatic_pass
            else "FAIL_CLOSED_MARKED_POISSON_PHASE_TRANSFER"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report
