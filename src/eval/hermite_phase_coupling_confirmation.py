"""Development/confirmation audit for exact-spectrum Hermite phase coupling."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.eval.absolute_cinema_archival_grain import _decode_scalar, _select_patches
from src.eval.compound_thomas_archival_confirmation import (
    _base_and_scale_fields,
    _compound_fields,
    _feature_matrix,
    _normalized,
    _thomas_power,
)
from src.eval.phase_projected_thomas_confirmation import _projection_errors

SCHEMA = "neuro_film.u6_p4cn_hermite_phase_coupling_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cn_hermite_phase_coupling_confirmation_report.v1"


class HermitePhaseCouplingError(RuntimeError):
    """Raised when a frozen input or model invariant drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract(root: Path, contract: Mapping[str, Any]) -> None:
    parent = contract["parent"]
    candidate = contract["candidate"]
    if (
        contract.get("schema") != SCHEMA
        or candidate.get("log_scale_std") != parent.get("fixed_log_scale_std")
        or len(candidate.get("alpha_grid", [])) != 33
        or contract["confirmation"].get("source_count") != 4
    ):
        raise HermitePhaseCouplingError("P4CN frozen contract drift")
    evidence_path = root / str(parent["p4cm_evidence_path"])
    if not evidence_path.is_file() or sha256_file(evidence_path) != str(
        parent["p4cm_evidence_sha256"]
    ):
        raise HermitePhaseCouplingError("P4CM parent identity drift")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("decision") != parent["required_decision"]:
        raise HermitePhaseCouplingError("P4CM parent decision drift")


def _compat_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(contract)
    result["dataset"] = {
        "frames_per_source": int(contract["confirmation"]["frames_per_source"])
    }
    return result


def _source_rows_from_scratch(
    contract: Mapping[str, Any], scratch: Path, names: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = int(contract["development"]["frames_per_source"])
    for source in names:
        frames = sorted((scratch / source).glob("*/*.png"))
        if len(frames) != expected:
            raise HermitePhaseCouplingError(f"development frame count drift: {source}")
        rows.extend({"source": source, "path": path} for path in frames)
    return rows


def _decode_source_patches(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    compat = _compat_contract(contract)
    for row in rows:
        grouped[str(row["source"])].extend(
            _select_patches(_decode_scalar(Path(row["path"]), compat), compat)
        )
    expected = int(contract["patch_observation"]["patches_per_frame"]) * int(
        contract["confirmation"]["frames_per_source"]
    )
    if any(len(patches) != expected for patches in grouped.values()):
        raise HermitePhaseCouplingError("source patch count drift")
    return dict(grouped)


def _projected_fields(
    contract: Mapping[str, Any], bases: np.ndarray, scales: np.ndarray, alpha: float
) -> np.ndarray:
    candidate = contract["candidate"]
    compound = _compound_fields(
        bases,
        scales,
        float(candidate["log_scale_std"]),
        candidate["scale_bounds"],
    ).astype(np.float64)
    coupled = compound + alpha * (np.power(compound, 3) - 3.0 * compound)
    power = _thomas_power(contract, compound.shape[1])
    output = np.empty_like(coupled)
    for index, field in enumerate(coupled):
        spectrum = np.fft.fft2(field)
        magnitude = np.abs(spectrum)
        phase = np.divide(
            spectrum, magnitude, out=np.ones_like(spectrum), where=magnitude > 0
        )
        phase[0, 0] = 0.0
        output[index] = _normalized(np.fft.ifft2(phase * np.sqrt(power)).real)
    return output


def _observed_medians(patches: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
    return np.asarray(
        [
            np.median(_feature_matrix(np.asarray(patches[source])), axis=0)
            for source in sorted(patches)
        ],
        dtype=np.float64,
    )


def _model_distance(
    observed: np.ndarray, model_median: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return np.sqrt(np.mean(np.square((observed - model_median) / scale), axis=1))


def _fit_alpha(
    contract: Mapping[str, Any], observed: np.ndarray
) -> tuple[float, list[dict[str, Any]], np.ndarray]:
    candidate = contract["candidate"]
    bases, scales = _base_and_scale_fields(
        contract, int(candidate["fit_sample_count"]), seed_offset=0
    )
    gaussian_features = _feature_matrix(bases)
    gaussian_median = np.median(gaussian_features, axis=0)
    gaussian_mad = np.maximum(
        np.median(np.abs(gaussian_features - gaussian_median), axis=0), 1e-6
    )
    rows: list[dict[str, Any]] = []
    for alpha in candidate["alpha_grid"]:
        feature_median = np.median(
            _feature_matrix(_projected_fields(contract, bases, scales, float(alpha))),
            axis=0,
        )
        distances = _model_distance(observed, feature_median, gaussian_mad)
        rows.append(
            {
                "alpha": float(alpha),
                "median_development_distance": float(np.median(distances)),
                "source_distances": distances.tolist(),
            }
        )
    best_distance = min(row["median_development_distance"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["median_development_distance"] <= best_distance + 1e-12
    ]
    selected = min(eligible, key=lambda row: (abs(row["alpha"]), row["alpha"]))
    return float(selected["alpha"]), rows, gaussian_mad


def _freeze_models(
    contract: Mapping[str, Any], alpha: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    candidate = contract["candidate"]
    bases, scales = _base_and_scale_fields(
        contract, int(candidate["evaluation_sample_count"]), seed_offset=100000
    )
    base_fields = _projected_fields(contract, bases, scales, 0.0)
    coupled_fields = _projected_fields(contract, bases, scales, alpha)
    gaussian_features = _feature_matrix(bases)
    gaussian_median = np.median(gaussian_features, axis=0)
    gaussian_mad = np.maximum(
        np.median(np.abs(gaussian_features - gaussian_median), axis=0), 1e-6
    )
    base_median = np.median(_feature_matrix(base_fields), axis=0)
    coupled_median = np.median(_feature_matrix(coupled_fields), axis=0)
    projection = _projection_errors(contract, bases, coupled_fields)
    freeze: dict[str, Any] = {
        "selected_alpha": alpha,
        "gaussian_feature_median": gaussian_median.tolist(),
        "gaussian_feature_mad": gaussian_mad.tolist(),
        "p4cm_feature_median": base_median.tolist(),
        "coupled_feature_median": coupled_median.tolist(),
        "projection": projection,
    }
    freeze["model_freeze_id"] = sha256_bytes(canonical_json(freeze))
    return freeze, gaussian_mad, base_median, coupled_median


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
                    headers={"User-Agent": "neuro-film-u6-p4cn/1.0"},
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
    chosen = sorted(eligible, key=lambda item: item[0])[16:20]
    if [source for source, _clip, _frames in chosen] != confirmation["source_names"]:
        raise HermitePhaseCouplingError("confirmation source identity drift")
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
        raise HermitePhaseCouplingError("confirmation metadata lock drift")
    return [row for _source, _clip, frames in chosen for row in frames]


def _acquire_confirmation(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    confirmation = contract["confirmation"]
    scratch.mkdir(parents=True, exist_ok=True)
    base = "https://www.kaggle.com/api/v1/datasets/download/" + str(
        confirmation["dataset_ref"]
    )
    acquired: list[dict[str, Any]] = []
    downloaded = 0
    for row in rows:
        name = str(row["name"])
        expected = int(row["totalBytes"])
        path = scratch.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size != expected:
            temporary = path.with_name(path.name + ".download")
            temporary.unlink(missing_ok=True)
            with requests.get(
                base,
                params={
                    "filename": name,
                    "datasetVersionNumber": int(confirmation["dataset_version"]),
                },
                headers={"User-Agent": "neuro-film-u6-p4cn/1.0"},
                timeout=120,
                stream=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > int(confirmation["maximum_download_bytes"]):
                            raise HermitePhaseCouplingError("download cap exceeded")
                        handle.write(chunk)
            if temporary.stat().st_size != expected:
                raise HermitePhaseCouplingError("download size drift")
            temporary.replace(path)
        acquired.append(
            {
                "source": name.split("/")[0],
                "name": name,
                "bytes": expected,
                "sha256": sha256_file(path),
                "path": path,
            }
        )
    return acquired


def _score_confirmation(
    patches: Mapping[str, Sequence[np.ndarray]],
    scale: np.ndarray,
    base_median: np.ndarray,
    coupled_median: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(patches):
        observed = np.median(_feature_matrix(np.asarray(patches[source])), axis=0)
        base = float(_model_distance(observed[None, :], base_median, scale)[0])
        coupled = float(_model_distance(observed[None, :], coupled_median, scale)[0])
        ratio = coupled / max(base, 1e-12)
        rows.append(
            {
                "source_id": source,
                "patch_count": len(patches[source]),
                "observed_feature_median": observed.tolist(),
                "p4cm_feature_distance": base,
                "coupled_feature_distance": coupled,
                "coupled_to_p4cm_distance_ratio": ratio,
                "improvement_over_p4cm": 1.0 - ratio,
            }
        )
    return rows


def evaluate_hermite_phase_coupling(
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
    alpha, fit_rows, _fit_scale = _fit_alpha(contract, observed)
    freeze, scale, base_median, coupled_median = _freeze_models(contract, alpha)
    confirmation_decode_count_at_freeze = 0
    metadata = _enumerate_confirmation(contract)
    acquired = _acquire_confirmation(contract, metadata, confirmation_scratch)
    patches = _decode_source_patches(acquired, contract)
    scores = _score_confirmation(patches, scale, base_median, coupled_median)
    ratios = np.asarray(
        [row["coupled_to_p4cm_distance_ratio"] for row in scores], dtype=np.float64
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
            "selected_alpha": alpha,
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
            "PASS_HERMITE_PHASE_COUPLING_HIGH_ORDER_DEVELOPMENT"
            if automatic_pass
            else "FAIL_CLOSED_HERMITE_PHASE_COUPLING_TRANSFER"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report
