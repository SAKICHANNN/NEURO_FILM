"""Held-source cross-component archival grain-correlation audit."""

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
from PIL import Image
from scipy import ndimage

SCHEMA = "neuro_film.u6_p4cq_cross_component_grain_correlation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cq_cross_component_grain_correlation_report.v1"


class CrossComponentGrainCorrelationError(RuntimeError):
    """Raised when the frozen cross-component experiment drifts."""


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


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CrossComponentGrainCorrelationError("P4CQ contract schema drift")
    confirmation = payload.get("confirmation", {})
    candidate = payload.get("candidate", {})
    metrics = payload.get("metrics", {})
    if (
        confirmation.get("source_count") != 2
        or confirmation.get("frames_per_source") != 10
        or confirmation.get("expected_total_bytes") != 14791583
        or candidate.get("family")
        != "single-shared-lower-triangular-cross-component-correlation"
        or metrics.get("minimum_sources_beating_independent") != 2
        or metrics.get("minimum_sources_beating_cyclic_control") != 2
    ):
        raise CrossComponentGrainCorrelationError("P4CQ frozen contract drift")
    if set(payload["development"]["source_names"]) & set(
        confirmation["source_names"]
    ):
        raise CrossComponentGrainCorrelationError("P4CQ source roles overlap")
    return payload


def _source_rows_from_scratch(
    contract: Mapping[str, Any], scratch: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = int(contract["development"]["frames_per_source"])
    for source in contract["development"]["source_names"]:
        frames = sorted((scratch / str(source)).glob("*/*.png"))
        if len(frames) != expected:
            raise CrossComponentGrainCorrelationError(
                f"development frame count drift: {source}"
            )
        rows.extend({"source": str(source), "path": path} for path in frames)
    return rows


def _decode_rgb(path: Path, contract: Mapping[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != contract["decode"]["accepted_png_mode"]:
            raise CrossComponentGrainCorrelationError(
                f"confirmation PNG mode drift: {image.mode}"
            )
        values = np.asarray(image)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[-1] != 3:
        raise CrossComponentGrainCorrelationError("decoded RGB shape or dtype drift")
    return np.ascontiguousarray(values.astype(np.float64) / 255.0)


def _select_normalized_residuals(
    rgb: np.ndarray, contract: Mapping[str, Any]
) -> list[np.ndarray]:
    rule = contract["patch_observation"]
    size = int(rule["patch_size_pixels"])
    border = int(rule["border_pixels"])
    sigma = float(rule["lowpass_gaussian_sigma_pixels"])
    low = ndimage.gaussian_filter(rgb, sigma=(sigma, sigma, 0.0), mode="reflect")
    weights = np.asarray(rule["selection_luminance"], dtype=np.float64)
    low_luma = np.tensordot(low, weights, axes=([-1], [0]))
    gy, gx = np.gradient(low_luma)
    minimum, maximum = map(float, rule["accepted_patch_luminance_interval"])
    candidates: list[tuple[float, int, int]] = []
    for y in range(border, rgb.shape[0] - border - size + 1, size):
        for x in range(border, rgb.shape[1] - border - size + 1, size):
            patch = low_luma[y : y + size, x : x + size]
            mean = float(np.mean(patch, dtype=np.float64))
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
                candidates.append((score, y, x))
    candidates.sort()
    count = int(rule["patches_per_frame"])
    if len(candidates) < count:
        raise CrossComponentGrainCorrelationError("insufficient flat RGB patches")
    output: list[np.ndarray] = []
    for _score, y, x in candidates[:count]:
        residual = np.asarray(
            rgb[y : y + size, x : x + size]
            - low[y : y + size, x : x + size],
            dtype=np.float64,
        )
        residual -= np.mean(residual, axis=(0, 1), dtype=np.float64)
        rms = np.sqrt(np.mean(np.square(residual), axis=(0, 1), dtype=np.float64))
        if np.any(rms < 1e-6) or not np.all(np.isfinite(rms)):
            raise CrossComponentGrainCorrelationError("RGB residual is degenerate")
        normalized = np.ascontiguousarray(residual / rms)
        normalized.setflags(write=False)
        output.append(normalized)
    return output


def _correlation_matrix(residual: np.ndarray) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise CrossComponentGrainCorrelationError("invalid normalized RGB residual")
    flattened = values.reshape(-1, 3)
    matrix = flattened.T @ flattened / float(len(flattened))
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 1.0)
    return np.ascontiguousarray(matrix)


def _fit_shared_matrix(
    residuals: Sequence[np.ndarray], contract: Mapping[str, Any]
) -> np.ndarray:
    if not residuals:
        raise CrossComponentGrainCorrelationError("no development residuals")
    matrix = np.mean(
        np.asarray([_correlation_matrix(value) for value in residuals]), axis=0
    )
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 1.0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    off_diagonal = matrix[np.triu_indices(3, 1)]
    candidate = contract["candidate"]
    if (
        not np.all(np.isfinite(matrix))
        or float(np.min(eigenvalues)) < float(candidate["minimum_eigenvalue"])
        or float(np.max(np.abs(off_diagonal)))
        > float(candidate["maximum_absolute_off_diagonal"])
    ):
        raise CrossComponentGrainCorrelationError(
            "fitted cross-component matrix failed the frozen envelope"
        )
    output = np.ascontiguousarray(matrix)
    output.setflags(write=False)
    return output


def _matrix_error(observed: np.ndarray, candidate: np.ndarray) -> float:
    indexes = np.triu_indices(3, 1)
    return math.sqrt(
        float(np.mean(np.square(observed[indexes] - candidate[indexes])))
    )


def _group_observations(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, list[np.ndarray]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].extend(
            _select_normalized_residuals(_decode_rgb(Path(row["path"]), contract), contract)
        )
    expected = int(contract["confirmation"]["frames_per_source"]) * int(
        contract["patch_observation"]["patches_per_frame"]
    )
    if any(len(values) != expected for values in grouped.values()):
        raise CrossComponentGrainCorrelationError("RGB patch count drift")
    return dict(grouped)


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
                    headers={"User-Agent": "neuro-film-u6-p4cq/1.0"},
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
    chosen = sorted(eligible, key=lambda item: item[0])[28:30]
    if [source for source, _clip, _frames in chosen] != confirmation["source_names"]:
        raise CrossComponentGrainCorrelationError("confirmation source identity drift")
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
        raise CrossComponentGrainCorrelationError("confirmation metadata lock drift")
    flattened = [row for _source, _clip, frames in chosen for row in frames]
    if sum(int(row["totalBytes"]) for row in flattened) != int(
        confirmation["expected_total_bytes"]
    ):
        raise CrossComponentGrainCorrelationError("confirmation byte total drift")
    return flattened


def _acquire_confirmation(
    contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], scratch: Path
) -> list[dict[str, Any]]:
    confirmation = contract["confirmation"]
    scratch.mkdir(parents=True, exist_ok=True)
    base = "https://www.kaggle.com/api/v1/datasets/download/" + str(
        confirmation["dataset_ref"]
    )
    downloaded = 0
    acquired: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        expected = int(row["totalBytes"])
        path = scratch.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size != expected:
            temporary = path.with_name(path.name + ".download")
            if temporary.exists():
                temporary.unlink()
            with requests.get(
                base,
                params={
                    "filename": name,
                    "datasetVersionNumber": int(confirmation["dataset_version"]),
                },
                headers={"User-Agent": "neuro-film-u6-p4cq/1.0"},
                timeout=120,
                stream=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if downloaded > int(confirmation["maximum_download_bytes"]):
                                raise CrossComponentGrainCorrelationError(
                                    "confirmation download exceeded frozen cap"
                                )
            if temporary.stat().st_size != expected:
                raise CrossComponentGrainCorrelationError(
                    "confirmation member size drift"
                )
            temporary.replace(path)
        acquired.append(
            {
                "source": name.split("/", 1)[0],
                "name": name,
                "bytes": expected,
                "sha256": sha256_file(path),
                "path": path,
            }
        )
    return acquired


def evaluate_cross_component_grain_correlation(
    root: Path, contract_path: Path, scratch: Path
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    development_rows = _source_rows_from_scratch(
        contract, Path(str(contract["development"]["source_scratch"]))
    )
    development = _group_observations(development_rows, contract)
    fitted = _fit_shared_matrix(
        [value for source in sorted(development) for value in development[source]],
        contract,
    )
    permutation = np.asarray([1, 2, 0])
    cyclic = np.ascontiguousarray(fitted[np.ix_(permutation, permutation)])
    independent = np.eye(3, dtype=np.float64)
    model_freeze = {
        "fitted_matrix": fitted.tolist(),
        "cyclic_control_matrix": cyclic.tolist(),
        "independent_control_matrix": independent.tolist(),
        "development_sources": sorted(development),
        "development_patch_count": sum(len(value) for value in development.values()),
    }
    model_freeze["model_freeze_id"] = sha256_bytes(canonical_json(model_freeze))

    selected = _enumerate_confirmation(contract)
    acquired = _acquire_confirmation(contract, selected, scratch)
    confirmation = _group_observations(acquired, contract)
    source_rows: list[dict[str, Any]] = []
    for source in sorted(confirmation):
        observed = np.mean(
            np.asarray([_correlation_matrix(value) for value in confirmation[source]]),
            axis=0,
        )
        candidate_error = _matrix_error(observed, fitted)
        independent_error = _matrix_error(observed, independent)
        cyclic_error = _matrix_error(observed, cyclic)
        source_rows.append(
            {
                "source": source,
                "patch_count": len(confirmation[source]),
                "observed_matrix": observed.tolist(),
                "candidate_error": candidate_error,
                "independent_error": independent_error,
                "cyclic_control_error": cyclic_error,
                "improvement_over_independent": 1.0 - candidate_error / independent_error,
                "improvement_over_cyclic_control": 1.0 - candidate_error / cyclic_error,
            }
        )
    improvements = np.asarray(
        [row["improvement_over_independent"] for row in source_rows], dtype=np.float64
    )
    cyclic_improvements = np.asarray(
        [row["improvement_over_cyclic_control"] for row in source_rows],
        dtype=np.float64,
    )
    candidate_ratios = np.asarray(
        [row["candidate_error"] / row["independent_error"] for row in source_rows],
        dtype=np.float64,
    )
    confirmation_mean = np.mean(
        np.asarray([row["observed_matrix"] for row in source_rows]), axis=0
    )
    development_confirmation_rmse = _matrix_error(confirmation_mean, fitted)
    gates = contract["metrics"]
    gate_results = {
        "all_sources_beat_independent": int(np.count_nonzero(improvements > 0.0))
        >= int(gates["minimum_sources_beating_independent"]),
        "median_improvement_over_independent": float(np.median(improvements))
        >= float(gates["minimum_median_improvement_over_independent"]),
        "worst_error_ratio_to_independent": float(np.max(candidate_ratios))
        <= float(gates["maximum_worst_error_ratio_to_independent"]),
        "all_sources_beat_cyclic_control": int(
            np.count_nonzero(cyclic_improvements > 0.0)
        )
        >= int(gates["minimum_sources_beating_cyclic_control"]),
        "median_improvement_over_cyclic_control": float(
            np.median(cyclic_improvements)
        )
        >= float(gates["minimum_median_improvement_over_cyclic_control"]),
        "development_confirmation_matrix_rmse": development_confirmation_rmse
        <= float(gates["maximum_development_to_confirmation_matrix_rmse"]),
    }
    passed = all(gate_results.values())
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_path": contract_path.relative_to(root).as_posix(),
        "contract_sha256": sha256_file(contract_path),
        "model_freeze": model_freeze,
        "confirmation_files": [
            {key: value for key, value in row.items() if key != "path"}
            for row in acquired
        ],
        "source_results": source_rows,
        "summary": {
            "sources_beating_independent": int(np.count_nonzero(improvements > 0.0)),
            "median_improvement_over_independent": float(np.median(improvements)),
            "worst_error_ratio_to_independent": float(np.max(candidate_ratios)),
            "sources_beating_cyclic_control": int(
                np.count_nonzero(cyclic_improvements > 0.0)
            ),
            "median_improvement_over_cyclic_control": float(
                np.median(cyclic_improvements)
            ),
            "development_confirmation_matrix_rmse": development_confirmation_rmse,
        },
        "gates": gate_results,
        "decision": (
            "PASS_RETAIN_GENERIC_CROSS_COMPONENT_GRAIN_CORRELATION"
            if passed
            else "FAIL_CLOSED_CROSS_COMPONENT_GRAIN_CORRELATION_TRANSFER"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "CrossComponentGrainCorrelationError",
    "_correlation_matrix",
    "_fit_shared_matrix",
    "_matrix_error",
    "_select_normalized_residuals",
    "canonical_json",
    "evaluate_cross_component_grain_correlation",
    "load_contract",
]
