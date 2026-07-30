"""Integrity and ordered-pair audit for the bounded FilmMatch chart source.

The audit establishes file integrity and an ordered same-scene mapping.  It
does not identify either lane's colour operator or promote the source beyond
its frozen internal-research claim ceiling.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


class FilmMatchSourceAuditError(ValueError):
    """Raised when the frozen source, TIFF contract, or mapping gate fails."""


_SEQUENCE = re.compile(r"_1\.(\d+)\.1\.tif$", re.IGNORECASE)
_ICC_PROFILE_TAG = 34675


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sequence_number(name: str) -> int:
    match = _SEQUENCE.search(name)
    if match is None:
        raise FilmMatchSourceAuditError(f"unexpected chart filename: {name}")
    return int(match.group(1))


def _inspect_tiff(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        with tifffile.TiffFile(path) as source:
            if len(source.pages) != 1:
                raise FilmMatchSourceAuditError(f"multi-page TIFF: {path.name}")
            page = source.pages[0]
            shape = tuple(int(value) for value in page.shape)
            dtype = str(page.dtype)
            has_icc = _ICC_PROFILE_TAG in page.tags
    except (OSError, tifffile.TiffFileError) as exc:
        raise FilmMatchSourceAuditError(f"TIFF decode failed: {path.name}") from exc
    if shape != tuple(expected["shape"]) or dtype != str(expected["dtype"]):
        raise FilmMatchSourceAuditError(
            f"TIFF shape/dtype drift: {path.name}: {shape}/{dtype}"
        )
    if has_icc != bool(expected["embedded_icc"]):
        raise FilmMatchSourceAuditError(f"TIFF ICC state drift: {path.name}")
    return {
        "shape": list(shape),
        "dtype": dtype,
        "embedded_icc": has_icc,
        "pages": 1,
    }


def validate_download_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    acquisition = config["acquisition"]
    expected_count = int(acquisition["expected_total_files"])
    expected_bytes = int(acquisition["expected_total_bytes"])
    rows = list(manifest.get("files", []))
    if (
        manifest.get("schema_version")
        != "u5-r2aw0-filmmatch-download-manifest-v1"
        or manifest.get("experiment_id") != config["experiment_id"]
        or int(manifest.get("file_count", -1)) != expected_count
        or int(manifest.get("bytes", -1)) != expected_bytes
        or len(rows) != expected_count
    ):
        raise FilmMatchSourceAuditError("download manifest identity/count drift")

    expected_folders = {
        str(folder["lane"]): folder for folder in acquisition["folders"]
    }
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    seen_file_ids: set[str] = set()
    total_bytes = 0
    validated: list[dict[str, Any]] = []
    for row in rows:
        lane = str(row.get("lane", ""))
        if lane not in expected_folders:
            raise FilmMatchSourceAuditError(f"unknown manifest lane: {lane}")
        relative = Path(str(row.get("relative_path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != (lane,)
        ):
            raise FilmMatchSourceAuditError("unsafe or cross-lane manifest path")
        normalized = relative.as_posix()
        digest = str(row.get("sha256", ""))
        file_id = str(row.get("file_id", ""))
        if (
            normalized in seen_paths
            or digest in seen_hashes
            or not file_id
            or file_id in seen_file_ids
        ):
            raise FilmMatchSourceAuditError(
                "duplicate path, exact file hash, or remote file identity"
            )
        path = root / relative
        if not path.is_file():
            raise FilmMatchSourceAuditError(f"manifest file missing: {normalized}")
        size = int(row.get("bytes", -1))
        folder = expected_folders[lane]
        if "expected_file_bytes" in folder:
            expected_size = int(folder["expected_file_bytes"])
        elif path.suffix.lower() in {".tif", ".tiff"}:
            expected_size = int(folder["expected_tiff_file_bytes"])
        else:
            drx_sizes = list(map(int, folder["expected_drx_file_bytes"]))
            validation_index = 0 if str(row["name"]).endswith("1.19.1.drx") else 1
            expected_size = sorted(drx_sizes)[validation_index]
        if size != expected_size:
            raise FilmMatchSourceAuditError(f"file size contract drift: {normalized}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise FilmMatchSourceAuditError(f"file size/hash drift: {normalized}")
        seen_paths.add(normalized)
        seen_hashes.add(digest)
        seen_file_ids.add(file_id)
        total_bytes += size
        validated.append({**dict(row), "path": path})
    if total_bytes != expected_bytes:
        raise FilmMatchSourceAuditError("manifest byte total drift")

    for lane, folder in expected_folders.items():
        lane_rows = [row for row in validated if row["lane"] == lane]
        if len(lane_rows) != int(folder["expected_files"]):
            raise FilmMatchSourceAuditError(f"lane count drift: {lane}")
        if "expected_sequence" in folder:
            start, end = map(int, folder["expected_sequence"])
            observed = sorted(sequence_number(str(row["name"])) for row in lane_rows)
            if observed != list(range(start, end + 1)):
                raise FilmMatchSourceAuditError(f"lane sequence drift: {lane}")
    return validated


def extract_mapping_feature(
    path: Path,
    *,
    sample_grid: Sequence[int],
    quantiles: Sequence[float],
) -> np.ndarray:
    """Return deterministic sparse code-value statistics without full decode."""

    rows, columns = map(int, sample_grid)
    try:
        pixels = tifffile.memmap(path)
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise FilmMatchSourceAuditError(
            f"TIFF is not memory-mappable: {path.name}"
        ) from exc
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise FilmMatchSourceAuditError(f"unexpected feature image shape: {path.name}")
    row_index = np.linspace(0, pixels.shape[0] - 1, rows, dtype=np.int64)
    column_index = np.linspace(0, pixels.shape[1] - 1, columns, dtype=np.int64)
    sampled = np.asarray(pixels[np.ix_(row_index, column_index)], dtype=np.float64)
    sampled /= float(np.iinfo(pixels.dtype).max)
    feature = np.concatenate(
        [
            np.quantile(sampled, np.asarray(quantiles), axis=(0, 1)).reshape(-1),
            sampled.mean(axis=(0, 1)),
            sampled.std(axis=(0, 1)),
        ]
    )
    if not np.all(np.isfinite(feature)):
        raise FilmMatchSourceAuditError(f"non-finite feature: {path.name}")
    return feature


def ordered_mapping_audit(
    film_features: np.ndarray,
    digital_features: np.ndarray,
    *,
    epsilon: float,
    permutations: int,
    permutation_seed: int,
) -> dict[str, Any]:
    film = np.asarray(film_features, dtype=np.float64)
    digital = np.asarray(digital_features, dtype=np.float64)
    if (
        film.ndim != 2
        or film.shape != digital.shape
        or film.shape[0] < 3
        or not np.all(np.isfinite(film))
        or not np.all(np.isfinite(digital))
    ):
        raise FilmMatchSourceAuditError("mapping features must be finite matching NxD")
    film = (film - film.mean(axis=0)) / (film.std(axis=0) + epsilon)
    digital = (digital - digital.mean(axis=0)) / (
        digital.std(axis=0) + epsilon
    )
    costs = np.mean((film[:, None, :] - digital[None, :, :]) ** 2, axis=2)
    population = film.shape[0]
    indexes = np.arange(population)
    circular = np.asarray(
        [
            np.mean(costs[indexes, (indexes + shift) % population])
            for shift in range(population)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(permutation_seed)
    null = np.asarray(
        [
            np.mean(costs[indexes, rng.permutation(population)])
            for _ in range(permutations)
        ],
        dtype=np.float64,
    )
    observed = float(circular[0])
    null_median = float(np.median(null))
    return {
        "pairs": population,
        "feature_dimensions": int(film.shape[1]),
        "ordered_cost": observed,
        "best_circular_shift": int(np.argmin(circular)),
        "next_best_circular_cost": float(np.min(circular[1:])),
        "permutation_median_cost": null_median,
        "cost_to_permutation_median_ratio": observed / null_median,
        "permutation_p": float((1 + np.count_nonzero(null <= observed)) / (1 + permutations)),
    }


def audit_source(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = validate_download_manifest(manifest, root=root, config=config)
    audit = config["integrity_audit"]
    expected_tiff = audit["expected_tiff"]
    tiff_rows = [row for row in rows if Path(str(row["name"])).suffix.lower() == ".tif"]
    inspections: dict[str, dict[str, Any]] = {}
    for row in tiff_rows:
        lane = str(row["lane"])
        if lane not in expected_tiff and lane != "validation":
            raise FilmMatchSourceAuditError(f"missing TIFF contract: {lane}")
        expected = (
            expected_tiff[lane]
            if lane != "validation"
            else expected_tiff["sony_reflective"]
        )
        inspections[str(row["relative_path"])] = _inspect_tiff(row["path"], expected)

    feature_config = audit["mapping_feature"]
    gate = audit["mapping_gate"]
    mapping_rows: list[dict[str, Any]] = []
    for lane in audit["pair_lanes"]:
        film_rows = sorted(
            (row for row in rows if row["lane"] == lane["film_lane"]),
            key=lambda row: sequence_number(str(row["name"])),
        )
        digital_rows = sorted(
            (row for row in rows if row["lane"] == lane["digital_lane"]),
            key=lambda row: sequence_number(str(row["name"])),
        )
        expected_pairs = int(lane["expected_pairs"])
        if len(film_rows) != expected_pairs or len(digital_rows) != expected_pairs:
            raise FilmMatchSourceAuditError(f"pair population drift: {lane['pair_lane']}")
        film_features = np.stack(
            [
                extract_mapping_feature(
                    row["path"],
                    sample_grid=feature_config["sample_grid"],
                    quantiles=feature_config["quantiles"],
                )
                for row in film_rows
            ]
        )
        digital_features = np.stack(
            [
                extract_mapping_feature(
                    row["path"],
                    sample_grid=feature_config["sample_grid"],
                    quantiles=feature_config["quantiles"],
                )
                for row in digital_rows
            ]
        )
        result = ordered_mapping_audit(
            film_features,
            digital_features,
            epsilon=float(feature_config["domain_standardization_epsilon"]),
            permutations=int(gate["permutations"]),
            permutation_seed=int(gate["permutation_seed"]),
        )
        result["pair_lane"] = lane["pair_lane"]
        result["pair_sequence"] = [
            {
                "pair_index": index,
                "film_relative_path": film["relative_path"],
                "film_sha256": film["sha256"],
                "digital_relative_path": digital["relative_path"],
                "digital_sha256": digital["sha256"],
            }
            for index, (film, digital) in enumerate(
                zip(film_rows, digital_rows, strict=True)
            )
        ]
        result["passed"] = bool(
            result["best_circular_shift"] == 0
            and result["permutation_p"] <= float(gate["maximum_permutation_p"])
            and result["cost_to_permutation_median_ratio"]
            <= float(gate["maximum_cost_to_permutation_median_ratio"])
        )
        if not result["passed"]:
            raise FilmMatchSourceAuditError(
                f"ordered mapping gate failed: {lane['pair_lane']}"
            )
        mapping_rows.append(result)

    validation_rows = sorted(
        str(row["relative_path"]) for row in rows if row["lane"] == "validation"
    )
    report = {
        "schema_version": audit["schema_version"],
        "experiment_id": config["experiment_id"],
        "download_manifest_canonical_sha256": canonical_sha256(manifest),
        "audit_contract_canonical_sha256": canonical_sha256(audit),
        "source_integrity_passed": True,
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "exact_duplicate_hashes": 0,
        "tiff_count": len(tiff_rows),
        "tiff_contract_counts": {
            lane: sum(row["lane"] == lane for row in tiff_rows)
            for lane in sorted(expected_tiff)
        },
        "ordered_mapping_passed": all(row["passed"] for row in mapping_rows),
        "pair_lanes": mapping_rows,
        "validation_fit_forbidden": True,
        "validation_files": validation_rows,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report
