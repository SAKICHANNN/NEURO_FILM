"""Access-locked helpers for the one-shot FilmSet final evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .baselines import LAB_STATS_SCHEMA, SLICED_SCHEMA, LabMeanStdOperator, SlicedTransportOperator
from .manifests import FILMSET_MANIFEST_SCHEMA, FilmSetManifestError
from .operators import OPERATOR_SCHEMA, AffineColorOperator
from .splines import L2_OPERATOR_SCHEMA, AffineMonotoneSplineOperator


FINAL_ROLE = "final_628_lockbox"
FINAL_ACCESS = "final_evaluator_only"


def operator_from_frozen_bundle(payload: dict[str, Any]) -> Any:
    schema = payload.get("schema")
    if schema == OPERATOR_SCHEMA:
        return AffineColorOperator.from_dict(payload)
    if schema == LAB_STATS_SCHEMA:
        return LabMeanStdOperator.from_dict(payload)
    if schema == SLICED_SCHEMA:
        return SlicedTransportOperator.from_dict(payload)
    if schema == L2_OPERATOR_SCHEMA:
        return AffineMonotoneSplineOperator.from_dict(payload)
    raise ValueError(f"unsupported frozen operator schema: {schema!r}")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_final_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_identities: int,
    expected_rows: int,
    domains: Iterable[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Parse the final manifest only through its strict evaluator contract."""
    observed = sha256_file(path)
    if observed.lower() != expected_sha256.lower():
        raise FilmSetManifestError(f"final manifest hash mismatch: {observed}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema_version") != FILMSET_MANIFEST_SCHEMA:
            raise FilmSetManifestError(f"final manifest line {line_number}: unsupported schema")
        if row.get("research_pool") != FINAL_ROLE:
            raise FilmSetManifestError(f"final manifest line {line_number}: unexpected role")
        if row.get("payload_access") != FINAL_ACCESS:
            raise FilmSetManifestError(f"final manifest line {line_number}: unexpected access")
        rows.append(row)
    if len(rows) != expected_rows:
        raise FilmSetManifestError(f"final manifest row count {len(rows)} != {expected_rows}")
    required = {"input", *map(str, domains)}
    by_content: dict[str, dict[str, dict[str, Any]]] = {}
    clusters: dict[str, str] = {}
    for row in rows:
        content_id = str(row["content_id"])
        domain = str(row["domain"])
        if domain not in required:
            raise FilmSetManifestError(f"final manifest undeclared domain: {domain}")
        if domain in by_content.setdefault(content_id, {}):
            raise FilmSetManifestError(f"final manifest duplicate role: {content_id}/{domain}")
        by_content[content_id][domain] = row
        cluster = str(row["duplicate_cluster_id"])
        prior = clusters.setdefault(content_id, cluster)
        if prior != cluster:
            raise FilmSetManifestError(f"final identity spans clusters: {content_id}")
    if len(by_content) != expected_identities:
        raise FilmSetManifestError(
            f"final identity count {len(by_content)} != {expected_identities}"
        )
    for content_id, domain_rows in by_content.items():
        if set(domain_rows) != required:
            raise FilmSetManifestError(f"final identity incomplete: {content_id}")
    return dict(sorted(by_content.items()))


def verify_payloads(
    by_content: dict[str, dict[str, dict[str, Any]]], dataset_root: Path
) -> None:
    """Verify every payload before any image decoder is allowed to run."""
    root = dataset_root.resolve()
    for content_id, domain_rows in by_content.items():
        for domain, row in sorted(domain_rows.items()):
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise FilmSetManifestError("manifest path escapes the FilmSet root")
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise FilmSetManifestError("manifest path escapes the FilmSet root") from exc
            observed = sha256_file(path)
            if observed.lower() != str(row["sha256"]).lower():
                raise FilmSetManifestError(
                    f"payload hash mismatch: {content_id}/{domain}/{relative.as_posix()}"
                )


def cluster_bootstrap_improvement(
    basic: Iterable[float],
    primary: Iterable[float],
    cluster_ids: Iterable[str],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    basic_values = np.asarray(tuple(basic), dtype=np.float64)
    primary_values = np.asarray(tuple(primary), dtype=np.float64)
    clusters = np.asarray(tuple(cluster_ids), dtype=object)
    if basic_values.shape != primary_values.shape or basic_values.shape != clusters.shape:
        raise ValueError("bootstrap inputs must have equal one-dimensional shapes")
    if basic_values.ndim != 1 or len(basic_values) < 2 or resamples < 100:
        raise ValueError("bootstrap requires at least two rows and 100 resamples")
    unique = np.unique(clusters)
    grouped = np.asarray(
        [np.mean((basic_values - primary_values)[clusters == value]) for value in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(resamples, len(grouped)))
    sampled = np.mean(grouped[draws], axis=1)
    return {
        "improvement_mean": float(np.mean(grouped)),
        "ci95_low": float(np.percentile(sampled, 2.5)),
        "ci95_high": float(np.percentile(sampled, 97.5)),
        "clusters": int(len(unique)),
        "resamples": int(resamples),
    }


def flatten_review_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "raw_excursion_max": float(metrics["raw_excursion_max"]),
        "new_display_clip_pixel_fraction_vs_target": float(
            metrics["new_display_clip_pixel_fraction_vs_target"]
        ),
        "mean_delta_e00_to_target": float(metrics["mean_delta_e00_to_target"]),
        "red_cyan_boundary_occupancy": float(metrics["red_cyan_boundary_occupancy"]),
        "speckle_candidate_percent": float(
            metrics["chroma_speckle"]["speckle_candidate_percent"]
        ),
    }


def select_review_cases(
    rows: list[dict[str, Any]],
    *,
    axes: Iterable[str],
    composite_count: int,
    per_axis_count: int,
) -> dict[str, Any]:
    """Select fixed percentile-rank composite worst cases and axis extremes."""
    if not rows:
        raise ValueError("cannot select review cases from no rows")
    axis_names = tuple(axes)
    values = np.asarray(
        [[flatten_review_metrics(row["metrics"])[axis] for axis in axis_names] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("review metrics contain non-finite values")
    ranks = np.empty_like(values)
    for column in range(values.shape[1]):
        order = np.lexsort((np.arange(len(rows)), values[:, column]))
        ranks[order, column] = np.linspace(0.0, 1.0, len(rows))
    composite = np.mean(ranks, axis=1)
    stable_ids = np.asarray([str(row["content_id"]) for row in rows], dtype=object)
    composite_order = np.lexsort((stable_ids, -composite))
    by_axis: dict[str, list[str]] = {}
    for column, axis in enumerate(axis_names):
        order = np.lexsort((stable_ids, -values[:, column]))
        by_axis[axis] = [str(rows[index]["content_id"]) for index in order[:per_axis_count]]
    composite_ids = [
        str(rows[index]["content_id"]) for index in composite_order[:composite_count]
    ]
    union = list(dict.fromkeys(composite_ids + [item for axis in axis_names for item in by_axis[axis]]))
    return {"composite": composite_ids, "by_axis": by_axis, "union": union}
