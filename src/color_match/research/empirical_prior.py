"""Strict research-only empirical neutral-photography prior artifacts.

The artifact contains aggregate RGB moments and source-file identities, never
source pixels.  It is intentionally in the research package: a passing
experiment does not override the source-data licence or open product use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..canonical import canonical_sha256
from ..contracts import ReferenceMatchContractError


EMPIRICAL_PRIOR_SCHEMA_ID = "neuro-film.empirical-neutral-prior.v1"
EMPIRICAL_PRIOR_MODE = "empirical-neutral-photo-v1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class EmpiricalNeutralPrior:
    """Validated aggregate moments for one frozen neutral-photo population."""

    prior_id: str
    dataset_id: str
    mean: np.ndarray
    covariance: np.ndarray
    source_image_count: int
    source_pixel_count: int
    artifact: Mapping[str, Any]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def compute_empirical_prior_id(payload: Mapping[str, Any]) -> str:
    """Compute the canonical identity while excluding the identity field."""

    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "empirical prior payload must be a mapping"
        )
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "prior_id"}
    )


def validate_empirical_prior_payload(
    payload: Mapping[str, Any],
) -> EmpiricalNeutralPrior:
    """Validate all provenance, rights and numerical fields fail closed."""

    if not isinstance(payload, Mapping):
        raise ReferenceMatchContractError(
            "empirical prior payload must be a mapping"
        )
    expected_top = {
        "schema_id",
        "prior_id",
        "dataset",
        "rights",
        "ingest",
        "statistics",
        "source_images",
        "claim_ceiling",
    }
    if set(payload) != expected_top:
        raise ReferenceMatchContractError(
            "empirical prior top-level keys mismatch"
        )
    if payload["schema_id"] != EMPIRICAL_PRIOR_SCHEMA_ID:
        raise ReferenceMatchContractError(
            "unsupported empirical prior schema"
        )
    prior_id = payload["prior_id"]
    if not _is_sha256(prior_id) or prior_id != compute_empirical_prior_id(
        payload
    ):
        raise ReferenceMatchContractError(
            "empirical prior_id does not match canonical payload"
        )

    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "dataset_id",
        "manifest_relative_path",
        "manifest_sha256",
        "summary_relative_path",
        "summary_sha256",
    }:
        raise ReferenceMatchContractError(
            "empirical prior dataset provenance mismatch"
        )
    dataset_id = dataset["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ReferenceMatchContractError(
            "empirical prior dataset_id must be non-empty"
        )
    for key in ("manifest_sha256", "summary_sha256"):
        if not _is_sha256(dataset[key]):
            raise ReferenceMatchContractError(
                f"empirical prior {key} must be canonical SHA-256"
            )
    for key in ("manifest_relative_path", "summary_relative_path"):
        relative = Path(str(dataset[key]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) in {"", "."}
        ):
            raise ReferenceMatchContractError(
                f"empirical prior {key} must be a bounded relative path"
            )

    rights = payload["rights"]
    if not isinstance(rights, Mapping) or set(rights) != {
        "scope",
        "product_use_allowed",
        "commercial_use_allowed",
        "licence_snapshot_sha256",
        "file_list_snapshot_sha256",
    }:
        raise ReferenceMatchContractError(
            "empirical prior rights fields mismatch"
        )
    if (
        rights["scope"] != "research-only"
        or rights["product_use_allowed"] is not False
        or rights["commercial_use_allowed"] is not False
    ):
        raise ReferenceMatchContractError(
            "empirical prior must retain its research-only rights ceiling"
        )
    licence_hashes = rights["licence_snapshot_sha256"]
    file_list_hashes = rights["file_list_snapshot_sha256"]
    if (
        not isinstance(licence_hashes, Mapping)
        or set(licence_hashes) != {"adobe", "adobe_mit"}
        or not all(_is_sha256(value) for value in licence_hashes.values())
        or not isinstance(file_list_hashes, Mapping)
        or set(file_list_hashes) != {"adobe", "adobe_mit"}
        or not all(_is_sha256(value) for value in file_list_hashes.values())
    ):
        raise ReferenceMatchContractError(
            "empirical prior licence snapshot identities mismatch"
        )

    ingest = payload["ingest"]
    if not isinstance(ingest, Mapping) or ingest != {
        "selected_manifest_column": "raw_default_srgb16",
        "encoded_state": "display-srgb16",
        "working_state": "display-linear-linear-srgb",
        "aggregation": "equal-image-pixel-mixture-v1",
        "expert_or_target_pixels_read": False,
    }:
        raise ReferenceMatchContractError(
            "empirical prior ingest contract mismatch"
        )

    statistics = payload["statistics"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "mean_linear_rgb",
        "population_covariance_linear_rgb",
        "source_image_count",
        "source_pixel_count",
    }:
        raise ReferenceMatchContractError(
            "empirical prior statistics fields mismatch"
        )
    image_count = statistics["source_image_count"]
    pixel_count = statistics["source_pixel_count"]
    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count < 4
        or isinstance(pixel_count, bool)
        or not isinstance(pixel_count, int)
        or pixel_count < image_count * 64
    ):
        raise ReferenceMatchContractError(
            "empirical prior source counts are invalid"
        )
    mean = np.asarray(statistics["mean_linear_rgb"], dtype=np.float64)
    covariance = np.asarray(
        statistics["population_covariance_linear_rgb"],
        dtype=np.float64,
    )
    if (
        mean.shape != (3,)
        or covariance.shape != (3, 3)
        or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()
        or np.any(mean < 0.0)
        or np.any(mean > 1.0)
        or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0)
        or float(np.linalg.eigvalsh(covariance).min()) <= 0.0
        or np.any(np.diag(covariance) > 0.250000000001)
    ):
        raise ReferenceMatchContractError(
            "empirical prior RGB moments are invalid"
        )

    source_images = payload["source_images"]
    if not isinstance(source_images, list) or len(source_images) != image_count:
        raise ReferenceMatchContractError(
            "empirical prior source-image inventory count mismatch"
        )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    summed_pixels = 0
    for record in source_images:
        if not isinstance(record, Mapping) or set(record) != {
            "id",
            "relative_path",
            "sha256",
            "width",
            "height",
            "pixel_count",
            "licence_group",
        }:
            raise ReferenceMatchContractError(
                "empirical prior source-image record fields mismatch"
            )
        image_id = record["id"]
        relative_path = record["relative_path"]
        sha256 = record["sha256"]
        width = record["width"]
        height = record["height"]
        pixels = record["pixel_count"]
        if (
            not isinstance(image_id, str)
            or not image_id
            or not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not _is_sha256(sha256)
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 8
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 8
            or isinstance(pixels, bool)
            or not isinstance(pixels, int)
            or pixels != width * height
            or record["licence_group"] not in {"adobe", "adobe_mit"}
        ):
            raise ReferenceMatchContractError(
                "empirical prior source-image record is invalid"
            )
        if (
            image_id in seen_ids
            or relative_path in seen_paths
            or sha256 in seen_hashes
        ):
            raise ReferenceMatchContractError(
                "empirical prior source-image identities must be unique"
            )
        seen_ids.add(image_id)
        seen_paths.add(relative_path)
        seen_hashes.add(sha256)
        summed_pixels += pixels
    if summed_pixels != pixel_count:
        raise ReferenceMatchContractError(
            "empirical prior source pixel count does not match inventory"
        )
    claim_ceiling = payload["claim_ceiling"]
    if (
        not isinstance(claim_ceiling, str)
        or "research" not in claim_ceiling.lower()
        or "product" not in claim_ceiling.lower()
    ):
        raise ReferenceMatchContractError(
            "empirical prior claim ceiling must state research/product boundary"
        )
    return EmpiricalNeutralPrior(
        prior_id=str(prior_id),
        dataset_id=dataset_id,
        mean=mean,
        covariance=covariance,
        source_image_count=image_count,
        source_pixel_count=pixel_count,
        artifact=payload,
    )


def load_empirical_neutral_prior(path: Path) -> EmpiricalNeutralPrior:
    """Load a UTF-8 JSON artifact and validate it before returning moments."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceMatchContractError(
            "could not load empirical neutral-prior artifact"
        ) from exc
    return validate_empirical_prior_payload(payload)


__all__ = [
    "EMPIRICAL_PRIOR_MODE",
    "EMPIRICAL_PRIOR_SCHEMA_ID",
    "EmpiricalNeutralPrior",
    "compute_empirical_prior_id",
    "load_empirical_neutral_prior",
    "validate_empirical_prior_payload",
]
