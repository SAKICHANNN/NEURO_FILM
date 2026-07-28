"""Build the research-only FiveK neutral-photo moment prior deterministically."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Callable, Mapping
from urllib.request import urlopen

import numpy as np
import tifffile

from ..canonical import canonical_sha256
from ..contracts import ReferenceMatchContractError
from ..strict_json import strict_json_loads
from .empirical_prior import (
    EMPIRICAL_PRIOR_SCHEMA_ID,
    compute_empirical_prior_id,
    validate_empirical_prior_payload,
)


FIVEK_DATASET_ID = "mit-adobe-fivek.freeze-v1.raw-default-srgb16"
FIVEK_MANIFEST_RELATIVE_PATH = (
    "outputs/fivek_auto_optimize/freeze_v1/manifest.csv"
)
FIVEK_SUMMARY_RELATIVE_PATH = (
    "outputs/fivek_auto_optimize/freeze_v1/summary.json"
)
FIVEK_MANIFEST_SHA256 = (
    "f820faf3dc872597b5259181040c07a2fc044c22423e05bc9db05096d88fc174"
)
FIVEK_SUMMARY_SHA256 = (
    "45032415c91f2f5a61e3415d2ce4c99ff27696e4fb2d0add4d385b92d4fc522b"
)
FIVEK_RIGHTS_URLS = {
    "adobe": {
        "licence": (
            "https://data.csail.mit.edu/graphics/fivek/legal/"
            "LicenseAdobe.txt"
        ),
        "file_list": (
            "https://data.csail.mit.edu/graphics/fivek/legal/filesAdobe.txt"
        ),
        "licence_sha256": (
            "8121ecfcf743d850d6b6dbd0744cfb517c4cec189387aa010bf378077de84fcf"
        ),
        "file_list_sha256": (
            "9f4f262b25522d93f97a5b40f48f42de357ae0b14ef7098a576c912c546e8383"
        ),
    },
    "adobe_mit": {
        "licence": (
            "https://data.csail.mit.edu/graphics/fivek/legal/"
            "LicenseAdobeMIT.txt"
        ),
        "file_list": (
            "https://data.csail.mit.edu/graphics/fivek/legal/"
            "filesAdobeMIT.txt"
        ),
        "licence_sha256": (
            "69f73c481140c40d3a02a15a739da76073de35646108a70bb1dfffadb182af3d"
        ),
        "file_list_sha256": (
            "c4cc6ef8cc9e519466ac130b977cd4292ffe9405f94d04227d0e2aee38aa5475"
        ),
    },
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def _verified_rights_lists(
    fetch: Callable[[str], bytes],
) -> dict[str, frozenset[str]]:
    file_lists: dict[str, frozenset[str]] = {}
    for group, contract in FIVEK_RIGHTS_URLS.items():
        licence = fetch(str(contract["licence"]))
        file_list = fetch(str(contract["file_list"]))
        if hashlib.sha256(licence).hexdigest() != contract["licence_sha256"]:
            raise ReferenceMatchContractError(
                f"FiveK {group} licence snapshot hash mismatch"
            )
        if (
            hashlib.sha256(file_list).hexdigest()
            != contract["file_list_sha256"]
        ):
            raise ReferenceMatchContractError(
                f"FiveK {group} file-list snapshot hash mismatch"
            )
        text = licence.decode("utf-8-sig")
        if (
            "RESEARCH LICENSE" not in text
            or "solely for your own research purposes" not in text
            or "commercial advantage or monetary compensation" not in text
        ):
            raise ReferenceMatchContractError(
                f"FiveK {group} licence no longer matches research-only scope"
            )
        names = frozenset(
            line.strip()
            for line in file_list.decode("utf-8-sig").splitlines()
            if line.strip()
        )
        if len(names) < 1000:
            raise ReferenceMatchContractError(
                f"FiveK {group} file-list snapshot is unexpectedly small"
            )
        file_lists[group] = names
    return file_lists


def _bounded_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ReferenceMatchContractError(
            "FiveK manifest contains a path outside the dataset root"
        ) from exc
    return candidate


def _linearize_srgb16(array: np.ndarray) -> np.ndarray:
    encoded = np.asarray(array, dtype=np.float64) / 65535.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def build_fivek_empirical_prior_payload(
    dataset_root: Path,
    *,
    fetch: Callable[[str], bytes] = _download,
) -> dict[str, object]:
    """Audit the frozen FiveK ingress and return a canonical moment artifact."""

    root = dataset_root.resolve()
    manifest_path = _bounded_path(root, FIVEK_MANIFEST_RELATIVE_PATH)
    summary_path = _bounded_path(root, FIVEK_SUMMARY_RELATIVE_PATH)
    if _sha256_path(manifest_path) != FIVEK_MANIFEST_SHA256:
        raise ReferenceMatchContractError(
            "FiveK neutral-prior manifest hash mismatch"
        )
    if _sha256_path(summary_path) != FIVEK_SUMMARY_SHA256:
        raise ReferenceMatchContractError(
            "FiveK neutral-prior summary hash mismatch"
        )
    summary = strict_json_loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("hp_count") != 128
        or summary.get("missing") != []
        or summary.get("raw_color_management")
        != (
            "rawpy/LibRaw generic camera decode via src.preprocess; "
            "display sRGB transfer encoded to uint16 TIFF."
        )
    ):
        raise ReferenceMatchContractError(
            "FiveK neutral-prior summary contract mismatch"
        )

    rights_lists = _verified_rights_lists(fetch)
    rows = list(
        csv.DictReader(
            io.StringIO(manifest_path.read_text(encoding="utf-8"))
        )
    )
    if len(rows) != 128:
        raise ReferenceMatchContractError(
            "FiveK neutral-prior manifest must contain exactly 128 rows"
        )

    image_records: list[dict[str, object]] = []
    image_means: list[np.ndarray] = []
    image_second_moments: list[np.ndarray] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    total_pixels = 0
    for row in rows:
        image_id = str(row["id"])
        source_name = str(row["source_name"])
        relative_path = str(row["raw_default_srgb16"]).replace("\\", "/")
        if (
            not image_id
            or image_id in seen_ids
            or not source_name
            or not relative_path
            or relative_path in seen_paths
        ):
            raise ReferenceMatchContractError(
                "FiveK neutral-prior manifest identities are not unique"
            )
        matching_groups = [
            group
            for group, listing in rights_lists.items()
            if source_name in listing
        ]
        if len(matching_groups) != 1:
            raise ReferenceMatchContractError(
                f"FiveK source {source_name} has ambiguous licence coverage"
            )
        image_path = _bounded_path(root, relative_path)
        image_sha256 = _sha256_path(image_path)
        if image_sha256 in seen_hashes:
            raise ReferenceMatchContractError(
                "FiveK neutral-prior input contains a byte duplicate"
            )
        with tifffile.TiffFile(image_path) as tif:
            if len(tif.pages) != 1:
                raise ReferenceMatchContractError(
                    "FiveK neutral-prior TIFF must have exactly one page"
                )
            page = tif.pages[0]
            array = page.asarray()
            orientation_tag = page.tags.get("Orientation")
            orientation = (
                int(orientation_tag.value)
                if orientation_tag is not None
                else 1
            )
        if (
            array.dtype != np.uint16
            or array.ndim != 3
            or array.shape[2] != 3
            or orientation != 1
        ):
            raise ReferenceMatchContractError(
                "FiveK neutral-prior input must be orientation-1 uint16 RGB"
            )
        height, width, _ = array.shape
        pixels = _linearize_srgb16(array).reshape(-1, 3)
        if not np.isfinite(pixels).all():
            raise ReferenceMatchContractError(
                "FiveK neutral-prior input contains non-finite pixels"
            )
        pixel_count = int(len(pixels))
        image_mean = pixels.mean(axis=0, dtype=np.float64)
        image_second = (pixels.T @ pixels) / float(pixel_count)
        image_means.append(image_mean)
        image_second_moments.append(image_second)
        total_pixels += pixel_count
        image_records.append(
            {
                "id": image_id,
                "relative_path": relative_path,
                "sha256": image_sha256,
                "width": int(width),
                "height": int(height),
                "pixel_count": pixel_count,
                "licence_group": matching_groups[0],
            }
        )
        seen_ids.add(image_id)
        seen_paths.add(relative_path)
        seen_hashes.add(image_sha256)

    mean = np.mean(np.stack(image_means), axis=0, dtype=np.float64)
    second = np.mean(
        np.stack(image_second_moments),
        axis=0,
        dtype=np.float64,
    )
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
        raise ReferenceMatchContractError(
            "FiveK neutral-prior covariance is not positive definite"
        )
    payload: dict[str, object] = {
        "schema_id": EMPIRICAL_PRIOR_SCHEMA_ID,
        "prior_id": "",
        "dataset": {
            "dataset_id": FIVEK_DATASET_ID,
            "manifest_relative_path": FIVEK_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": FIVEK_MANIFEST_SHA256,
            "summary_relative_path": FIVEK_SUMMARY_RELATIVE_PATH,
            "summary_sha256": FIVEK_SUMMARY_SHA256,
        },
        "rights": {
            "scope": "research-only",
            "product_use_allowed": False,
            "commercial_use_allowed": False,
            "licence_snapshot_sha256": {
                "adobe": FIVEK_RIGHTS_URLS["adobe"]["licence_sha256"],
                "adobe_mit": FIVEK_RIGHTS_URLS["adobe_mit"][
                    "licence_sha256"
                ],
            },
            "file_list_snapshot_sha256": {
                "adobe": FIVEK_RIGHTS_URLS["adobe"]["file_list_sha256"],
                "adobe_mit": FIVEK_RIGHTS_URLS["adobe_mit"][
                    "file_list_sha256"
                ],
            },
        },
        "ingest": {
            "selected_manifest_column": "raw_default_srgb16",
            "encoded_state": "display-srgb16",
            "working_state": "display-linear-linear-srgb",
            "aggregation": "equal-image-pixel-mixture-v1",
            "expert_or_target_pixels_read": False,
        },
        "statistics": {
            "mean_linear_rgb": mean.tolist(),
            "population_covariance_linear_rgb": covariance.tolist(),
            "source_image_count": len(image_records),
            "source_pixel_count": total_pixels,
        },
        "source_images": image_records,
        "claim_ceiling": (
            "research-only neutral-photo moment evidence; no product, "
            "commercial, photographic-preference, film-stock, calibration "
            "or authenticity claim"
        ),
    }
    payload["prior_id"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "prior_id"}
    )
    validate_empirical_prior_payload(payload)
    if payload["prior_id"] != compute_empirical_prior_id(payload):
        raise AssertionError("empirical prior canonical identity drift")
    return payload


__all__ = [
    "FIVEK_DATASET_ID",
    "FIVEK_MANIFEST_RELATIVE_PATH",
    "FIVEK_MANIFEST_SHA256",
    "FIVEK_RIGHTS_URLS",
    "FIVEK_SUMMARY_RELATIVE_PATH",
    "FIVEK_SUMMARY_SHA256",
    "build_fivek_empirical_prior_payload",
]
