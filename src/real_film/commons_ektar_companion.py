"""Bounded same-author Commons companion discovery for controlled Ektar evidence."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.real_film.commons_stock_source import metadata_value


class CommonsEktarCompanionError(ValueError):
    """Raised when frozen source or candidate facts violate the contract."""


_FLICKR_PEOPLE_ID = re.compile(r"flickr\.com/people/([^/\"<]+)", re.IGNORECASE)
_FLICKR_PHOTOS_ID = re.compile(r"flickr\.com/photos/([^/\"<]+)", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_TITLE_STOP = {
    "file",
    "jpg",
    "jpeg",
    "png",
    "kodak",
    "ektar",
    "100",
    "film",
    "photo",
    "photograph",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "neuro-film.sf3-a3e-commons-ektar-companion-discovery-contract.v1":
        raise CommonsEktarCompanionError("unexpected contract schema")
    manifest = path.parents[1] / str(contract["input_selection_manifest"])
    if sha256_file(manifest) != contract["input_selection_manifest_sha256"]:
        raise CommonsEktarCompanionError("input selection manifest identity drift")
    return contract


def flickr_identity(row: Mapping[str, Any]) -> str:
    text = f"{row.get('author_raw_html', '')} {row.get('credit_raw_html', '')}"
    # A Commons Flickr import normally exposes the stable NSID in the author
    # ``people`` URL and a screen name in the credited ``photos`` URL.  Those
    # strings are aliases, not conflicting identities, so prefer the NSID and
    # use a photos owner only when no people URL exists.
    matches = _FLICKR_PEOPLE_ID.findall(text) or _FLICKR_PHOTOS_ID.findall(text)
    if not matches:
        return ""
    normalized = {value.strip().casefold() for value in matches if value.strip()}
    if len(normalized) != 1:
        raise CommonsEktarCompanionError("ambiguous Flickr identity")
    return next(iter(normalized))


def visible_text(value: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", value or "")).casefold().split())


def title_tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(visible_text(value))
        if token not in _TITLE_STOP and not token.isdecimal() and len(token) > 1
    }


def title_jaccard(left: str, right: str) -> float:
    a, b = title_tokens(left), title_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_capture_time(value: str) -> datetime | None:
    clean = visible_text(value).replace("t", " ").replace("z", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def normalize_search_page(page: Mapping[str, Any]) -> dict[str, Any]:
    infos = page.get("imageinfo")
    if not isinstance(infos, list) or len(infos) != 1:
        raise CommonsEktarCompanionError("candidate page requires one imageinfo record")
    info = infos[0]
    metadata = info.get("extmetadata") or {}
    required = ("url", "descriptionurl", "sha1", "width", "height", "size")
    if any(info.get(key) in (None, "") for key in required):
        raise CommonsEktarCompanionError("candidate imageinfo is incomplete")
    return {
        "page_id": int(page["pageid"]),
        "title": str(page["title"]),
        "file_page_url": str(info["descriptionurl"]),
        "original_url": str(info["url"]),
        "thumbnail_url": str(info.get("thumburl", "")),
        "api_sha1_base36": str(info["sha1"]),
        "byte_size": int(info["size"]),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "mime": str(info.get("mime", "")),
        "upload_timestamp": str(info.get("timestamp", "")),
        "camera_model": metadata_value(metadata, "Model"),
        "date_time_original": metadata_value(metadata, "DateTimeOriginal"),
        "description_raw_html": metadata_value(metadata, "ImageDescription"),
        "credit_raw_html": metadata_value(metadata, "Credit"),
        "author_raw_html": metadata_value(metadata, "Artist"),
        "license_short_name": metadata_value(metadata, "LicenseShortName"),
        "license_url": metadata_value(metadata, "LicenseUrl"),
        "usage_terms": metadata_value(metadata, "UsageTerms"),
    }


def metadata_candidate(
    source: Mapping[str, Any], candidate: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any] | None:
    if int(candidate["page_id"]) == int(source["page_id"]):
        return None
    if candidate["license_short_name"] not in set(policy["allowed_licenses"]):
        return None
    if candidate["mime"] not in set(policy["allowed_mime"]):
        return None
    if min(int(candidate["width"]), int(candidate["height"])) < int(policy["minimum_short_dimension"]):
        return None
    model = visible_text(str(candidate.get("camera_model", "")))
    if policy["digital_camera_model_required"] and not model:
        return None
    if any(token.casefold() in model for token in policy["forbidden_digital_model_tokens"]):
        return None
    candidate_text = visible_text(
        f"{candidate.get('title', '')} {candidate.get('description_raw_html', '')}"
    )
    if any(token.casefold() in candidate_text for token in policy["forbidden_candidate_text_tokens"]):
        return None
    source_time = parse_capture_time(str(source.get("date_time_original", "")))
    candidate_time = parse_capture_time(str(candidate.get("date_time_original", "")))
    if source_time is None or candidate_time is None:
        return None
    delta = abs((candidate_time - source_time).total_seconds())
    if delta > float(policy["maximum_capture_time_delta_seconds"]):
        return None
    jaccard = title_jaccard(str(source["title"]), str(candidate["title"]))
    if jaccard < float(policy["minimum_title_token_jaccard"]):
        return None
    row = dict(candidate)
    row["capture_time_delta_seconds"] = delta
    row["title_token_jaccard"] = jaccard
    return row


def select_metadata_candidates(
    sources: Sequence[Mapping[str, Any]],
    pages_by_identity: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    limit = int(contract["pixel_preflight"]["maximum_candidates_per_target"])
    for source in sorted(sources, key=lambda row: int(row["page_id"])):
        identity = flickr_identity(source)
        if not identity:
            continue
        candidates = []
        for page in pages_by_identity.get(identity, []):
            row = metadata_candidate(source, page, contract["candidate_metadata"])
            if row is not None:
                candidates.append(row)
        candidates.sort(
            key=lambda row: (
                -float(row["title_token_jaccard"]),
                float(row["capture_time_delta_seconds"]),
                int(row["page_id"]),
            )
        )
        for candidate in candidates[:limit]:
            selected.append(
                {
                    "source_page_id": int(source["page_id"]),
                    "source_title": str(source["title"]),
                    "flickr_identity": identity,
                    "candidate": candidate,
                }
            )
    if len(selected) > int(contract["pixel_preflight"]["maximum_thumbnail_downloads"]):
        raise CommonsEktarCompanionError("thumbnail candidate ceiling exceeded")
    return selected


def _native_gray(path: Path, maximum_long_side: int = 1024) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = min(1.0, maximum_long_side / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        rgb = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    result = 0
    for value in bits.ravel():
        result = (result << 1) | int(value)
    return result


def registered_similarity(source_path: Path, candidate_path: Path) -> dict[str, Any]:
    source = _native_gray(source_path)
    candidate = _native_gray(candidate_path)
    sift = cv2.SIFT_create(nfeatures=4096)
    source_keypoints, source_desc = sift.detectAndCompute(source, None)
    candidate_keypoints, candidate_desc = sift.detectAndCompute(candidate, None)
    if source_desc is None or candidate_desc is None:
        return {"sift_matches": 0, "homography_inliers": 0, "homography_inlier_ratio": 0.0}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(candidate_desc, source_desc, k=2)
    matches = [left for left, right in pairs if left.distance < 0.75 * right.distance]
    base = {
        "source_shape": list(source.shape),
        "candidate_shape": list(candidate.shape),
        "sift_matches": len(matches),
        "dhash_hamming": (_dhash(source) ^ _dhash(candidate)).bit_count(),
    }
    if len(matches) < 4:
        return {**base, "homography_inliers": 0, "homography_inlier_ratio": 0.0}
    candidate_points = np.float32([candidate_keypoints[m.queryIdx].pt for m in matches])
    source_points = np.float32([source_keypoints[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.findHomography(candidate_points, source_points, cv2.RANSAC, 3.0)
    if matrix is None or mask is None:
        return {**base, "homography_inliers": 0, "homography_inlier_ratio": 0.0}
    inlier_mask = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(candidate_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - source_points, axis=1)
    warped = cv2.warpPerspective(candidate, matrix, (source.shape[1], source.shape[0]))
    support = cv2.warpPerspective(
        np.full(candidate.shape, 255, dtype=np.uint8), matrix, (source.shape[1], source.shape[0])
    ) > 0
    source_grad = cv2.magnitude(
        cv2.Sobel(source, cv2.CV_32F, 1, 0), cv2.Sobel(source, cv2.CV_32F, 0, 1)
    )
    warped_grad = cv2.magnitude(
        cv2.Sobel(warped, cv2.CV_32F, 1, 0), cv2.Sobel(warped, cv2.CV_32F, 0, 1)
    )
    valid = support & np.isfinite(source_grad) & np.isfinite(warped_grad)
    if int(valid.sum()) < 256 or np.std(source_grad[valid]) == 0 or np.std(warped_grad[valid]) == 0:
        ncc = -1.0
    else:
        ncc = float(np.corrcoef(source_grad[valid], warped_grad[valid])[0, 1])
    return {
        **base,
        "homography_inliers": int(inlier_mask.sum()),
        "homography_inlier_ratio": float(inlier_mask.mean()),
        "median_reprojection_error_pixels": float(np.median(errors[inlier_mask])),
        "registered_gradient_ncc": ncc,
        "registered_support_fraction": float(valid.mean()),
    }


def pixel_gate(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "minimum_sift_matches": int(metrics.get("sift_matches", 0)) >= int(gates["minimum_sift_matches"]),
        "minimum_homography_inliers": int(metrics.get("homography_inliers", 0)) >= int(gates["minimum_homography_inliers"]),
        "minimum_homography_inlier_ratio": float(metrics.get("homography_inlier_ratio", 0.0))
        >= float(gates["minimum_homography_inlier_ratio"]),
        "maximum_median_reprojection_error": float(metrics.get("median_reprojection_error_pixels", math.inf))
        <= float(gates["maximum_median_reprojection_error_pixels"]),
        "minimum_registered_gradient_ncc": float(metrics.get("registered_gradient_ncc", -1.0))
        >= float(gates["minimum_registered_gradient_ncc"]),
        "not_near_duplicate": int(metrics.get("dhash_hamming", 0))
        > int(gates["near_duplicate_hamming_threshold"]),
    }


def audit_snapshot(root: Path, contract: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    sources = snapshot.get("source_rows")
    selected = snapshot.get("selected_candidates")
    if not isinstance(sources, list) or len(sources) != int(contract["expected_target_rows"]):
        raise CommonsEktarCompanionError("snapshot source row count drift")
    if not isinstance(selected, list):
        raise CommonsEktarCompanionError("snapshot selected candidates missing")
    results = []
    for row in sorted(selected, key=lambda value: (value["source_page_id"], value["candidate"]["page_id"])):
        source = root / str(contract["source_pixel_root"]) / f"{row['source_page_id']}.img"
        candidate = root / str(contract["candidate_thumbnail_root"]) / str(row["candidate_local_name"])
        if sha256_file(candidate) != row["candidate_thumbnail_sha256"]:
            raise CommonsEktarCompanionError("candidate thumbnail identity drift")
        metrics = registered_similarity(source, candidate)
        checks = pixel_gate(metrics, contract["pixel_preflight"])
        results.append({**row, "metrics": metrics, "checks": checks, "passed": all(checks.values())})
    passed = [row for row in results if row["passed"]]
    decision = contract["decision_if_pass"] if passed else contract["decision_if_fail"]
    report = {
        "schema": "neuro-film.sf3-a3e-commons-ektar-companion-discovery-report.v1",
        "experiment_id": contract["experiment_id"],
        "input_selection_manifest_sha256": contract["input_selection_manifest_sha256"],
        "metadata_snapshot_sha256": sha256_bytes(canonical_json(snapshot)),
        "source_rows": len(sources),
        "flickr_identities": len(snapshot.get("pages_by_identity", {})),
        "network_requests": int(snapshot.get("network_requests", 0)),
        "metadata_candidates": len(selected),
        "pixel_candidates_passed": len(passed),
        "candidate_results": results,
        "decision": decision,
        "automatic_pass": bool(passed),
        "operator_fit_render_or_target_score_count": 0,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report
