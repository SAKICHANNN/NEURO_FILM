"""Metadata-only audit of one explicit film/digital Flickr album.

The album is unusually useful because one author published several title-
structured same-scene series and licensed nearly all rows CC BY-NC-SA 2.0.
This module does not download pixels, fit an operator, or infer missing stock,
process, scanner, exposure, or roll metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCHEMA = "neuro-film.u5-r2bo0-flickr-single-author-pair-source.v1"
API_ENDPOINT = "https://www.flickr.com/services/rest/"
_SITE_KEY = re.compile(r'api\.site_key = "([0-9a-f]+)"')


class FlickrAlbumAuditError(ValueError):
    """Raised when the frozen source or returned metadata drifts."""


@dataclass(frozen=True)
class PairRole:
    family_id: str
    scene_id: int
    role: str
    stock_label: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def classify_title(title: str, families: list[dict[str, Any]]) -> PairRole | None:
    """Map only an exact configured series title to one pair role."""

    matches: list[PairRole] = []
    for family in families:
        match = re.match(family["pattern"], title)
        if match is None:
            continue
        medium = match.group("medium")
        if medium == family["digital_medium"]:
            role = "digital"
        elif medium == family["film_medium"]:
            role = "film"
        else:
            raise FlickrAlbumAuditError("configured medium mapping is incomplete")
        matches.append(
            PairRole(
                family_id=family["family_id"],
                scene_id=int(match.group("scene")),
                role=role,
                stock_label=family["stock_label"],
            )
        )
    if len(matches) > 1:
        raise FlickrAlbumAuditError(f"title matches multiple families: {title}")
    return matches[0] if matches else None


def audit_payload(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Validate a Flickr REST response and derive title-exact complete pairs."""

    if config.get("schema") != SCHEMA or config["source"].get("image_payloads_allowed"):
        raise FlickrAlbumAuditError("invalid BO0 contract")
    if payload.get("stat") != "ok" or not isinstance(payload.get("photoset"), dict):
        raise FlickrAlbumAuditError("Flickr response is not an ok photoset payload")
    photoset = payload["photoset"]
    source = config["source"]
    if (
        str(photoset.get("id")) != source["album_id"]
        or str(photoset.get("owner")) != source["owner_nsid"]
        or int(photoset.get("total", -1)) != source["expected_public_photo_count"]
    ):
        raise FlickrAlbumAuditError("album identity or public count drift")
    rows = photoset.get("photo")
    if not isinstance(rows, list) or len(rows) != source["expected_public_photo_count"]:
        raise FlickrAlbumAuditError("album row count drift")

    allowed_licenses = {int(value) for value in source["allowed_license_ids"]}
    records: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"digital": [], "film": []}
    )
    photo_ids: list[str] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise FlickrAlbumAuditError("photo row is not an object")
        photo_id = str(raw.get("id", ""))
        title = raw.get("title")
        if not photo_id.isdigit() or not isinstance(title, str) or not title:
            raise FlickrAlbumAuditError("photo identity or title is invalid")
        photo_ids.append(photo_id)
        license_id = int(raw.get("license", -1))
        role = classify_title(title, config["pair_families"])
        derivative_url = raw.get("url_l")
        width = raw.get("width_l")
        height = raw.get("height_l")
        record = {
            "position": position,
            "photo_id": photo_id,
            "page_url": (
                f"https://www.flickr.com/photos/{source['owner_nsid']}/{photo_id}"
            ),
            "title": title,
            "license_id": license_id,
            "public_photo_media": bool(
                raw.get("ispublic") == 1
                and raw.get("media") == "photo"
                and raw.get("media_status") == "ready"
            ),
            "derivative_url_l": derivative_url if isinstance(derivative_url, str) else None,
            "derivative_width_l": int(width) if isinstance(width, int) else None,
            "derivative_height_l": int(height) if isinstance(height, int) else None,
            "family_id": role.family_id if role else None,
            "scene_id": role.scene_id if role else None,
            "role": role.role if role else None,
            "stock_label": role.stock_label if role else None,
        }
        records.append(record)
        if role:
            grouped[(role.family_id, role.scene_id)][role.role].append(record)

    duplicate_count = len(photo_ids) - len(set(photo_ids))
    complete_pairs: list[dict[str, Any]] = []
    ambiguous_or_incomplete: list[dict[str, Any]] = []
    for (family_id, scene_id), roles in sorted(grouped.items()):
        if len(roles["digital"]) == 1 and len(roles["film"]) == 1:
            digital = roles["digital"][0]
            film = roles["film"][0]
            complete_pairs.append(
                {
                    "pair_id": f"{family_id}/scene-{scene_id:02d}",
                    "family_id": family_id,
                    "scene_id": scene_id,
                    "digital_photo_id": digital["photo_id"],
                    "film_photo_id": film["photo_id"],
                    "stock_label": film["stock_label"],
                    "both_allowed_license": bool(
                        digital["license_id"] in allowed_licenses
                        and film["license_id"] in allowed_licenses
                    ),
                    "both_public_photo_media": bool(
                        digital["public_photo_media"] and film["public_photo_media"]
                    ),
                    "both_bounded_derivatives": bool(
                        digital["derivative_url_l"] and film["derivative_url_l"]
                    ),
                }
            )
        else:
            ambiguous_or_incomplete.append(
                {
                    "family_id": family_id,
                    "scene_id": scene_id,
                    "digital_count": len(roles["digital"]),
                    "film_count": len(roles["film"]),
                }
            )

    eligible = [
        pair
        for pair in complete_pairs
        if pair["both_allowed_license"]
        and pair["both_public_photo_media"]
        and pair["both_bounded_derivatives"]
    ]
    counts = Counter(pair["family_id"] for pair in eligible)
    largest_share = max(counts.values(), default=0) / max(len(eligible), 1)
    gates = config["feasibility_gates"]
    checks = {
        "complete_rights_eligible_pairs": len(eligible)
        >= gates["minimum_complete_rights_eligible_pairs"],
        "independent_capture_families": len(counts)
        >= gates["minimum_independent_capture_families"],
        "largest_family_share": largest_share
        <= gates["maximum_largest_family_pair_share"],
        "duplicate_photo_ids": duplicate_count
        <= gates["maximum_duplicate_photo_ids"],
        "pair_rows_public_photo_media": (
            all(pair["both_public_photo_media"] for pair in complete_pairs)
            if gates["require_all_pair_rows_public_photo_media"]
            else True
        ),
        "bounded_derivative_urls": (
            all(pair["both_bounded_derivatives"] for pair in complete_pairs)
            if gates["require_bounded_derivative_url"]
            else True
        ),
    }
    passed = bool(all(checks.values()))
    stable = {
        "schema": "neuro-film.u5-r2bo0-flickr-single-author-pair-source-report.v1",
        "node": config["node"],
        "source": {
            "album_url": source["album_url"],
            "album_id": source["album_id"],
            "owner_nsid": source["owner_nsid"],
            "license_label": source["license_label"],
            "metadata_requests": 2,
            "pixel_payloads_requested": 0,
        },
        "metrics": {
            "public_photo_count": len(records),
            "license_counts": {
                str(key): value
                for key, value in sorted(Counter(row["license_id"] for row in records).items())
            },
            "title_matched_complete_pairs": len(complete_pairs),
            "eligible_complete_pairs": len(eligible),
            "family_pair_counts": dict(sorted(counts.items())),
            "largest_family_pair_share": largest_share,
            "ambiguous_or_incomplete_series_rows": ambiguous_or_incomplete,
            "duplicate_photo_ids": duplicate_count,
        },
        "checks": checks,
        "automatic_pass": passed,
        "branch": config["branches"]["pass" if passed else "fail"],
        "complete_pairs": complete_pairs,
        "records": records,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest()}


def fetch_album_payload(config: dict[str, Any], timeout_seconds: float = 30.0) -> dict[str, Any]:
    """Perform exactly one album-page and one bounded REST metadata request."""

    source = config["source"]
    headers = {"User-Agent": "neuro-film-source-audit/1.0"}
    request = Request(source["album_url"], headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        html = response.read().decode("utf-8", "replace")
    match = _SITE_KEY.search(html)
    if match is None:
        raise FlickrAlbumAuditError("ephemeral Flickr site key is unavailable")
    query = urlencode(
        {
            "method": "flickr.photosets.getPhotos",
            "api_key": match.group(1),
            "photoset_id": source["album_id"],
            "user_id": source["owner_nsid"],
            "extras": "license,url_l,date_taken,media",
            "per_page": source["expected_public_photo_count"],
            "format": "json",
            "nojsoncallback": 1,
        }
    )
    request = Request(f"{API_ENDPOINT}?{query}", headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise FlickrAlbumAuditError("Flickr response is not an object")
    return payload


__all__ = [
    "FlickrAlbumAuditError",
    "PairRole",
    "SCHEMA",
    "audit_payload",
    "classify_title",
    "fetch_album_payload",
]
