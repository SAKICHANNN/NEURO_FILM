"""Metadata-only audit of a small CC-BY film/digital B&W composite series."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCHEMA = "neuro-film.u5-r2bo9-flickr-bw-composite-pair-source.v1"
API_ENDPOINT = "https://www.flickr.com/services/rest/"
_SITE_KEY = re.compile(r'api\.site_key = "([0-9a-f]+)"')


class FlickrBwCompositeSourceError(ValueError):
    """Raised when the frozen Flickr source metadata drifts."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def audit_payload(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != SCHEMA or config["source"].get("image_payloads_allowed"):
        raise FlickrBwCompositeSourceError("invalid BO9 contract")
    if re.fullmatch(r"[0-9a-f]{40}", str(config.get("software_commit", ""))) is None:
        raise FlickrBwCompositeSourceError("software commit is not frozen")
    if payload.get("stat") != "ok" or not isinstance(payload.get("photos"), dict):
        raise FlickrBwCompositeSourceError("Flickr search response is not valid")
    source = config["source"]
    photos = payload["photos"]
    if int(photos.get("total", -1)) != int(source["expected_search_total"]):
        raise FlickrBwCompositeSourceError("search result count drift")
    rows = photos.get("photo")
    if not isinstance(rows, list) or len(rows) != int(source["expected_search_total"]):
        raise FlickrBwCompositeSourceError("search rows are incomplete")
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    if len(by_id) != len(rows):
        raise FlickrBwCompositeSourceError("duplicate or malformed photo identity")

    allowed = {int(value) for value in source["allowed_license_ids"]}
    gates = config["feasibility_gates"]
    records: list[dict[str, Any]] = []
    for expected in config["eligible_composites"]:
        photo_id = str(expected["photo_id"])
        row = by_id.get(photo_id)
        if row is None:
            raise FlickrBwCompositeSourceError(f"missing frozen composite {photo_id}")
        description = row.get("description", {}).get("_content")
        if not isinstance(description, str):
            raise FlickrBwCompositeSourceError("description is unavailable")
        records.append(
            {
                "scene_id": int(expected["scene_id"]),
                "photo_id": photo_id,
                "page_url": f"https://www.flickr.com/photos/{source['owner_nsid']}/{photo_id}",
                "title": str(row.get("title", "")),
                "license_id": int(row.get("license", -1)),
                "public_photo_media": bool(
                    int(row.get("ispublic", 0)) == 1 and row.get("media") == "photo"
                ),
                "derivative_url_l": row.get("url_l"),
                "derivative_width_l": int(row.get("width_l", -1)),
                "derivative_height_l": int(row.get("height_l", -1)),
                "role_text_exact": all(text in description for text in gates["require_role_text"]),
                "film_role": "top/minolta-xg-2",
                "digital_role": "bottom/sony-slt-77",
            }
        )

    exclusions = []
    for expected in config["explicit_exclusions"]:
        photo_id = str(expected["photo_id"])
        exclusions.append(
            {
                "photo_id": photo_id,
                "present": photo_id in by_id,
                "reason": str(expected["reason"]),
            }
        )
    checks = {
        "complete_composites": len(records) == int(gates["required_complete_composites"]),
        "allowed_licenses": all(row["license_id"] in allowed for row in records),
        "public_photo_media": all(row["public_photo_media"] for row in records),
        "bounded_derivatives": all(isinstance(row["derivative_url_l"], str) for row in records),
        "frozen_dimensions": all(
            row["derivative_width_l"] == int(gates["required_width"])
            and row["derivative_height_l"] == int(gates["required_height"])
            for row in records
        ),
        "explicit_role_text": all(row["role_text_exact"] for row in records),
        "excluded_rows_present": all(row["present"] for row in exclusions),
    }
    passed = all(checks.values())
    stable = {
        "schema": "neuro-film.u5-r2bo9-flickr-bw-composite-pair-source-report.v1",
        "node": config["node"],
        "software_commit": config["software_commit"],
        "source": {
            "owner_nsid": source["owner_nsid"],
            "tag": source["tag"],
            "license_label": source["license_label"],
            "metadata_requests": 2,
            "pixel_payloads_requested": 0,
        },
        "metrics": {
            "search_total": len(rows),
            "complete_composites": len(records),
            "excluded_single_view_rows": len(exclusions),
            "independent_authors": 1,
        },
        "checks": checks,
        "automatic_pass": passed,
        "branch": config["branches"]["pass" if passed else "fail"],
        "records": records,
        "explicit_exclusions": exclusions,
        "training_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}


def fetch_search_payload(config: dict[str, Any], timeout_seconds: float = 30.0) -> dict[str, Any]:
    source = config["source"]
    headers = {"User-Agent": "neuro-film-source-audit/1.0"}
    with urlopen(Request(source["search_page_url"], headers=headers), timeout=timeout_seconds) as response:
        html = response.read().decode("utf-8", "replace")
    match = _SITE_KEY.search(html)
    if match is None:
        raise FlickrBwCompositeSourceError("ephemeral Flickr site key is unavailable")
    query = urlencode(
        {
            "method": "flickr.photos.search",
            "api_key": match.group(1),
            "user_id": source["owner_nsid"],
            "tags": source["tag"],
            "tag_mode": "all",
            "extras": "license,description,tags,url_l,owner_name,date_taken,media",
            "per_page": 100,
            "page": 1,
            "content_type": 1,
            "media": "photos",
            "format": "json",
            "nojsoncallback": 1,
        }
    )
    with urlopen(Request(f"{API_ENDPOINT}?{query}", headers=headers), timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise FlickrBwCompositeSourceError("Flickr response is not an object")
    return payload


__all__ = [
    "FlickrBwCompositeSourceError",
    "SCHEMA",
    "audit_payload",
    "canonical_bytes",
    "fetch_search_payload",
]
