"""Bounded metadata-only audit for Flickr's Analog + Digital pair pool.

The source is interesting because its rules ask contributors to submit the
same subject on film and digital cameras.  This module deliberately opens only
the signed-out pool HTML and individual photo pages.  It never requests image
payloads and treats adjacency as a candidate relationship, not verified
pairing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html import unescape
from itertools import pairwise
from typing import Any
from urllib.request import Request, urlopen

POOL_URL = "https://www.flickr.com/groups/lomoplusdigital/pool/"
PHOTO_URL = "https://www.flickr.com/photos/{owner_alias}/{photo_id}"
_PHOTO_LINK = re.compile(
    r'href="/photos/([^/]+)/([0-9]+)/in/pool-lomoplusdigital"'
    r'[^>]*title="([^"]*)"'
)
_KEYWORDS = re.compile(r'<meta name="keywords" content="([^"]*)"')
_DATE_TAKEN = re.compile(r'"dateTaken":"([^"]+)"')
_MODEL_MARKER = 'params: {"photoModel"'
_DERIVATIVE_RIGHTS_LICENSES = frozenset({1, 2, 4, 5, 9, 10, 11, 12, 14, 15})
_FILM_MARKERS = (
    "i shot film",
    "scan fujifilm",
    "scan kodak",
    "scan agfa",
    "scan paradies",
    "scan rossmann",
    "scan schlecker",
    "expired film",
)
_DIGITAL_MARKERS = (
    "digitalfoto",
    "digitale variante",
    "- digital",
    "digital variant",
)
_DIGITAL_CAMERA_MARKERS = ("dslr", "nikon d", "nikond", "sony alpha")
_FILM_METADATA_MARKERS = (
    "i shot film",
    "film",
    "analog",
    "scan kodak",
    "scan fujifilm",
    "scan agfa",
)


class FlickrPairAuditError(ValueError):
    """Raised when the bounded source contract or returned HTML drifts."""


@dataclass(frozen=True)
class PoolRow:
    page: int
    position: int
    owner_alias: str
    photo_id: str
    pool_title: str


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FlickrPairAuditError(f"{name} must be a non-empty string")
    return value


def parse_pool_page(html: str, *, page: int) -> list[PoolRow]:
    """Extract one ordered, de-duplicated signed-out pool page."""

    if page not in (1, 2):
        raise FlickrPairAuditError("only the frozen signed-out pages 1-2 are allowed")
    seen: set[str] = set()
    rows: list[PoolRow] = []
    for owner_alias, photo_id, title in _PHOTO_LINK.findall(html):
        if photo_id in seen:
            continue
        seen.add(photo_id)
        rows.append(
            PoolRow(
                page=page,
                position=len(rows),
                owner_alias=owner_alias,
                photo_id=photo_id,
                pool_title=unescape(title),
            )
        )
    if not rows:
        raise FlickrPairAuditError("pool page contains no photo rows")
    return rows


def parse_photo_page(html: str, row: PoolRow) -> dict[str, Any]:
    """Extract stable metadata from Flickr's signed-out photo page model."""

    marker = html.find(_MODEL_MARKER)
    if marker < 0:
        raise FlickrPairAuditError(f"photo model missing for {row.photo_id}")
    start = html.find("{", marker)
    try:
        payload, _ = json.JSONDecoder().raw_decode(html[start:])
    except (json.JSONDecodeError, TypeError) as exc:
        raise FlickrPairAuditError(
            f"photo model is not strict JSON for {row.photo_id}"
        ) from exc
    model = payload.get("photoModel")
    if not isinstance(model, dict) or model.get("id") != row.photo_id:
        raise FlickrPairAuditError(f"photo identity mismatch for {row.photo_id}")
    owner = model.get("owner")
    if not isinstance(owner, dict):
        raise FlickrPairAuditError(f"owner metadata missing for {row.photo_id}")
    license_id = model.get("license")
    if not isinstance(license_id, int) or not 0 <= license_id <= 16:
        raise FlickrPairAuditError(f"invalid license for {row.photo_id}")
    sizes = model.get("sizes")
    if not isinstance(sizes, dict) or not sizes:
        raise FlickrPairAuditError(f"size metadata missing for {row.photo_id}")
    keywords_match = _KEYWORDS.search(html)
    date_match = _DATE_TAKEN.search(html)
    title = _nonempty(model.get("title"), "photo title")
    description = model.get("description")
    if not isinstance(description, str):
        raise FlickrPairAuditError(f"description drift for {row.photo_id}")
    keywords = (
        tuple(
            item.strip().lower()
            for item in unescape(keywords_match.group(1)).split(",")
            if item.strip()
        )
        if keywords_match
        else ()
    )
    combined = " ".join((title, description, " ".join(keywords))).lower()
    film_explicit = any(marker in combined for marker in _FILM_MARKERS)
    digital_explicit = any(marker in combined for marker in _DIGITAL_MARKERS)
    return {
        "page": row.page,
        "position": row.position,
        "owner_alias": row.owner_alias,
        "owner_nsid": _nonempty(owner.get("nsid"), "owner nsid"),
        "photo_id": row.photo_id,
        "title": title,
        "description": description,
        "license_id": license_id,
        "derivative_rights_eligible": license_id
        in _DERIVATIVE_RIGHTS_LICENSES,
        "date_taken": date_match.group(1) if date_match else None,
        "keywords": list(keywords),
        "original_width": model.get("oWidth"),
        "original_height": model.get("oHeight"),
        "film_explicit": film_explicit,
        "digital_explicit": digital_explicit,
        "digital_camera_metadata": any(
            marker in combined for marker in _DIGITAL_CAMERA_MARKERS
        ),
        "film_metadata": any(
            marker in combined for marker in _FILM_METADATA_MARKERS
        ),
    }


def candidate_adjacent_pairs(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only explicit film/digital adjacent candidates.

    Pool adjacency is weak evidence.  Both sides must independently identify
    their medium and share the same owner; the result remains a candidate.
    """

    ordered = sorted(records, key=lambda item: (item["page"], item["position"]))
    pairs: list[dict[str, Any]] = []
    for left, right in pairwise(ordered):
        if (
            left["page"] != right["page"]
            or left["owner_nsid"] != right["owner_nsid"]
        ):
            continue
        orientations = (
            (left, right)
            if left["film_explicit"] and right["digital_explicit"]
            else (right, left)
            if right["film_explicit"] and left["digital_explicit"]
            else None
        )
        if orientations is None:
            continue
        film, digital = orientations
        pairs.append(
            {
                "owner_nsid": film["owner_nsid"],
                "film_photo_id": film["photo_id"],
                "digital_photo_id": digital["photo_id"],
                "both_derivative_rights_eligible": bool(
                    film["derivative_rights_eligible"]
                    and digital["derivative_rights_eligible"]
                ),
            }
        )
    return pairs


def summarize_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise FlickrPairAuditError("records must not be empty")
    photo_ids = [row["photo_id"] for row in records]
    if len(photo_ids) != len(set(photo_ids)):
        raise FlickrPairAuditError("photo IDs must be unique")
    pairs = candidate_adjacent_pairs(records)
    rights_records = [row for row in records if row["derivative_rights_eligible"]]
    rights_owners = Counter(row["owner_nsid"] for row in rights_records)
    return {
        "visible_record_count": len(records),
        "visible_owner_count": len({row["owner_nsid"] for row in records}),
        "license_counts": {
            str(key): value
            for key, value in sorted(
                Counter(row["license_id"] for row in records).items()
            )
        },
        "derivative_rights_record_count": len(rights_records),
        "derivative_rights_owner_count": len(rights_owners),
        "largest_rights_owner_share": (
            max(rights_owners.values()) / len(rights_records)
            if rights_records
            else 0.0
        ),
        "rights_rows_with_digital_camera_metadata": sum(
            row["digital_camera_metadata"] for row in rights_records
        ),
        "rights_rows_with_film_metadata": sum(
            row["film_metadata"] for row in rights_records
        ),
        "explicit_adjacent_candidate_pair_count": len(pairs),
        "rights_eligible_explicit_pair_count": sum(
            pair["both_derivative_rights_eligible"] for pair in pairs
        ),
        "explicit_adjacent_candidate_pairs": pairs,
        "pixel_payloads_requested": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "decision": (
            "open_bounded_pixel_preflight"
            if any(pair["both_derivative_rights_eligible"] for pair in pairs)
            else "close_current_visible_pool_no_rights_eligible_explicit_pair"
        ),
    }


def _default_fetch(url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"User-Agent": "neuro-film-source-audit/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", "replace")


def run_metadata_audit(
    *,
    fetch: Callable[[str, float], str] = _default_fetch,
    timeout_seconds: float = 30.0,
    workers: int = 6,
) -> dict[str, Any]:
    """Fetch the frozen two-page signed-out surface and audit its metadata."""

    if not 1 <= workers <= 8:
        raise FlickrPairAuditError("workers must be in [1, 8]")
    pages = [
        (1, POOL_URL),
        (2, f"{POOL_URL}page2/"),
    ]
    pool_rows: list[PoolRow] = []
    for page, url in pages:
        pool_rows.extend(
            parse_pool_page(fetch(url, timeout_seconds), page=page)
        )
    ids = [row.photo_id for row in pool_rows]
    if len(ids) != len(set(ids)):
        raise FlickrPairAuditError("signed-out pool pages overlap")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch,
                PHOTO_URL.format(
                    owner_alias=row.owner_alias, photo_id=row.photo_id
                ),
                timeout_seconds,
            ): row
            for row in pool_rows
        }
        for future in as_completed(futures):
            row = futures[future]
            records.append(parse_photo_page(future.result(), row))
    records.sort(key=lambda item: (item["page"], item["position"]))
    return {
        "schema": "neuro-film.u5-r2bb0-flickr-analog-digital-source.v1",
        "source": {
            "pool_url": POOL_URL,
            "group_rule": (
                "same subject submitted as one film and one digital photo"
            ),
            "signed_out_pages_audited": 2,
        },
        "summary": summarize_audit(records),
        "records": records,
    }


__all__ = [
    "FlickrPairAuditError",
    "PoolRow",
    "candidate_adjacent_pairs",
    "parse_photo_page",
    "parse_pool_page",
    "run_metadata_audit",
    "summarize_audit",
]
