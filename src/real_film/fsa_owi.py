"""Metadata acquisition and fail-closed gates for the LOC FSA/OWI archive."""

from __future__ import annotations

import hashlib
import html
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

import requests


LOC_ID_RE = re.compile(r"\bfsac[./]([0-9a-z]+)\b", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")


class FsaOwiAcquisitionError(RuntimeError):
    """Raised when the source or frozen metadata contract is violated."""


def plain_text(value: object) -> str:
    """Return stable readable text from the small HTML fragments in extmetadata."""
    unescaped = html.unescape(str(value or ""))
    return " ".join(TAG_RE.sub(" ", unescaped).split())


def extract_loc_id(credit_html: object) -> str | None:
    """Extract and normalize the LOC digital id carried by Commons credit data."""
    match = LOC_ID_RE.search(html.unescape(str(credit_html or "")))
    return f"fsac.{match.group(1).lower()}" if match else None


def _metadata_value(metadata: Mapping[str, Any], key: str) -> str:
    entry = metadata.get(key, {})
    if not isinstance(entry, Mapping):
        return ""
    return str(entry.get("value", ""))


def parse_page(page: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one complete MediaWiki page record to the stable manifest schema."""
    image_info = page.get("imageinfo")
    if not isinstance(image_info, Sequence) or not image_info:
        raise FsaOwiAcquisitionError("page is missing imageinfo")
    info = image_info[0]
    if not isinstance(info, Mapping):
        raise FsaOwiAcquisitionError("invalid imageinfo record")
    metadata = info.get("extmetadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    categories = tuple(
        sorted(
            item.strip()
            for item in _metadata_value(metadata, "Categories").split("|")
            if item.strip()
        )
    )
    title = str(page["title"])
    return {
        "commons_page_id": int(page["pageid"]),
        "commons_title": title,
        "commons_page_url": "https://commons.wikimedia.org/wiki/" + quote(
            title.replace(" ", "_"), safe=":()/,-_.'"
        ),
        "loc_fsac_id": extract_loc_id(_metadata_value(metadata, "Credit")),
        "creator": plain_text(_metadata_value(metadata, "Artist")),
        "date": plain_text(_metadata_value(metadata, "DateTimeOriginal")),
        "description": plain_text(_metadata_value(metadata, "ObjectName")),
        "license": plain_text(_metadata_value(metadata, "LicenseShortName")),
        "usage_terms": plain_text(_metadata_value(metadata, "UsageTerms")),
        "attribution_required": plain_text(
            _metadata_value(metadata, "AttributionRequired")
        ).lower(),
        "categories": list(categories),
        "source_url": str(info.get("descriptionurl", "")),
        "original_url": str(info.get("url", "")),
        "derivative_url": str(info.get("thumburl", "")),
        "original_width": int(info.get("width", 0)),
        "original_height": int(info.get("height", 0)),
        "original_bytes": int(info.get("size", 0)),
        "commons_sha1": str(info.get("sha1", "")),
        "mime": str(info.get("mime", "")),
        "derivative_width": int(info.get("thumbwidth", 0)),
        "derivative_height": int(info.get("thumbheight", 0)),
    }


def merge_api_pages(
    accumulated: dict[int, dict[str, Any]], pages: Sequence[Mapping[str, Any]]
) -> None:
    """Merge split generator/property batches without losing imageinfo records."""
    for page in pages:
        page_id = int(page["pageid"])
        existing = accumulated.setdefault(page_id, {})
        for key, value in page.items():
            if key == "imageinfo" and key in existing:
                if existing[key] != value:
                    raise FsaOwiAcquisitionError(
                        f"conflicting imageinfo for Commons page {page_id}"
                    )
            else:
                existing[key] = value


def fetch_category_records(
    config: Mapping[str, Any], *, session: requests.Session | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate the frozen Commons category and return records plus page evidence."""
    client = session or requests.Session()
    client.headers.setdefault(
        "User-Agent", "neuro-film-research/0.1 (metadata provenance audit)"
    )
    phase = config["phase_a_metadata"]
    collection = config["collection"]
    base_params: dict[str, Any] = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": collection["commons_category"],
        "gcmtype": "file",
        "gcmlimit": int(phase["page_size"]),
        "prop": "imageinfo",
        "iiprop": "url|size|sha1|mime|extmetadata",
        "iiurlwidth": int(config["phase_b_pilot"]["thumbnail_width"]),
        "format": "json",
        "formatversion": 2,
    }
    continuation: dict[str, Any] = {}
    pages: dict[int, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for call_index in range(100):
        params = {**base_params, **continuation}
        response: requests.Response | None = None
        last_error: Exception | None = None
        for attempt in range(int(phase["max_retries"])):
            try:
                response = client.get(
                    collection["commons_api_url"],
                    params=params,
                    timeout=int(phase["timeout_seconds"]),
                )
                response.raise_for_status()
                if len(response.content) > int(phase["max_response_bytes"]):
                    raise FsaOwiAcquisitionError("metadata response exceeded byte cap")
                break
            except (requests.RequestException, FsaOwiAcquisitionError) as error:
                last_error = error
                if attempt + 1 < int(phase["max_retries"]):
                    time.sleep(0.5 * (2**attempt))
        if response is None or last_error is not None and not response.ok:
            raise FsaOwiAcquisitionError(f"metadata request failed: {last_error}")
        try:
            payload = response.json()
        except ValueError as error:
            raise FsaOwiAcquisitionError("metadata response was not JSON") from error
        if "error" in payload:
            raise FsaOwiAcquisitionError(f"MediaWiki API error: {payload['error']}")
        response_pages = payload.get("query", {}).get("pages", [])
        merge_api_pages(pages, response_pages)
        evidence.append(
            {
                "call_index": call_index,
                "response_bytes": len(response.content),
                "response_sha256": hashlib.sha256(response.content).hexdigest(),
                "page_records": len(response_pages),
                "continuation_in": dict(sorted(continuation.items())),
                "continuation_out": dict(sorted(payload.get("continue", {}).items())),
            }
        )
        if "continue" not in payload:
            break
        continuation = dict(payload["continue"])
    else:
        raise FsaOwiAcquisitionError("metadata pagination exceeded 100 calls")
    incomplete = [page_id for page_id, page in pages.items() if "imageinfo" not in page]
    if incomplete:
        raise FsaOwiAcquisitionError(
            f"{len(incomplete)} Commons pages remained without imageinfo"
        )
    records = [parse_page(page) for page in pages.values()]
    records.sort(key=lambda row: (str(row["loc_fsac_id"] or "~"), row["commons_title"]))
    return records, evidence


def evaluate_metadata_gate(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen metadata, rights and grouping eligibility thresholds."""
    phase = config["phase_a_metadata"]
    license_gate = config["license_gate"]
    count = len(records)
    unique_page_ids = len({int(row["commons_page_id"]) for row in records})
    unique_titles = len({str(row["commons_title"]) for row in records})
    loc_ids = [str(row["loc_fsac_id"]) for row in records if row.get("loc_fsac_id")]
    unique_loc_ids = len(set(loc_ids))
    public_domain = sum(
        str(row.get("license")) == license_gate["required_license_short_name"]
        and license_gate["required_commons_category"] in row.get("categories", [])
        for row in records
    )
    loc_source = sum(
        license_gate["required_source_category"] in row.get("categories", [])
        for row in records
    )
    creators = Counter(str(row.get("creator", "")).strip() for row in records)
    creators.pop("", None)
    coverage = lambda numerator: float(numerator / count) if count else 0.0
    checks = {
        "minimum_unique_records": unique_page_ids >= int(phase["minimum_unique_records"]),
        "unique_titles": unique_titles == count,
        "unique_loc_identifiers": unique_loc_ids == len(loc_ids),
        "minimum_unique_photographers": len(creators)
        >= int(phase["minimum_unique_photographers"]),
        "minimum_loc_identifier_coverage": coverage(len(loc_ids))
        >= float(phase["minimum_loc_identifier_coverage"]),
        "minimum_public_domain_coverage": coverage(public_domain)
        >= float(phase["minimum_public_domain_coverage"]),
        "minimum_loc_source_coverage": coverage(loc_source)
        >= float(phase["minimum_loc_source_coverage"]),
    }
    passed = bool(count and all(checks.values()))
    return {
        "passed": passed,
        "decision": "pilot_allowed" if passed else "metadata_ineligible",
        "checks": checks,
        "records": count,
        "unique_page_ids": unique_page_ids,
        "unique_titles": unique_titles,
        "unique_loc_identifiers": unique_loc_ids,
        "loc_identifier_coverage": coverage(len(loc_ids)),
        "public_domain_coverage": coverage(public_domain),
        "loc_source_coverage": coverage(loc_source),
        "unique_creators": len(creators),
        "creator_counts": dict(sorted(creators.items())),
        "mime_counts": dict(sorted(Counter(str(row.get("mime")) for row in records).items())),
        "total_original_bytes_metadata": sum(int(row.get("original_bytes", 0)) for row in records),
    }
