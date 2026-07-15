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


DIRECT_LOC_ID_RE = re.compile(r"\bfsac\.([0-9a-z]+)\b", re.IGNORECASE)
LOC_PATH_ID_RE = re.compile(
    r"/fsac/(?:[^\s\"'<>]+/)*([0-9a-z]+?)(?:_150px|[rtvu])?\."
    r"(?:jpe?g|gif|tiff?)\b",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")


class FsaOwiAcquisitionError(RuntimeError):
    """Raised when the source or frozen metadata contract is violated."""


def plain_text(value: object) -> str:
    """Return stable readable text from the small HTML fragments in extmetadata."""
    unescaped = html.unescape(str(value or ""))
    return " ".join(TAG_RE.sub(" ", unescaped).split())


def extract_loc_id(credit_html: object) -> str | None:
    """Extract and normalize the LOC digital id carried by Commons credit data."""
    credit = html.unescape(str(credit_html or ""))
    direct = DIRECT_LOC_ID_RE.search(credit)
    if direct:
        return f"fsac.{direct.group(1).lower()}"
    paths = LOC_PATH_ID_RE.findall(credit)
    return f"fsac.{paths[-1].lower()}" if paths else None


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
    # requests.Session already carries a generic python-requests user agent;
    # replace it because Wikimedia requires an identifying bot/research agent.
    client.headers["User-Agent"] = (
        "neuro-film-research/0.1 "
        "(https://github.com/SAKICHANNN/NEURO_FILM; metadata provenance audit)"
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


def creator_group(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    """Recover the curated photographer category, avoiding free-text aliases."""
    prefix = str(config["canonical_recovery"]["creator_category_prefix"])
    names = sorted(
        {
            str(category)[len(prefix) :].strip()
            for category in record.get("categories", [])
            if str(category).startswith(prefix) and str(category)[len(prefix) :].strip()
        }
    )
    return " + ".join(names) if names else None


def _canonical_rank(record: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, ...]:
    title = str(record["commons_title"])
    title_lower = title.lower()
    identifier = str(record["loc_fsac_id"]).split(".", 1)[-1]
    forbidden = tuple(
        str(token).lower() for token in config["canonical_recovery"]["forbidden_title_tokens"]
    )
    edited = any(token in title_lower for token in forbidden)
    explicit_v = bool(re.search(rf"\b{re.escape(identifier)}v\b", title_lower))
    lccn = "lccn" in title_lower
    jpeg = str(record.get("mime")) == "image/jpeg"
    area = int(record.get("original_width", 0)) * int(record.get("original_height", 0))
    return (edited, not explicit_v, not lccn, not jpeg, -area, title_lower)


def build_canonical_subset(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one strict, deterministic Commons representation per LOC scan id."""
    license_gate = config["license_gate"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    excluded = Counter()
    for record in records:
        identifier = record.get("loc_fsac_id")
        if not identifier:
            excluded["missing_loc_id"] += 1
            continue
        if str(record.get("license")) != license_gate["required_license_short_name"]:
            excluded["license_mismatch"] += 1
            continue
        categories = record.get("categories", [])
        if license_gate["required_commons_category"] not in categories:
            excluded["missing_public_domain_category"] += 1
            continue
        if license_gate["required_source_category"] not in categories:
            excluded["missing_loc_source_category"] += 1
            continue
        grouped.setdefault(str(identifier), []).append(record)
    selected: list[dict[str, Any]] = []
    for identifier, candidates in sorted(grouped.items()):
        chosen = dict(min(candidates, key=lambda row: _canonical_rank(row, config)))
        chosen["creator_group"] = creator_group(chosen, config)
        chosen["canonical_loc_family_size"] = len(candidates)
        chosen["canonical_selection_rank"] = list(_canonical_rank(chosen, config)[:-1])
        selected.append(chosen)
    recovery = config["canonical_recovery"]
    known = [row for row in selected if row["creator_group"]]
    creator_counts = Counter(str(row["creator_group"]) for row in known)
    checks = {
        "minimum_unique_loc_records": len(selected)
        >= int(recovery["minimum_unique_loc_records"]),
        "minimum_creator_group_coverage": (len(known) / len(selected) if selected else 0.0)
        >= float(recovery["minimum_creator_group_coverage"]),
        "minimum_creators_with_eight_records": sum(
            count >= 8 for count in creator_counts.values()
        )
        >= int(recovery["minimum_creators_with_eight_records"]),
        "one_record_per_loc_id": len(selected)
        == len({str(row["loc_fsac_id"]) for row in selected}),
    }
    passed = bool(selected and all(checks.values()))
    audit = {
        "passed": passed,
        "decision": "canonical_pilot_allowed" if passed else "canonical_ineligible",
        "checks": checks,
        "canonical_records": len(selected),
        "raw_eligible_records": sum(len(value) for value in grouped.values()),
        "duplicate_representations_removed": sum(len(value) - 1 for value in grouped.values()),
        "excluded_records": dict(sorted(excluded.items())),
        "creator_group_coverage": len(known) / len(selected) if selected else 0.0,
        "creator_group_counts": dict(sorted(creator_counts.items())),
    }
    return selected, audit
