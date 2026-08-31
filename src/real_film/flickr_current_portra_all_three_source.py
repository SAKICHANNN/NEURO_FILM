"""Metadata-only current-Portra all-three Flickr source audit."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

API_ENDPOINT = "https://www.flickr.com/services/rest/"
_SITE_KEY = re.compile(r'api\.site_key = "([0-9a-f]+)"')


class FlickrCurrentPortraSourceError(ValueError):
    """Raised when the frozen Flickr metadata source drifts or is malformed."""


PayloadFetcher = Callable[[dict[str, Any]], Sequence[dict[str, Any]]]


def canonical_bytes(value: object) -> bytes:
    """Serialize one evidence object deterministically."""

    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text(row: dict[str, Any]) -> str:
    description = row.get("description")
    if isinstance(description, dict):
        description = description.get("_content", "")
    return " ".join(
        (str(row.get("title", "")), str(description or ""), str(row.get("tags", "")))
    ).casefold()


def classify_target_row(
    row: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    """Classify only an explicit target-stock metadata row."""

    text = _text(row)
    current = any(
        indicator.casefold() in text
        for indicator in config["source"]["current_portra_indicators"]
    )
    matches = {
        "kodak_ektar_100": bool(
            re.search(r"\b(?:kodak\s+)?ektar\s*100\b|\bkodakektar100\b", text)
        ),
        "fujifilm_velvia_50": bool(
            re.search(
                r"\b(?:fuji(?:film)?\s+)?velvia\s*50\b|\bfujivelvia50\b",
                text,
            )
        ),
        "kodak_portra_400_current": current
        and bool(re.search(r"\b(?:kodak\s+)?portra\s*400\b|\bkodakportra400\b", text)),
    }
    stock_ids = [stock_id for stock_id, matched in matches.items() if matched]
    if not stock_ids:
        return None
    if len(stock_ids) != 1:
        raise FlickrCurrentPortraSourceError(
            f"photo {row.get('id')} matches multiple target stocks"
        )
    description = row.get("description")
    if isinstance(description, dict):
        description = description.get("_content", "")
    alteration_terms = sorted(
        term
        for term in config["source"]["disqualifying_alteration_terms"]
        if term.casefold() in text
    )
    return {
        "photo_id": str(row.get("id", "")),
        "owner_nsid": str(row.get("owner", "")),
        "owner_name": str(row.get("ownername", "")),
        "license_id": int(row.get("license", -1)),
        "media": str(row.get("media", "")),
        "title": str(row.get("title", "")),
        "description": str(description or ""),
        "tags": str(row.get("tags", "")),
        "date_taken": str(row.get("datetaken", "")),
        "stock_id": stock_ids[0],
        "current_portra_explicit": stock_ids[0] == "kodak_portra_400_current",
        "disqualifying_alteration_terms": alteration_terms,
    }


def audit_metadata_payloads(
    payloads: Sequence[dict[str, Any]],
    baseline_payload: bytes,
    config: dict[str, Any],
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    """Audit frozen Flickr API pages without requesting media URLs or pixels."""

    source = config["source"]
    if len(payloads) != int(source["api_pages"]):
        raise FlickrCurrentPortraSourceError("Flickr API page count drift")
    ordered_payloads = list(reversed(payloads)) if reverse else list(payloads)
    rows: list[dict[str, Any]] = []
    totals: set[int] = set()
    pages: set[int] = set()
    for payload in ordered_payloads:
        if payload.get("stat") != "ok" or not isinstance(payload.get("photos"), dict):
            raise FlickrCurrentPortraSourceError("invalid Flickr API payload")
        photos = payload["photos"]
        totals.add(int(photos.get("total", -1)))
        pages.add(int(photos.get("page", -1)))
        page_rows = photos.get("photo")
        if not isinstance(page_rows, list):
            raise FlickrCurrentPortraSourceError("Flickr page rows are unavailable")
        rows.extend(page_rows)
    if totals != {int(source["expected_public_photo_count"])}:
        raise FlickrCurrentPortraSourceError("public photo count drift")
    if pages != set(range(1, int(source["api_pages"]) + 1)):
        raise FlickrCurrentPortraSourceError("Flickr API page identity drift")
    photo_ids = [str(row.get("id", "")) for row in rows]
    if any(not photo_id.isdigit() for photo_id in photo_ids):
        raise FlickrCurrentPortraSourceError("malformed photo identity")
    if len(photo_ids) != len(set(photo_ids)):
        raise FlickrCurrentPortraSourceError("duplicate photo identities")
    if len(rows) != int(source["expected_public_photo_count"]):
        raise FlickrCurrentPortraSourceError("incomplete public photo inventory")

    allowed = {int(value) for value in source["allowed_license_ids"]}
    targets = []
    for row in rows:
        record = classify_target_row(row, config)
        if record is not None and record["license_id"] in allowed:
            targets.append(record)
    targets.sort(key=lambda row: row["photo_id"])
    manifest_sha256 = _sha256(canonical_bytes(targets))
    stock_counts = dict(sorted(Counter(row["stock_id"] for row in targets).items()))
    unaltered_counts = dict(
        sorted(
            Counter(
                row["stock_id"]
                for row in targets
                if not row["disqualifying_alteration_terms"]
            ).items()
        )
    )

    baseline = config["baseline"]
    if len(baseline_payload) != int(baseline["bytes"]):
        raise FlickrCurrentPortraSourceError("A3Q evidence size drift")
    if _sha256(baseline_payload) != baseline["sha256"]:
        raise FlickrCurrentPortraSourceError("A3Q evidence hash drift")
    baseline_report = json.loads(baseline_payload)
    baseline_authors = list(
        baseline_report["portra_generation"][
            "qualifying_current_portra_three_stock_author_uids"
        ]
    )
    if len(baseline_authors) != int(baseline["required_qualifying_author_count"]):
        raise FlickrCurrentPortraSourceError("A3Q qualifying author count drift")
    distinct_new_author = source["owner_nsid"] not in baseline_authors
    augmented_authors = sorted([*baseline_authors, source["owner_nsid"]])

    expected_counts = source["expected_target_counts"]
    source_gates = {
        "complete_public_metadata_inventory": len(rows)
        == int(source["expected_public_photo_count"]),
        "target_manifest_sha256_exact": manifest_sha256
        == source["expected_target_manifest_sha256"],
        "target_stock_counts_exact": stock_counts == expected_counts,
        "one_owner_exact": all(
            row["owner_nsid"] == source["owner_nsid"] for row in targets
        ),
        "owner_name_exact": all(
            row["owner_name"] == source["author_name"] for row in targets
        ),
        "cc_by_2_license_exact": all(row["license_id"] in allowed for row in targets),
        "public_photo_media_exact": all(row["media"] == "photo" for row in targets),
        "current_portra_rows_explicit": all(
            row["current_portra_explicit"]
            for row in targets
            if row["stock_id"] == "kodak_portra_400_current"
        ),
        "new_author_distinct_from_a3q": distinct_new_author,
    }
    connectivity_gates = {
        "a3q_baseline_four_exact": len(baseline_authors)
        == int(baseline["required_qualifying_author_count"]),
        "new_author_has_all_three_stocks": set(stock_counts) == set(expected_counts),
        "augmented_current_portra_author_minimum": len(augmented_authors)
        >= int(baseline["required_total_after_addition"]),
    }

    minimums = config["admission_minimums"]
    group_evidence = config["group_evidence"]
    pixel_admission_gates = {
        "unaltered_rows_per_stock": all(
            unaltered_counts.get(stock_id, 0)
            >= int(minimums["unaltered_rows_per_stock"])
            for stock_id in expected_counts
        ),
        **{
            key: int(group_evidence[key]) >= int(minimums[key])
            for key in sorted(group_evidence)
        },
    }
    operation_counts = dict(config["operation_limits"])
    operation_gates = {
        "metadata_request_budget_exact": operation_counts["author_html_get_requests"]
        == 1
        and operation_counts["api_metadata_get_requests"] == int(source["api_pages"]),
        "zero_media_exif_pixel_fit_render_score": all(
            operation_counts[key] == 0
            for key in (
                "image_url_requests",
                "image_head_requests",
                "image_range_requests",
                "image_body_requests",
                "exif_requests",
                "pixel_decodes",
                "fit_calls",
                "render_calls",
                "score_calls",
            )
        ),
    }
    pixel_admission_passed = all(
        (
            *source_gates.values(),
            *connectivity_gates.values(),
            *pixel_admission_gates.values(),
            *operation_gates.values(),
        )
    )
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a4b-flickr-current-portra-all-three-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config[
            "decision_if_pixel_admission_passes"
            if pixel_admission_passed
            else "decision_if_pixel_admission_fails"
        ],
        "source": {
            "author_name": source["author_name"],
            "owner_nsid": source["owner_nsid"],
            "license_label": source["license_label"],
            "public_photo_count": len(rows),
            "api_pages": int(source["api_pages"]),
        },
        "target_manifest": {
            "rows": targets,
            "sha256": manifest_sha256,
            "stock_counts": stock_counts,
            "unaltered_stock_counts": unaltered_counts,
        },
        "connectivity": {
            "baseline_qualifying_author_uids": baseline_authors,
            "new_author_distinct": distinct_new_author,
            "augmented_qualifying_author_uids": augmented_authors,
            "augmented_qualifying_author_count": len(augmented_authors),
        },
        "group_evidence": group_evidence,
        "admission_minimums": minimums,
        "source_gates": source_gates,
        "connectivity_gates": connectivity_gates,
        "pixel_admission_gates": pixel_admission_gates,
        "operation_counts": operation_counts,
        "operation_gates": operation_gates,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _sha256(canonical_bytes(report))
    return report


def fetch_metadata_payloads(
    config: dict[str, Any], timeout_seconds: float = 30.0
) -> list[dict[str, Any]]:
    """Fetch only the author page and official metadata API pages."""

    source = config["source"]
    headers = {"User-Agent": "K-MCFM-source-audit/1.0"}
    request = urllib.request.Request(source["author_page_url"], headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        html = response.read().decode("utf-8", "replace")
    match = _SITE_KEY.search(html)
    if match is None:
        raise FlickrCurrentPortraSourceError("Flickr site key is unavailable")
    payloads: list[dict[str, Any]] = []
    for page in range(1, int(source["api_pages"]) + 1):
        query = urlencode(
            {
                "method": source["api_method"],
                "api_key": match.group(1),
                "user_id": source["owner_nsid"],
                "extras": "license,description,tags,date_taken,owner_name,media",
                "per_page": int(source["per_page"]),
                "page": page,
                "content_type": 1,
                "media": "photos",
                "format": "json",
                "nojsoncallback": 1,
            }
        )
        api_request = urllib.request.Request(f"{API_ENDPOINT}?{query}", headers=headers)
        with urllib.request.urlopen(api_request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise FlickrCurrentPortraSourceError("Flickr response is not an object")
        payloads.append(payload)
    return payloads


def run_flickr_current_portra_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    payload_fetcher: PayloadFetcher | None = None,
) -> dict[str, Any]:
    """Run the frozen metadata-only source audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    payloads = (
        list(payload_fetcher(config))
        if payload_fetcher is not None
        else fetch_metadata_payloads(config)
    )
    root = config_path.resolve().parents[1]
    baseline_payload = (root / config["baseline"]["path"]).read_bytes()
    return audit_metadata_payloads(payloads, baseline_payload, config, reverse=reverse)


__all__ = [
    "FlickrCurrentPortraSourceError",
    "audit_metadata_payloads",
    "canonical_bytes",
    "classify_target_row",
    "fetch_metadata_payloads",
    "run_flickr_current_portra_source_audit",
]
