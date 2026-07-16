"""Wikimedia Commons stock-category metadata snapshot and eligibility audit."""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


class CommonsStockSourceError(ValueError):
    """Raised when the SF0.4 metadata contract is invalid."""


_HTML_TAG = re.compile(r"<[^>]+>")


def normalize_author(value: str) -> str:
    """Normalize visible author text conservatively for source-group audits."""
    visible = html.unescape(_HTML_TAG.sub(" ", value or ""))
    return " ".join(visible.casefold().split())


def metadata_value(extmetadata: Mapping[str, Any], key: str) -> str:
    value = extmetadata.get(key, {})
    return str(value.get("value", "")) if isinstance(value, Mapping) else ""


def normalize_file_page(page: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one MediaWiki file page without discarding raw credit strings."""
    image_info = page.get("imageinfo")
    if not isinstance(image_info, list) or len(image_info) != 1:
        raise CommonsStockSourceError("file page requires exactly one imageinfo record")
    info = image_info[0]
    extmetadata = info.get("extmetadata", {})
    title = str(page.get("title", ""))
    if not title.startswith("File:"):
        raise CommonsStockSourceError("category member is not a file page")
    categories = sorted(
        str(row["title"])
        for row in page.get("categories", [])
        if isinstance(row, Mapping) and isinstance(row.get("title"), str)
    )
    required = ("url", "descriptionurl", "sha1", "width", "height", "size", "user")
    if any(info.get(key) in (None, "") for key in required):
        raise CommonsStockSourceError(f"incomplete imageinfo: {title}")
    return {
        "page_id": int(page["pageid"]),
        "title": title,
        "file_page_url": str(info["descriptionurl"]),
        "original_url": str(info["url"]),
        "derivative_1600_url": str(info.get("thumburl", "")),
        "api_sha1_base36": str(info["sha1"]),
        "byte_size": int(info["size"]),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "mime": str(info.get("mime", "")),
        "media_type": str(info.get("mediatype", "")),
        "uploader": str(info["user"]),
        "upload_timestamp": str(info.get("timestamp", "")),
        "author_raw_html": metadata_value(extmetadata, "Artist"),
        "credit_raw_html": metadata_value(extmetadata, "Credit"),
        "description_raw_html": metadata_value(extmetadata, "ImageDescription"),
        "date_time_original": metadata_value(extmetadata, "DateTimeOriginal"),
        "license_short_name": metadata_value(extmetadata, "LicenseShortName"),
        "license_url": metadata_value(extmetadata, "LicenseUrl"),
        "usage_terms": metadata_value(extmetadata, "UsageTerms"),
        "attribution_required": metadata_value(extmetadata, "AttributionRequired"),
        "categories": categories,
    }


def audit_snapshot(snapshot: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit an immutable snapshot; never access the network or decode pixels."""
    categories = snapshot.get("categories")
    if not isinstance(categories, list):
        raise CommonsStockSourceError("snapshot categories are missing")
    configured = {str(row["film_stock_id"]): row for row in config["categories"]}
    research_licenses = set(config["license_policy"]["research_free_licenses"])
    permissive_licenses = set(config["license_policy"]["permissive_candidate_licenses"])
    patterns = [re.compile(value, re.IGNORECASE) for value in config["metadata_flags"]["non_scene_title_patterns"]]
    gates = config["gates_for_exact_stock_pilot"]
    results: list[dict[str, Any]] = []
    for category in sorted(categories, key=lambda row: str(row["film_stock_id"])):
        stock_id = str(category["film_stock_id"])
        expected = configured.get(stock_id)
        if expected is None or category["category"] != expected["category"]:
            raise CommonsStockSourceError(f"unexpected category snapshot: {stock_id}")
        files = category.get("files")
        if not isinstance(files, list) or not files:
            raise CommonsStockSourceError(f"empty file snapshot: {stock_id}")
        titles = [str(row["title"]) for row in files]
        if len(set(titles)) != len(titles):
            raise CommonsStockSourceError(f"duplicate file title: {stock_id}")
        uploaders = Counter(str(row["uploader"]) for row in files)
        author_groups = Counter(
            normalize_author(str(row.get("author_raw_html", ""))) or "__missing_author__"
            for row in files
        )
        licenses = Counter(str(row["license_short_name"]) for row in files)
        permissive = sum(row["license_short_name"] in permissive_licenses for row in files)
        strict_derivative_rows = sum(
            row["license_short_name"] in permissive_licenses
            and bool(str(row.get("author_raw_html", "")).strip())
            and bool(str(row.get("file_page_url", "")).strip())
            and bool(str(row.get("original_url", "")).strip())
            and bool(str(row.get("derivative_1600_url", "")).strip())
            and row.get("derivative_1600_url") != row.get("original_url")
            and (
                bool(str(row.get("license_url", "")).strip())
                or (
                    row["license_short_name"] == "Public domain"
                    and bool(str(row.get("usage_terms", "")).strip())
                )
            )
            for row in files
        )
        minimum_dimension = sum(min(int(row["width"]), int(row["height"])) >= 512 for row in files) / len(files)
        title_flags = [row["title"] for row in files if any(pattern.search(str(row["title"])) for pattern in patterns)]
        largest_share = max(uploaders.values()) / len(files)
        largest_author_share = max(author_groups.values()) / len(files)
        unique_authors = len(author_groups) - int("__missing_author__" in author_groups)
        author_present = sum(bool(str(row.get("author_raw_html", "")).strip()) for row in files)
        checks = {
            "minimum_files": len(files) >= int(gates["minimum_files"]),
            "minimum_unique_uploaders": len(uploaders) >= int(gates["minimum_unique_uploaders"]),
            "largest_uploader_share": largest_share <= float(gates["maximum_largest_uploader_share"]),
            "minimum_permissive_candidate_files": permissive >= int(gates["minimum_permissive_candidate_files"]),
            "minimum_dimension_fraction": minimum_dimension >= float(gates["minimum_fraction_minimum_dimension_512"]),
            "non_scene_title_fraction": len(title_flags) / len(files) <= float(gates["maximum_non_scene_title_flag_fraction"]),
            "every_row_free_license": all(row["license_short_name"] in research_licenses for row in files),
            "every_row_source_url": all(bool(row["file_page_url"] and row["original_url"]) for row in files),
            "every_row_integrity_metadata": all(bool(row["api_sha1_base36"]) and row["width"] > 0 and row["height"] > 0 for row in files),
            "minimum_unique_normalized_authors": unique_authors >= int(gates.get("minimum_unique_normalized_authors", 0)),
            "largest_normalized_author_share": largest_author_share <= float(gates.get("maximum_largest_normalized_author_share", 1.0)),
            "minimum_strict_derivative_rights_rows": strict_derivative_rows >= int(gates.get("minimum_strict_derivative_rights_rows", 0)),
        }
        exact = expected["label_scope"] == "exact_stock_community_category"
        results.append({
            "film_stock_id": stock_id,
            "category": category["category"],
            "label_scope": expected["label_scope"],
            "files": len(files),
            "unique_uploaders": len(uploaders),
            "largest_uploader": uploaders.most_common(1)[0][0],
            "largest_uploader_share": largest_share,
            "permissive_candidate_files": permissive,
            "strict_derivative_rights_files": strict_derivative_rows,
            "unique_normalized_authors": unique_authors,
            "largest_normalized_author": author_groups.most_common(1)[0][0],
            "largest_normalized_author_share": largest_author_share,
            "minimum_dimension_512_fraction": minimum_dimension,
            "non_scene_title_flags": title_flags,
            "author_metadata_present_fraction": author_present / len(files),
            "licence_counts": dict(sorted(licenses.items())),
            "top_uploaders": dict(uploaders.most_common(10)),
            "top_normalized_authors": dict(author_groups.most_common(10)),
            "checks": checks,
            "metadata_gate_passed": all(checks.values()),
            "exact_stock_pilot_eligible": exact and all(checks.values()),
        })
    missing = sorted(set(configured) - {row["film_stock_id"] for row in results})
    if missing:
        raise CommonsStockSourceError(f"configured categories missing from snapshot: {missing}")
    exact_passes = sum(row["exact_stock_pilot_eligible"] for row in results)
    conditional = config.get("conditional_pixel_pilot", {})
    required_exact_passes = (
        2
        if conditional.get("allowed_only_if_at_least_two_new_exact_stocks_pass_metadata_gates")
        else 3
    )
    pixel_allowed = exact_passes >= required_exact_passes
    return {
        "category_results": results,
        "exact_stock_metadata_passes": exact_passes,
        "required_exact_stock_metadata_passes": required_exact_passes,
        "conditional_pixel_pilot_allowed": pixel_allowed,
        "decision": "metadata_pass_freeze_bounded_pixel_pilot" if pixel_allowed else "insufficient_exact_stock_metadata_support",
        "image_payloads_downloaded_or_decoded": False,
    }
