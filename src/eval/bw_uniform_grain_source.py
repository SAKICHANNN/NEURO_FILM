"""Bounded source contract for new B&W uniform film-grain scans."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import re
import urllib.parse
from typing import Any

from src.eval.real_uniform_grain_source import (
    _download_exact,
    _request_json,
    hash_file,
)


class BWUniformGrainSourceError(RuntimeError):
    """Raised when the frozen B&W source contract is violated."""


_SHA1 = re.compile(r"[0-9a-f]{40}")
_ALLOWED_HOST = "upload.wikimedia.org"


def validate_contract(config: Mapping[str, Any]) -> None:
    """Validate the exact three-file freeze without network access."""
    if config.get("schema") != "neuro_film.u6_p4z_bw_uniform_grain_source.v1":
        raise BWUniformGrainSourceError("unsupported source schema")
    rows = config.get("files")
    if not isinstance(rows, list) or not rows:
        raise BWUniformGrainSourceError("source files are missing")
    selection = config["selection"]
    if len(rows) != int(selection["exact_file_count"]):
        raise BWUniformGrainSourceError("exact_file_count mismatch")
    titles = [str(row["title"]) for row in rows]
    if len(set(titles)) != len(titles):
        raise BWUniformGrainSourceError("duplicate Commons title")
    stocks = Counter(str(row["film_stock_id"]) for row in rows)
    if stocks != Counter(selection["expected_rows_per_stock"]):
        raise BWUniformGrainSourceError("stock support mismatch")
    if len({str(row["uploader"]) for row in rows}) != 1:
        raise BWUniformGrainSourceError("source nuisance must remain one uploader")
    if len({str(row["scanner"]) for row in rows}) != 1:
        raise BWUniformGrainSourceError("scanner nuisance must remain fixed")
    total_bytes = sum(int(row["expected_bytes"]) for row in rows)
    if total_bytes != int(selection["expected_total_bytes"]):
        raise BWUniformGrainSourceError("expected_total_bytes mismatch")
    if total_bytes > int(selection["maximum_total_bytes"]):
        raise BWUniformGrainSourceError("source exceeds bounded byte budget")
    expected_titles = {
        "File:Tmax100 Grain.png",
        "File:Tri-x 400 grain1.png",
        "File:Tri-x 400 grain2.png",
    }
    if set(titles) != expected_titles:
        raise BWUniformGrainSourceError("unexpected Commons file set")
    for row in rows:
        if not _SHA1.fullmatch(str(row["api_sha1"]).casefold()):
            raise BWUniformGrainSourceError("invalid API SHA-1")
        parsed = urllib.parse.urlparse(str(row["original_url"]))
        if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
            raise BWUniformGrainSourceError("original URL leaves frozen host")
        if str(row["license_short_name"]) != "CC0":
            raise BWUniformGrainSourceError("source must remain exact CC0")
        if str(row["mime"]) != "image/png":
            raise BWUniformGrainSourceError("source must remain PNG")
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise BWUniformGrainSourceError("invalid source dimensions")
        if str(row["process_type"]) != "unknown":
            raise BWUniformGrainSourceError("process_type cannot be inferred")
        if str(row["physical_roll"]) != "unknown":
            raise BWUniformGrainSourceError("physical_roll cannot be inferred")
    if (
        config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("stock_calibration_allowed")
    ):
        raise BWUniformGrainSourceError(
            "source feasibility cannot open fitting/training/calibration"
        )


def normalize_api_payload(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate live Commons imageinfo against the exact frozen rows."""
    validate_contract(config)
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        raise BWUniformGrainSourceError("Commons API pages are missing")
    by_title = {str(page.get("title", "")): page for page in pages}
    expected_titles = {str(row["title"]) for row in config["files"]}
    if set(by_title) != expected_titles:
        raise BWUniformGrainSourceError("Commons title set drift")
    normalized: list[dict[str, Any]] = []
    for expected in sorted(config["files"], key=lambda row: str(row["title"])):
        page = by_title[str(expected["title"])]
        infos = page.get("imageinfo", [])
        if page.get("missing") or not isinstance(infos, list) or len(infos) != 1:
            raise BWUniformGrainSourceError(
                "Commons imageinfo is missing or ambiguous"
            )
        info = infos[0]
        metadata = info.get("extmetadata", {})
        observed = {
            "title": str(page["title"]),
            "page_id": int(page["pageid"]),
            "original_url": str(info["url"]),
            "description_url": str(info["descriptionurl"]),
            "api_sha1": str(info["sha1"]).casefold(),
            "expected_bytes": int(info["size"]),
            "width": int(info["width"]),
            "height": int(info["height"]),
            "mime": str(info["mime"]),
            "uploader": str(info["user"]),
            "upload_timestamp": str(info["timestamp"]),
            "license_short_name": str(
                metadata.get("LicenseShortName", {}).get("value", "")
            ),
            "license_url": str(
                metadata.get("LicenseUrl", {}).get("value", "")
            ),
            "usage_terms": str(
                metadata.get("UsageTerms", {}).get("value", "")
            ),
        }
        for key in (
            "original_url",
            "api_sha1",
            "expected_bytes",
            "width",
            "height",
            "mime",
            "uploader",
            "upload_timestamp",
            "license_short_name",
        ):
            if observed[key] != expected[key]:
                raise BWUniformGrainSourceError(
                    f"Commons metadata drift for {observed['title']}: {key}"
                )
        normalized.append(observed)
    return normalized


def fetch_metadata_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch one bounded current-imageinfo response and validate it."""
    validate_contract(config)
    payload = _request_json(
        str(config["source"]["api_endpoint"]),
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "titles": "|".join(str(row["title"]) for row in config["files"]),
            "iiprop": (
                "url|size|sha1|mime|mediatype|timestamp|user|extmetadata"
            ),
            "iilimit": "1",
        },
        str(config["source"]["user_agent"]),
    )
    return {
        "schema": "neuro_film.u6_p4z_bw_uniform_grain_metadata_snapshot.v1",
        "source_contract_id": config["experiment_id"],
        "request_count": 1,
        "image_payloads_downloaded_or_decoded": False,
        "rows": normalize_api_payload(payload, config),
    }


def acquire_files(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Acquire only the three frozen payloads."""
    validate_contract(config)
    rows: list[dict[str, Any]] = []
    for source in sorted(config["files"], key=lambda row: str(row["title"])):
        result = _download_exact(source, root / str(source["path"]))
        relative = Path(str(result["path"])).resolve().relative_to(root.resolve())
        rows.append(
            {
                **result,
                "path": relative.as_posix(),
                "title": source["title"],
                "film_stock_id": source["film_stock_id"],
                "uploader_group": source["uploader"],
                "scanner_group": source["scanner"],
                "physical_roll": "unknown",
                "process_type": "unknown",
                "allowed_use": source["allowed_use"],
            }
        )
    return {
        "schema": "neuro_film.u6_p4z_bw_uniform_grain_acquisition_manifest.v1",
        "source_contract_id": config["experiment_id"],
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "rows": rows,
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "BWUniformGrainSourceError",
    "acquire_files",
    "fetch_metadata_snapshot",
    "normalize_api_payload",
    "validate_contract",
]
