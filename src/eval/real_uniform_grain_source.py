"""Bounded source and acquisition contract for real uniform film-grain scans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class UniformGrainSourceError(RuntimeError):
    """Raised when the frozen uniform-grain source contract is violated."""


_SHA1 = re.compile(r"[0-9a-f]{40}")
_ALLOWED_HOST = "upload.wikimedia.org"


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the repository's canonical evidence JSON representation."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_contract(config: Mapping[str, Any]) -> None:
    """Validate the exact eight-file source freeze without network access."""
    if config.get("schema") != "neuro_film.u6_p4r_uniform_grain_source.v1":
        raise UniformGrainSourceError("unsupported source schema")
    rows = config.get("files")
    if not isinstance(rows, list) or not rows:
        raise UniformGrainSourceError("source files are missing")
    expected_count = int(config["selection"]["exact_file_count"])
    if len(rows) != expected_count:
        raise UniformGrainSourceError("exact_file_count mismatch")
    titles = [str(row["title"]) for row in rows]
    if len(set(titles)) != len(titles):
        raise UniformGrainSourceError("duplicate Commons title")
    stocks = Counter(str(row["film_stock_id"]) for row in rows)
    if stocks != Counter(config["selection"]["expected_rows_per_stock"]):
        raise UniformGrainSourceError("stock support mismatch")
    if len({str(row["uploader"]) for row in rows}) != 1:
        raise UniformGrainSourceError("source nuisance must remain one uploader")
    if len({str(row["scanner"]) for row in rows}) != 1:
        raise UniformGrainSourceError("scanner nuisance must remain fixed")
    total_bytes = sum(int(row["expected_bytes"]) for row in rows)
    if total_bytes != int(config["selection"]["expected_total_bytes"]):
        raise UniformGrainSourceError("expected_total_bytes mismatch")
    if total_bytes > int(config["selection"]["maximum_total_bytes"]):
        raise UniformGrainSourceError("source exceeds bounded byte budget")
    for row in rows:
        title = str(row["title"])
        if not title.startswith("File:Kodak") or not title.endswith(".tif"):
            raise UniformGrainSourceError("unexpected Commons file title")
        if not _SHA1.fullmatch(str(row["api_sha1"]).casefold()):
            raise UniformGrainSourceError("invalid API SHA-1")
        parsed = urllib.parse.urlparse(str(row["original_url"]))
        if parsed.scheme != "https" or parsed.hostname != _ALLOWED_HOST:
            raise UniformGrainSourceError("original URL leaves frozen host")
        if str(row["license_short_name"]) != "CC0":
            raise UniformGrainSourceError("source must remain exact CC0")
        if str(row["mime"]) != "image/tiff":
            raise UniformGrainSourceError("source must remain TIFF")
        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise UniformGrainSourceError("invalid source dimensions")
        if str(row["process_type"]) != "unknown":
            raise UniformGrainSourceError("process_type cannot be inferred")
        if str(row["physical_roll"]) != "unknown":
            raise UniformGrainSourceError("physical_roll cannot be inferred")
    if config.get("training_allowed") or config.get("stock_calibration_allowed"):
        raise UniformGrainSourceError("source feasibility cannot open training/calibration")


def _request_json(
    endpoint: str,
    parameters: Mapping[str, str],
    user_agent: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint + "?" + urllib.parse.urlencode(parameters),
        headers={"User-Agent": user_agent},
    )
    for delay in (0, 2, 5, 10):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise UniformGrainSourceError("Commons API returned non-object JSON")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or delay == 10:
                raise
    raise UniformGrainSourceError("unreachable Commons retry state")


def normalize_api_payload(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate live imageinfo against the exact frozen source rows."""
    validate_contract(config)
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        raise UniformGrainSourceError("Commons API pages are missing")
    by_title = {str(page.get("title", "")): page for page in pages}
    expected_titles = {str(row["title"]) for row in config["files"]}
    if set(by_title) != expected_titles:
        raise UniformGrainSourceError("Commons title set drift")
    normalized: list[dict[str, Any]] = []
    for expected in sorted(config["files"], key=lambda row: str(row["title"])):
        page = by_title[str(expected["title"])]
        infos = page.get("imageinfo", [])
        if page.get("missing") or not isinstance(infos, list) or len(infos) != 1:
            raise UniformGrainSourceError("Commons imageinfo is missing or ambiguous")
        info = infos[0]
        metadata = info.get("extmetadata", {})
        license_name = str(metadata.get("LicenseShortName", {}).get("value", ""))
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
            "license_short_name": license_name,
            "license_url": str(metadata.get("LicenseUrl", {}).get("value", "")),
            "usage_terms": str(metadata.get("UsageTerms", {}).get("value", "")),
        }
        for key in (
            "original_url",
            "api_sha1",
            "expected_bytes",
            "width",
            "height",
            "mime",
            "uploader",
            "license_short_name",
        ):
            if observed[key] != expected[key]:
                raise UniformGrainSourceError(
                    f"Commons metadata drift for {observed['title']}: {key}"
                )
        normalized.append(observed)
    return normalized


def fetch_metadata_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch exactly one bounded current-imageinfo response and audit it."""
    validate_contract(config)
    titles = "|".join(str(row["title"]) for row in config["files"])
    payload = _request_json(
        str(config["source"]["api_endpoint"]),
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "titles": titles,
            "iiprop": (
                "url|size|sha1|mime|mediatype|timestamp|user|extmetadata"
            ),
            "iilimit": "1",
        },
        str(config["source"]["user_agent"]),
    )
    rows = normalize_api_payload(payload, config)
    return {
        "schema": "neuro_film.u6_p4r_uniform_grain_metadata_snapshot.v1",
        "source_contract_id": config["experiment_id"],
        "request_count": 1,
        "image_payloads_downloaded_or_decoded": False,
        "rows": rows,
    }


def _download_exact(row: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row["expected_bytes"])
    expected_sha1 = str(row["api_sha1"]).casefold()
    if destination.is_file():
        if (
            destination.stat().st_size == expected_size
            and hash_file(destination, "sha1") == expected_sha1
        ):
            return {
                "path": str(destination),
                "bytes": expected_size,
                "sha1": expected_sha1,
                "sha256": hash_file(destination, "sha256"),
                "reused": True,
            }
        raise UniformGrainSourceError("existing destination fails frozen identity")
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise UniformGrainSourceError("stale partial download requires inspection")
    request = urllib.request.Request(
        str(row["original_url"]),
        headers={"User-Agent": "neuro-film-u6-p4r/1.0"},
    )
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > expected_size:
                        raise UniformGrainSourceError("download exceeds exact byte size")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if size != expected_size or hash_file(temporary, "sha1") != expected_sha1:
            raise UniformGrainSourceError("download identity mismatch")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "path": str(destination),
        "bytes": size,
        "sha1": expected_sha1,
        "sha256": hash_file(destination, "sha256"),
        "reused": False,
    }


def acquire_files(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Acquire only the frozen eight exact payloads and return a local manifest."""
    validate_contract(config)
    rows: list[dict[str, Any]] = []
    for source in sorted(config["files"], key=lambda row: str(row["title"])):
        result = _download_exact(source, root / str(source["path"]))
        relative_path = (
            Path(str(result["path"])).resolve().relative_to(root.resolve())
        )
        rows.append(
            {
                **result,
                "path": relative_path.as_posix(),
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
        "schema": "neuro_film.u6_p4r_uniform_grain_acquisition_manifest.v1",
        "source_contract_id": config["experiment_id"],
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "rows": rows,
        "claim_ceiling": config["claim_ceiling"],
    }


def write_atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()
