"""Resumable acquisition and exact-stock filtering for the full YFCC100M index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import requests

from src.real_film.yfcc_stock_source import atomic_json


class YfccFullIndexError(ValueError):
    """Raised when the frozen SF1.1 source or scan contract fails closed."""


def hash_file_evidence(path: Path, multipart_part_bytes: int) -> dict[str, Any]:
    """Compute SHA-256 and an S3 multipart ETag in one sequential pass."""
    if multipart_part_bytes <= 0:
        raise YfccFullIndexError("multipart part size must be positive")
    sha256 = hashlib.sha256()
    part_digests: list[bytes] = []
    with path.open("rb") as handle:
        while chunk := handle.read(multipart_part_bytes):
            sha256.update(chunk)
            part_digests.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    if not part_digests:
        raise YfccFullIndexError("cannot hash an empty multipart object")
    multipart = hashlib.md5(b"".join(part_digests), usedforsecurity=False).hexdigest()
    return {
        "sha256": sha256.hexdigest(),
        "multipart_etag": f"{multipart}-{len(part_digests)}",
        "multipart_parts": len(part_digests),
        "multipart_part_bytes": multipart_part_bytes,
    }


def validate_source_headers(headers: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    content_length = int(headers.get("Content-Length", "-1"))
    etag = str(headers.get("ETag", "")).strip('"')
    last_modified = str(headers.get("Last-Modified", ""))
    accept_ranges = str(headers.get("Accept-Ranges", ""))
    if content_length != int(source["expected_bytes"]):
        raise YfccFullIndexError("source Content-Length drifted")
    if etag != str(source["expected_etag"]):
        raise YfccFullIndexError("source ETag drifted")
    if last_modified != str(source["expected_last_modified"]):
        raise YfccFullIndexError("source Last-Modified drifted")
    expected_ranges = source.get("expected_accept_ranges")
    if expected_ranges is not None and accept_ranges.casefold() != str(expected_ranges).casefold():
        raise YfccFullIndexError("source Accept-Ranges drifted")
    return {
        "content_length": content_length,
        "etag": etag,
        "last_modified": last_modified,
        "accept_ranges": accept_ranges or None,
    }


def validate_sqlite_header(path: Path, expected_bytes: int) -> None:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise YfccFullIndexError("SQLite byte size mismatch")
    with path.open("rb") as handle:
        if handle.read(16) != b"SQLite format 3\x00":
            raise YfccFullIndexError("SQLite header mismatch")


def validate_download_manifest(
    manifest: Mapping[str, Any], config: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    """Bind an audit to the hash-complete downloader manifest and local SQLite."""
    expected_bytes = int(config["source"]["expected_bytes"])
    validate_sqlite_header(path, expected_bytes)
    if manifest.get("dataset_id") != config["dataset_id"]:
        raise YfccFullIndexError("download manifest dataset drifted")
    if int(manifest.get("bytes", -1)) != expected_bytes:
        raise YfccFullIndexError("download manifest byte size drifted")
    if Path(str(manifest.get("path", ""))).resolve() != path.resolve():
        raise YfccFullIndexError("download manifest path drifted")
    source = manifest.get("source", {})
    validate_source_headers(
        {
            "Content-Length": str(source.get("content_length", -1)),
            "ETag": str(source.get("etag", "")),
            "Last-Modified": str(source.get("last_modified", "")),
            "Accept-Ranges": str(source.get("accept_ranges", "")),
        },
        config,
    )
    digest = str(manifest.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise YfccFullIndexError("download manifest SHA-256 is invalid")
    if manifest.get("multipart_etag_verified") is not True:
        raise YfccFullIndexError("download manifest lacks multipart ETag verification")
    if manifest.get("multipart_etag") != config["source"]["expected_etag"]:
        raise YfccFullIndexError("download manifest multipart ETag drifted")
    if int(manifest.get("multipart_parts", -1)) != int(config["source"]["expected_multipart_parts"]):
        raise YfccFullIndexError("download manifest multipart part count drifted")
    if int(manifest.get("multipart_part_bytes", -1)) != int(config["source"]["multipart_part_bytes"]):
        raise YfccFullIndexError("download manifest multipart part size drifted")
    if manifest.get("image_payloads_downloaded_or_decoded") is not False:
        raise YfccFullIndexError("download manifest violates metadata-only contract")
    return {
        "sqlite_sha256": digest,
        "sqlite_bytes": expected_bytes,
        "source": source,
    }


def download_full_index(
    config: Mapping[str, Any],
    destination: Path,
    *,
    progress_path: Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Resume the immutable S3 object and atomically promote only a valid SQLite file."""
    client = session or requests.Session()
    client.headers["User-Agent"] = str(config["user_agent"])
    source = config["source"]
    limits = config["download_limits"]
    timeout = (
        float(limits["connect_timeout_seconds"]),
        float(limits["read_timeout_seconds"]),
    )
    head = client.head(str(source["url"]), timeout=timeout, allow_redirects=True)
    head.raise_for_status()
    evidence = validate_source_headers(head.headers, config)
    head.close()
    expected_bytes = int(source["expected_bytes"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_sqlite_header(destination, expected_bytes)
    else:
        temporary = destination.with_suffix(destination.suffix + ".part")
        current = temporary.stat().st_size if temporary.exists() else 0
        if current > expected_bytes:
            raise YfccFullIndexError("partial file exceeds frozen source size")
        free = shutil.disk_usage(destination.parent).free
        remaining = expected_bytes - current
        if free - remaining < int(limits["minimum_free_space_after_download_bytes"]):
            raise YfccFullIndexError("insufficient post-download free-space headroom")
        retries = int(limits["request_retries"])
        while current < expected_bytes:
            request_headers = {"Range": f"bytes={current}-"} if current else {}
            last_error: Exception | None = None
            for attempt in range(retries):
                response: requests.Response | None = None
                try:
                    response = client.get(
                        str(source["url"]), headers=request_headers,
                        timeout=timeout, stream=True, allow_redirects=True,
                    )
                    expected_status = 206 if current else 200
                    if response.status_code != expected_status:
                        raise YfccFullIndexError(f"unexpected resume status: {response.status_code}")
                    mode = "ab" if current else "wb"
                    checkpoint_bytes = int(limits["progress_checkpoint_bytes"])
                    next_checkpoint = current + checkpoint_bytes
                    with temporary.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=int(limits["chunk_bytes"])):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            current += len(chunk)
                            if current > expected_bytes:
                                raise YfccFullIndexError("download exceeded frozen byte ceiling")
                            if progress_path is not None and current >= next_checkpoint:
                                atomic_json(progress_path, {
                                    "schema_version": 1, "complete": False,
                                    "bytes": current, "expected_bytes": expected_bytes,
                                    "fraction": current / expected_bytes,
                                    "source": evidence,
                                })
                                next_checkpoint = current + checkpoint_bytes
                    last_error = None
                    break
                except (OSError, requests.RequestException, YfccFullIndexError) as exc:
                    last_error = exc
                    current = temporary.stat().st_size if temporary.exists() else 0
                    request_headers = {"Range": f"bytes={current}-"} if current else {}
                    if attempt + 1 < retries:
                        time.sleep(float(limits["retry_backoff_seconds"]) * (2 ** attempt))
                finally:
                    if response is not None:
                        response.close()
            if last_error is not None:
                raise YfccFullIndexError("full-index download failed after retries") from last_error
        validate_sqlite_header(temporary, expected_bytes)
        os.replace(temporary, destination)
    multipart_part_bytes = int(source["multipart_part_bytes"])
    file_evidence = hash_file_evidence(destination, multipart_part_bytes)
    if file_evidence["multipart_etag"] != str(source["expected_etag"]):
        raise YfccFullIndexError("source multipart ETag verification failed")
    if file_evidence["multipart_parts"] != int(source["expected_multipart_parts"]):
        raise YfccFullIndexError("source multipart part count drifted")
    with sqlite3.connect(f"file:{destination.as_posix()}?mode=ro", uri=True) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='yfcc100m_dataset'"
        ).fetchone()
        if table is None:
            raise YfccFullIndexError("expected YFCC table is missing")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(yfcc100m_dataset)")]
    required = set(config["scan"]["required_columns"])
    if not required.issubset(columns):
        raise YfccFullIndexError("YFCC table schema is incomplete")
    result = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": file_evidence["sha256"],
        "multipart_etag": file_evidence["multipart_etag"],
        "multipart_parts": file_evidence["multipart_parts"],
        "multipart_part_bytes": file_evidence["multipart_part_bytes"],
        "multipart_etag_verified": True,
        "source": evidence,
        "sqlite_table": "yfcc100m_dataset",
        "sqlite_columns": columns,
        "image_payloads_downloaded_or_decoded": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    if progress_path is not None:
        atomic_json(progress_path, {**result, "complete": True})
    return result


def _normalized_text(row: Mapping[str, Any]) -> str:
    value = unquote_plus(" ".join(str(row.get(key) or "") for key in ("title", "description", "usertags"))).casefold()
    return re.sub(r"[+_-]+", " ", value)


def audit_candidate_rows(rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    patterns = [(str(row["film_stock_id"]), re.compile(str(row["exact_regex"]), re.IGNORECASE)) for row in config["stock_patterns"]]
    matches: list[dict[str, Any]] = []
    ambiguous_matches: list[dict[str, Any]] = []
    missing_uid_rows: list[dict[str, Any]] = []
    seen_photoids: set[int] = set()
    for source in rows:
        photoid = int(source["photoid"])
        if photoid in seen_photoids:
            raise YfccFullIndexError(f"duplicate photoid in full-index scan: {photoid}")
        seen_photoids.add(photoid)
        text = _normalized_text(source)
        matched_stock_ids = [stock_id for stock_id, pattern in patterns if pattern.search(text)]
        uid = source.get("uid")
        if matched_stock_ids and (uid is None or not str(uid).strip()):
            missing_uid_rows.append({"matched_stock_ids": matched_stock_ids, **dict(source)})
            continue
        if len(matched_stock_ids) == 1:
            matches.append({"film_stock_id": matched_stock_ids[0], **dict(source)})
        elif len(matched_stock_ids) > 1:
            ambiguous_matches.append({"matched_stock_ids": matched_stock_ids, **dict(source)})
    matches.sort(
        key=lambda row: (
            str(row["film_stock_id"]),
            str(row["uid"]).casefold(),
            int(row["photoid"]),
        )
    )
    ambiguous_matches.sort(key=lambda row: (str(row["uid"]).casefold(), int(row["photoid"])))
    missing_uid_rows.sort(key=lambda row: int(row["photoid"]))
    results: dict[str, Any] = {}
    for stock_id, _ in patterns:
        stock_rows = [row for row in matches if row["film_stock_id"] == stock_id]
        uids = Counter(str(row["uid"]) for row in stock_rows)
        results[stock_id] = {
            "rows": len(stock_rows),
            "author_uids": len(uids),
            "largest_author_share": max(uids.values(), default=0) / max(len(stock_rows), 1),
        }
    overlap_gates: dict[str, Any] = {}
    for gate in config["shared_author_gates"]:
        left = {str(row["uid"]) for row in matches if row["film_stock_id"] == gate["left_stock_id"]}
        right = {str(row["uid"]) for row in matches if row["film_stock_id"] == gate["right_stock_id"]}
        shared = sorted(left & right)
        checks = {
            "minimum_left_rows": results[gate["left_stock_id"]]["rows"] >= int(gate["minimum_rows_each_stock"]),
            "minimum_right_rows": results[gate["right_stock_id"]]["rows"] >= int(gate["minimum_rows_each_stock"]),
            "minimum_left_uids": results[gate["left_stock_id"]]["author_uids"] >= int(gate["minimum_uids_each_stock"]),
            "minimum_right_uids": results[gate["right_stock_id"]]["author_uids"] >= int(gate["minimum_uids_each_stock"]),
            "minimum_shared_uids": len(shared) >= int(gate["minimum_shared_uids"]),
        }
        overlap_gates[str(gate["gate_id"])] = {
            "left_stock_id": gate["left_stock_id"], "right_stock_id": gate["right_stock_id"],
            "shared_uids": shared, "shared_uid_count": len(shared),
            "checks": checks, "metadata_gate_passed": all(checks.values()),
        }
    return {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "candidate_results": results,
        "shared_author_results": overlap_gates,
        "matches": matches,
        "ambiguous_multi_stock_rows": ambiguous_matches,
        "ambiguous_multi_stock_row_count": len(ambiguous_matches),
        "missing_uid_rows": missing_uid_rows,
        "missing_uid_row_count": len(missing_uid_rows),
        "any_shared_author_gate_passed": any(row["metadata_gate_passed"] for row in overlap_gates.values()),
        "image_payloads_downloaded_or_decoded": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def scan_full_index(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one broad SQL prefilter and exact-regex audit without loading pixels."""
    scan = config["scan"]
    columns = list(scan["required_columns"])
    text_expression = "lower(coalesce(title,'') || ' ' || coalesce(description,'') || ' ' || coalesce(usertags,''))"
    broad = [str(value).casefold() for value in scan["broad_prefilter_terms"]]
    where = " OR ".join(f"instr({text_expression}, ?) > 0" for _ in broad)
    licence_placeholders = ",".join("?" for _ in config["rights_filter"]["allowed_license_urls"])
    query = (
        f"SELECT {','.join(columns)} FROM yfcc100m_dataset "
        f"WHERE marker=0 AND licenseurl IN ({licence_placeholders}) AND ({where})"
    )
    parameters = [*config["rights_filter"]["allowed_license_urls"], *broad]
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        cursor = connection.execute(query, parameters)
        rows = (dict(zip(columns, row, strict=True)) for row in cursor)
        return audit_candidate_rows(rows, config)
