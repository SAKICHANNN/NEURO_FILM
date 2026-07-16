"""Bounded YFCC15M metadata acquisition and exact film-stock evidence scan."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class YfccStockSourceError(ValueError):
    """Raised when the frozen YFCC metadata contract fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def validate_parquet_index(payload: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the remote dataset-server index before downloading any shard."""
    if payload.get("pending") or payload.get("failed"):
        raise YfccStockSourceError("dataset-server index is incomplete")
    rows = payload.get("parquet_files")
    if not isinstance(rows, list):
        raise YfccStockSourceError("parquet_files is missing")
    expected_names = list(config["metadata_freeze"]["expected_filenames"])
    normalized = sorted(
        (
            {
                "filename": str(row["filename"]),
                "url": str(row["url"]),
                "size": int(row["size"]),
                "split": str(row["split"]),
            }
            for row in rows
            if str(row.get("split", "")) == config["metadata_freeze"]["split"]
        ),
        key=lambda row: row["filename"],
    )
    if [row["filename"] for row in normalized] != expected_names:
        raise YfccStockSourceError("parquet filename set drifted")
    if len(normalized) != int(config["metadata_freeze"]["expected_shards"]):
        raise YfccStockSourceError("parquet shard count drifted")
    if sum(row["size"] for row in normalized) != int(config["metadata_freeze"]["expected_total_bytes"]):
        raise YfccStockSourceError("parquet byte total drifted")
    if any(row["size"] <= 8 or not row["url"].startswith("https://") for row in normalized):
        raise YfccStockSourceError("unsafe parquet index row")
    return normalized


def fetch_parquet_index(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request = urllib.request.Request(
        str(config["metadata_freeze"]["index_url"]),
        headers={"User-Agent": str(config["user_agent"])},
    )
    with urllib.request.urlopen(request, timeout=float(config["download_limits"]["timeout_seconds"])) as response:
        payload = json.load(response)
    return payload, validate_parquet_index(payload, config)


def _valid_parquet(path: Path, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size or expected_size < 8:
        return False
    with path.open("rb") as handle:
        head = handle.read(4)
        handle.seek(-4, os.SEEK_END)
        tail = handle.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def download_parquet_shards(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Download the frozen metadata shards sequentially with atomic resume."""
    if sum(int(row["size"]) for row in rows) > int(config["download_limits"]["maximum_total_bytes"]):
        raise YfccStockSourceError("metadata download exceeds frozen byte ceiling")
    root.mkdir(parents=True, exist_ok=True)
    limits = config["download_limits"]
    results: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        destination = root / str(row["filename"])
        expected_size = int(row["size"])
        if not _valid_parquet(destination, expected_size):
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                temporary.unlink()
            request = urllib.request.Request(str(row["url"]), headers={"User-Agent": str(config["user_agent"])})
            last_error: Exception | None = None
            for attempt in range(int(limits["request_retries"])):
                if attempt:
                    time.sleep(float(limits["retry_backoff_seconds"]) * (2 ** (attempt - 1)))
                try:
                    with urllib.request.urlopen(request, timeout=float(limits["timeout_seconds"])) as response:
                        with temporary.open("wb") as handle:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                    if not _valid_parquet(temporary, expected_size):
                        raise YfccStockSourceError(f"invalid parquet payload: {row['filename']}")
                    os.replace(temporary, destination)
                    last_error = None
                    break
                except (OSError, urllib.error.URLError, YfccStockSourceError) as exc:
                    last_error = exc
                    if temporary.exists():
                        temporary.unlink()
            if last_error is not None:
                raise YfccStockSourceError(f"download failed: {row['filename']}") from last_error
            if position + 1 < len(rows):
                time.sleep(float(limits["request_interval_seconds"]))
        results.append({
            "filename": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "source_url": str(row["url"]),
        })
    return {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "files": results,
        "total_bytes": sum(row["bytes"] for row in results),
        "image_payloads_downloaded_or_decoded": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def scan_stock_candidates(paths: Sequence[Path], config: Mapping[str, Any]) -> dict[str, Any]:
    """Scan local Parquet once and retain only exact phrases under allowed licences."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise YfccStockSourceError("duckdb is required for the local metadata scan") from exc
    if not paths or any(not path.is_file() or not _valid_parquet(path, path.stat().st_size) for path in paths):
        raise YfccStockSourceError("local parquet set is missing or invalid")
    connection = duckdb.connect()
    connection.execute("CREATE TEMP TABLE stock_patterns(stock_id VARCHAR, pattern VARCHAR)")
    patterns = [(str(row["film_stock_id"]), str(row["exact_regex"])) for row in config["stock_patterns"]]
    connection.executemany("INSERT INTO stock_patterns VALUES (?, ?)", patterns)
    query = """
        WITH raw AS (
          SELECT photoid, uid, unickname, title, description, usertags, pageurl,
                 downloadurl, licensename, licenseurl,
                 regexp_replace(lower(concat_ws(' ', coalesce(title, ''),
                   coalesce(description, ''), coalesce(usertags, ''))),
                   '[+_-]+', ' ', 'g') AS evidence_text
          FROM read_parquet(?)
          WHERE marker = 0 AND licenseurl IN (SELECT unnest(?))
        )
        SELECT p.stock_id, r.photoid, r.uid, r.unickname, r.title, r.description,
               r.usertags, r.pageurl, r.downloadurl, r.licensename, r.licenseurl
        FROM raw r JOIN stock_patterns p ON regexp_matches(r.evidence_text, p.pattern)
        ORDER BY p.stock_id, r.uid, r.photoid
    """
    rows = connection.execute(
        query,
        [[str(path.resolve()) for path in paths], list(config["rights_filter"]["allowed_license_urls"])],
    ).fetchall()
    columns = [item[0] for item in connection.description]
    matches = [dict(zip(columns, row, strict=True)) for row in rows]
    by_stock: dict[str, dict[str, Any]] = {}
    gates = config["metadata_gates"]
    for stock_id, _ in patterns:
        selected = [row for row in matches if row["stock_id"] == stock_id]
        authors = Counter(str(row["uid"]) for row in selected)
        largest = max(authors.values(), default=0) / max(len(selected), 1)
        checks = {
            "minimum_rows": len(selected) >= int(gates["minimum_rows"]),
            "minimum_author_uids": len(authors) >= int(gates["minimum_author_uids"]),
            "maximum_largest_author_share": largest <= float(gates["maximum_largest_author_share"]),
        }
        by_stock[stock_id] = {
            "rows": len(selected),
            "author_uids": len(authors),
            "largest_author_share": largest,
            "checks": checks,
            "metadata_gate_passed": all(checks.values()),
        }
    return {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "parquet_files": len(paths),
        "parquet_sha256": {path.name: sha256_file(path) for path in paths},
        "candidate_results": by_stock,
        "matches": matches,
        "passing_stocks": sorted(stock_id for stock_id, row in by_stock.items() if row["metadata_gate_passed"]),
        "image_payloads_downloaded_or_decoded": False,
        "claim_ceiling": config["claim_ceiling"],
    }
