"""Repair a frozen YFCC SQLite byte range through validated fixed-size requests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_full_index import (  # noqa: E402
    YfccFullIndexError,
    exclusive_dataset_lock,
    validate_source_headers,
    validate_sqlite_header,
)


def _expected_content_range(start: int, end_inclusive: int, total: int) -> str:
    return f"bytes {start}-{end_inclusive}/{total}"


def download_repair_range(
    config: dict,
    repair_path: Path,
    start: int,
    end_exclusive: int,
    *,
    chunk_bytes: int,
) -> None:
    source = config["source"]
    total = int(source["expected_bytes"])
    if not 0 <= start < end_exclusive <= total:
        raise YfccFullIndexError("invalid repair byte range")
    client = requests.Session()
    client.headers["User-Agent"] = str(config["user_agent"])
    timeout = (
        float(config["download_limits"]["connect_timeout_seconds"]),
        float(config["download_limits"]["read_timeout_seconds"]),
    )
    head = client.head(str(source["url"]), timeout=timeout, allow_redirects=True)
    try:
        head.raise_for_status()
        validate_source_headers(head.headers, config)
    finally:
        head.close()
    repair_path.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = end_exclusive - start
    current = repair_path.stat().st_size if repair_path.exists() else 0
    if current > target_bytes:
        raise YfccFullIndexError("repair file exceeds requested range")
    aligned = current - (current % chunk_bytes)
    if aligned != current:
        with repair_path.open("r+b") as handle:
            handle.truncate(aligned)
        current = aligned
    retries = int(config["download_limits"]["request_retries"])
    while current < target_bytes:
        remote_start = start + current
        remote_end = min(remote_start + chunk_bytes, end_exclusive) - 1
        expected = remote_end - remote_start + 1
        last_error: Exception | None = None
        for attempt in range(retries):
            response: requests.Response | None = None
            try:
                response = client.get(
                    str(source["url"]),
                    headers={"Range": f"bytes={remote_start}-{remote_end}"},
                    timeout=timeout,
                    stream=True,
                    allow_redirects=True,
                )
                if response.status_code != 206:
                    raise YfccFullIndexError(f"unexpected repair status: {response.status_code}")
                if response.headers.get("Content-Range") != _expected_content_range(remote_start, remote_end, total):
                    raise YfccFullIndexError("repair Content-Range drifted")
                written = 0
                with repair_path.open("ab") as handle:
                    for payload in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if payload:
                            handle.write(payload)
                            written += len(payload)
                if written != expected:
                    raise YfccFullIndexError("repair chunk byte count mismatch")
                current += written
                print(json.dumps({"repair_bytes": current, "target_bytes": target_bytes, "fraction": current / target_bytes}))
                last_error = None
                break
            except (OSError, requests.RequestException, YfccFullIndexError) as exc:
                last_error = exc
                with repair_path.open("r+b") as handle:
                    handle.truncate(current)
                if attempt + 1 < retries:
                    time.sleep(float(config["download_limits"]["retry_backoff_seconds"]) * (2 ** attempt))
            finally:
                if response is not None:
                    response.close()
        if last_error is not None:
            raise YfccFullIndexError("repair range failed after retries") from last_error


def apply_repair_range(source_path: Path, repair_path: Path, start: int, end_exclusive: int, total: int) -> None:
    validate_sqlite_header(source_path, total)
    expected = end_exclusive - start
    if repair_path.stat().st_size != expected:
        raise YfccFullIndexError("repair file is incomplete")
    with source_path.open("r+b") as destination, repair_path.open("rb") as repair:
        destination.seek(start)
        for payload in iter(lambda: repair.read(8 * 1024 * 1024), b""):
            destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())
    validate_sqlite_header(source_path, total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "real_film_yfcc_full_index_v1.json")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end-exclusive", type=int, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = ROOT / config["download"]["destination"]
    repair_path = source_path.with_suffix(source_path.suffix + f".{args.start}-{args.end_exclusive}.repair")
    with exclusive_dataset_lock(source_path):
        download_repair_range(config, repair_path, args.start, args.end_exclusive, chunk_bytes=args.chunk_bytes)
        apply_repair_range(source_path, repair_path, args.start, args.end_exclusive, int(config["source"]["expected_bytes"]))
    print(json.dumps({"patched": source_path.as_posix(), "start": args.start, "end_exclusive": args.end_exclusive}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
