"""Live-rights verification and bounded YFCC stock pixel pilot."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import requests
from PIL import Image, ImageOps

from src.real_film.yfcc_stock_source import atomic_json


class YfccStockPilotError(ValueError):
    """Raised when the frozen YFCC pixel contract fails closed."""


def _evidence_text(row: Mapping[str, Any]) -> str:
    return unquote_plus(" ".join(str(row.get(key) or "") for key in ("title", "description", "usertags"))).casefold()


def eligible_rows(report: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply prospective contamination exclusions to one exact-stock pool."""
    if report.get("parquet_sha256") is None:
        raise YfccStockPilotError("metadata report lacks frozen parquet hashes")
    stock_id = str(config["target_stock_id"])
    patterns = [re.compile(value, re.IGNORECASE) for value in config["process_contamination_exclusions"]]
    selected = [
        dict(row)
        for row in report.get("matches", [])
        if row.get("stock_id") == stock_id
        and not any(pattern.search(_evidence_text(row)) for pattern in patterns)
    ]
    selected.sort(key=lambda row: (str(row["uid"]), int(row["photoid"])))
    authors = Counter(str(row["uid"]) for row in selected)
    expected = config["selection"]["preflight_expected"]
    if len(selected) != int(expected["eligible_rows"]) or len(authors) != int(expected["author_uids"]):
        raise YfccStockPilotError("prospective contamination-filter preflight drifted")
    largest = max(authors.values(), default=0) / max(len(selected), 1)
    if abs(largest - float(expected["largest_author_share"])) > 1e-12:
        raise YfccStockPilotError("prospective author share drifted")
    return selected


def balanced_candidate_order(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return stable UID-round-robin candidates under a per-UID ceiling."""
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_uid[str(row["uid"])].append(dict(row))
    for values in by_uid.values():
        values.sort(key=lambda row: int(row["photoid"]))
    cap = int(config["selection"]["maximum_files_per_uid"])
    ordered: list[dict[str, Any]] = []
    for round_index in range(cap):
        for uid in sorted(by_uid, key=str.casefold):
            if round_index < len(by_uid[uid]):
                row = by_uid[uid][round_index]
                row["candidate_rank"] = len(ordered)
                ordered.append(row)
    return ordered


def candidate_image_urls(download_url: str) -> list[str]:
    """Prefer Flickr's documented large suffix while retaining the frozen URL."""
    secure = download_url.replace("http://", "https://", 1)
    large = re.sub(r"(?i)(\.[a-z0-9]+)$", r"_b\1", secure)
    return list(dict.fromkeys((large, secure)))


def _verify_payload(payload: bytes, minimum_short_dimension: int) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            transposed = ImageOps.exif_transpose(image)
            width, height = transposed.size
            mode = transposed.mode
            image_format = str(image.format or "")
    except Exception as exc:
        raise YfccStockPilotError("image decode failed") from exc
    if min(width, height) < minimum_short_dimension:
        raise YfccStockPilotError("image is below the minimum short dimension")
    return {"width": width, "height": height, "mode": mode, "format": image_format}


def _checkpoint(path: Path | None, config: Mapping[str, Any], attempts: list[dict[str, Any]], rows: list[dict[str, Any]], complete: bool) -> None:
    if path is not None:
        atomic_json(path, {
            "schema_version": 1,
            "pilot_id": config["pilot_id"],
            "complete": complete,
            "attempts": attempts,
            "files": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "rows": rows,
            "claim_ceiling": config["claim_ceiling"],
        })


def download_live_pixels(
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    root: Path,
    checkpoint_path: Path | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Verify live CC-BY pages and retain bounded, decoded pixels."""
    root.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    client.headers["User-Agent"] = str(config["user_agent"])
    limits = config["download_limits"]
    audit = config["pixel_audit"]
    maximum = int(config["selection"]["maximum_retained_files"])
    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(records) >= maximum:
            break
        photoid = int(candidate["photoid"])
        page_url = str(candidate["pageurl"]).replace("http://", "https://", 1)
        attempt: dict[str, Any] = {"candidate_rank": int(candidate["candidate_rank"]), "photoid": photoid, "uid": str(candidate["uid"])}
        try:
            page = client.get(page_url, timeout=float(limits["timeout_seconds"]), allow_redirects=True)
            page_text = page.text if page.status_code == 200 else ""
            attempt["page_status"] = int(page.status_code)
            attempt["live_page_url"] = str(page.url)
            page.close()
            if not re.search(str(config["live_rights"]["required_page_regex"]), page_text, re.IGNORECASE):
                attempt["decision"] = "reject_live_cc_by_2_not_confirmed"
                attempts.append(attempt)
                _checkpoint(checkpoint_path, config, attempts, records, False)
                continue
            payload: bytes | None = None
            selected_url = ""
            content_type = ""
            for image_url in candidate_image_urls(str(candidate["downloadurl"])):
                response = client.get(image_url, timeout=float(limits["timeout_seconds"]), stream=True, allow_redirects=True)
                if response.status_code != 200 or not str(response.headers.get("Content-Type", "")).casefold().startswith("image/"):
                    response.close()
                    continue
                buffer = bytearray()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        buffer.extend(chunk)
                    if len(buffer) > int(limits["maximum_bytes_per_file"]):
                        break
                content_type = str(response.headers.get("Content-Type", ""))
                selected_url = str(response.url)
                response.close()
                if 0 < len(buffer) <= int(limits["maximum_bytes_per_file"]):
                    payload = bytes(buffer)
                    break
            if payload is None:
                raise YfccStockPilotError("no bounded live image response")
            decoded = _verify_payload(payload, int(audit["minimum_short_dimension"]))
            if sum(int(row["bytes"]) for row in records) + len(payload) > int(limits["maximum_bytes_total"]):
                raise YfccStockPilotError("aggregate byte ceiling exceeded")
            suffix = "." + decoded["format"].casefold().replace("jpeg", "jpg")
            relative = Path(str(config["target_stock_id"])) / f"{photoid}{suffix}"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            digest = hashlib.sha256(payload).hexdigest()
            record = {
                "selection_index": len(records),
                "candidate_rank": int(candidate["candidate_rank"]),
                "film_stock_id": str(config["target_stock_id"]),
                "page_id": photoid,
                "photoid": photoid,
                "title": unquote_plus(str(candidate.get("title") or "")),
                "normalized_author_group": str(candidate["uid"]),
                "uploader": str(candidate["uid"]),
                "author_uid": str(candidate["uid"]),
                "author_nickname": str(candidate.get("unickname") or ""),
                "license_snapshot_name": str(candidate["licensename"]),
                "license_snapshot_url": str(candidate["licenseurl"]),
                "live_cc_by_2_confirmed": True,
                "file_page_url": page_url,
                "snapshot_download_url": str(candidate["downloadurl"]),
                "derivative_url": selected_url,
                "local_path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": digest,
                "content_type": content_type,
                **decoded,
            }
            records.append(record)
            attempt.update({"decision": "retain", "selection_index": record["selection_index"], "bytes": len(payload), "sha256": digest})
        except (OSError, requests.RequestException, YfccStockPilotError) as exc:
            attempt["decision"] = "reject_request_or_pixel_gate"
            attempt["reason"] = str(exc)
        attempts.append(attempt)
        _checkpoint(checkpoint_path, config, attempts, records, False)
        time.sleep(float(limits["request_interval_seconds"]))
    result = {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "complete": True,
        "attempts": attempts,
        "files": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "rows": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    _checkpoint(checkpoint_path, config, attempts, records, True)
    return result
