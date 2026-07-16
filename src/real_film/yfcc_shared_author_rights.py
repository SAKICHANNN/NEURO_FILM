"""Bounded page-only live-rights preflight for shared YFCC authors."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus, urlparse

import requests


class YfccSharedAuthorRightsError(ValueError):
    """Raised when the frozen SF1.2 contract fails closed."""


def _is_flickr_page_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    return parsed.scheme in {"http", "https"} and (host == "flickr.com" or host.endswith(".flickr.com"))


def _evidence_text(row: Mapping[str, Any]) -> str:
    return unquote_plus(
        " ".join(str(row.get(key) or "") for key in ("title", "description", "usertags"))
    ).casefold()


def select_shared_author_candidates(
    report: Mapping[str, Any], decision: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Freeze bounded page candidates from the hash-verified SF1.1 result."""
    gate_id = str(config["passing_gate_id"])
    if decision.get("decision") != "open_bounded_live_rights_preflight":
        raise YfccSharedAuthorRightsError("SF1.1 decision does not open live-rights preflight")
    if gate_id not in decision.get("passing_gate_ids", []):
        raise YfccSharedAuthorRightsError("configured gate did not pass SF1.1")
    if decision.get("pixel_download_allowed") is not False or decision.get("operator_fitting_allowed") is not False:
        raise YfccSharedAuthorRightsError("SF1.1 decision violates no-pixel/no-fitting contract")
    gate = report.get("shared_author_results", {}).get(gate_id)
    if not isinstance(gate, Mapping) or gate.get("metadata_gate_passed") is not True:
        raise YfccSharedAuthorRightsError("input report lacks the passing shared-author gate")
    shared = sorted((str(value) for value in gate.get("shared_uids", [])), key=str.casefold)
    if len(shared) != int(config["expected_support"]["shared_author_uids"]):
        raise YfccSharedAuthorRightsError("shared-author support drifted")
    stocks = [str(config["left_stock_id"]), str(config["right_stock_id"])]
    raw_rows = [
        dict(row)
        for row in report.get("matches", [])
        if str(row.get("uid")) in shared and str(row.get("film_stock_id")) in stocks
    ]
    if len(raw_rows) != int(config["expected_support"]["exclusive_candidate_rows"]):
        raise YfccSharedAuthorRightsError("exclusive shared-author row count drifted")
    exclusions = [re.compile(str(value), re.IGNORECASE) for value in config["process_contamination_exclusions"]]
    snapshot_license = str(config["live_rights"]["required_snapshot_license_url"])
    cap = int(config["selection"]["maximum_candidates_per_stock_per_author"])
    matrix: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for uid in shared:
        matrix[uid] = {}
        for stock in stocks:
            rows = [
                row
                for row in raw_rows
                if str(row["uid"]) == uid
                and str(row["film_stock_id"]) == stock
                and str(row.get("licenseurl")) == snapshot_license
                and not any(pattern.search(_evidence_text(row)) for pattern in exclusions)
            ]
            rows.sort(key=lambda row: int(row["photoid"]))
            selected = rows[:cap]
            for row in selected:
                if not _is_flickr_page_url(str(row.get("pageurl") or "")):
                    raise YfccSharedAuthorRightsError("candidate page URL is not a Flickr page")
            matrix[uid][stock] = selected
    maximum = int(config["selection"]["maximum_page_requests"])
    if sum(len(rows) for stock_rows in matrix.values() for rows in stock_rows.values()) > maximum:
        raise YfccSharedAuthorRightsError("candidate matrix exceeds the frozen request ceiling")
    return matrix


def _request_page(
    client: requests.Session,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    url = str(row["pageurl"]).replace("http://", "https://", 1)
    limits = config["request_limits"]
    rights = config["live_rights"]
    retries = int(limits["request_retries"])
    maximum_bytes = int(rights["maximum_html_bytes"])
    last_error = ""
    for attempt_index in range(retries):
        response: requests.Response | None = None
        try:
            response = client.get(
                url,
                timeout=float(limits["timeout_seconds"]),
                stream=True,
                allow_redirects=True,
            )
            status = int(response.status_code)
            content_type = str(response.headers.get("Content-Type", ""))
            final_url = str(response.url)
            if status in {429, 500, 502, 503, 504} and attempt_index + 1 < retries:
                last_error = f"transient HTTP {status}"
                sleep_fn(float(limits["retry_backoff_seconds"]) * (2**attempt_index))
                continue
            body = bytearray()
            if status == 200 and content_type.casefold().startswith(
                str(rights["required_content_type_prefix"]).casefold()
            ):
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        body.extend(chunk)
                    if len(body) > maximum_bytes:
                        break
            bounded = len(body) <= maximum_bytes
            text = bytes(body).decode(response.encoding or "utf-8", errors="ignore") if bounded else ""
            confirmed = (
                status == 200
                and content_type.casefold().startswith(str(rights["required_content_type_prefix"]).casefold())
                and bounded
                and _is_flickr_page_url(final_url)
                and re.search(str(rights["required_page_regex"]), text, re.IGNORECASE) is not None
            )
            return {
                "photoid": int(row["photoid"]),
                "uid": str(row["uid"]),
                "film_stock_id": str(row["film_stock_id"]),
                "snapshot_page_url": str(row["pageurl"]),
                "live_page_url": final_url,
                "page_status": status,
                "content_type": content_type,
                "html_bytes": len(body),
                "html_sha256": hashlib.sha256(body).hexdigest(),
                "live_cc_by_2_confirmed": confirmed,
                "decision": "retain_live_rights" if confirmed else "reject_live_rights",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempt_index": attempt_index,
            }
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt_index + 1 < retries:
                sleep_fn(float(limits["retry_backoff_seconds"]) * (2**attempt_index))
        finally:
            if response is not None:
                response.close()
    return {
        "photoid": int(row["photoid"]),
        "uid": str(row["uid"]),
        "film_stock_id": str(row["film_stock_id"]),
        "snapshot_page_url": str(row["pageurl"]),
        "live_page_url": None,
        "page_status": None,
        "content_type": None,
        "html_bytes": 0,
        "html_sha256": hashlib.sha256(b"").hexdigest(),
        "live_cc_by_2_confirmed": False,
        "decision": "reject_request_failure",
        "reason": last_error,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_index": retries - 1,
    }


def run_shared_author_rights_preflight(
    matrix: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Request only bounded Flickr HTML pages and apply the frozen author gate."""
    client = session or requests.Session()
    client.headers["User-Agent"] = str(config.get("user_agent", "K-MCFM-research-rights-preflight/1.0"))
    stocks = [str(config["left_stock_id"]), str(config["right_stock_id"])]
    attempts: list[dict[str, Any]] = []
    author_results: dict[str, Any] = {}
    for uid in sorted(matrix, key=str.casefold):
        stock_pass: dict[str, bool] = {}
        retained_pages: dict[str, int | None] = {}
        for stock in stocks:
            passed = False
            retained: int | None = None
            for row in matrix[uid][stock]:
                attempt = _request_page(client, row, config, sleep_fn=sleep_fn)
                attempts.append(attempt)
                sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
                if attempt["live_cc_by_2_confirmed"]:
                    passed = True
                    retained = int(row["photoid"])
                    break
            stock_pass[stock] = passed
            retained_pages[stock] = retained
        author_results[uid] = {
            "stock_live_rights": stock_pass,
            "retained_photoids": retained_pages,
            "usable_shared_author": all(stock_pass.values()),
        }
    usable = sorted(
        (uid for uid, result in author_results.items() if result["usable_shared_author"]),
        key=str.casefold,
    )
    passed = len(usable) >= int(config["decision_gates"]["minimum_usable_shared_authors"])
    return {
        "schema_version": 1,
        "preflight_id": config["preflight_id"],
        "input_report_sha256": config["input_report_sha256"],
        "input_decision_sha256": config["input_decision_sha256"],
        "attempts": attempts,
        "page_requests": len(attempts),
        "author_results": author_results,
        "usable_shared_author_uids": usable,
        "usable_shared_author_count": len(usable),
        "minimum_usable_shared_authors": int(config["decision_gates"]["minimum_usable_shared_authors"]),
        "decision": "pass_live_rights_feasibility" if passed else "close_shared_author_public_expansion",
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
