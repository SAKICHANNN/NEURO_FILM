"""Prospectively frozen urllib transport for the SF3.A3L metadata refresh."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.real_film.openverse_stock_source import (
    OpenverseStockSourceError,
    _normalize_result,
    _validate_config,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bound_json(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    data = path.read_bytes()
    if _sha256(data) != str(binding["sha256"]):
        raise OpenverseStockSourceError(f"bound input hash mismatch: {binding['path']}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise OpenverseStockSourceError(
            f"bound input is not an object: {binding['path']}"
        )
    return value


def load_effective_config(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load the frozen SF3.A2 scientific contract and apply A3L identity only."""
    base = _read_bound_json(root, contract["base_contract"])
    _read_bound_json(root, contract["superseded_access_result"])
    base["audit_id"] = str(contract["audit_id"])
    base["claim_ceiling"] = str(contract["claim_ceiling"])
    _validate_config(base)
    return base


def _bounded_get(
    url: str,
    *,
    transport: Mapping[str, Any],
    opener: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": str(transport["user_agent"]),
            "Accept": str(transport["accept"]),
        },
        method="GET",
    )
    maximum = int(transport["maximum_response_bytes"])
    response = None
    try:
        response = opener.open(request, timeout=float(transport["timeout_seconds"]))
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl())
        headers = response.headers
        body = response.read(maximum + 1)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        final_url = str(exc.geturl())
        headers = exc.headers
        body = exc.read(maximum + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {}, {
            "bounded_json_valid": False,
            "error": f"transport-error:{type(exc).__name__}",
            "request_utc": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        if response is not None:
            response.close()

    parsed = urllib.parse.urlparse(final_url)
    content_type = str(headers.get("Content-Type", ""))
    bounded = len(body) <= maximum
    valid = (
        status == 200
        and parsed.scheme == str(transport["required_scheme"])
        and parsed.hostname == str(transport["required_host"])
        and parsed.path == str(transport["required_path"])
        and content_type.casefold().startswith(str(transport["accept"]).casefold())
        and bounded
    )
    evidence = {
        "status": status,
        "final_url": final_url,
        "content_type": content_type,
        "bytes": len(body),
        "sha256": _sha256(body),
        "request_utc": datetime.now(timezone.utc).isoformat(),
        "bounded_json_valid": valid,
        "rate_limit_limit": headers.get("X-RateLimit-Limit"),
        "rate_limit_remaining": headers.get("X-RateLimit-Remaining"),
        "rate_limit_reset": headers.get("X-RateLimit-Reset"),
    }
    if not valid:
        return {}, evidence
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        evidence["bounded_json_valid"] = False
        evidence["error"] = "invalid_json"
        return {}, evidence
    if not isinstance(payload, dict):
        evidence["bounded_json_valid"] = False
        evidence["error"] = "non_object_json"
        return {}, evidence
    return payload, evidence


def fetch_snapshot_urllib(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    stock_order: Sequence[str] | None = None,
    opener: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch only frozen Openverse metadata with no pixel-bearing fields."""
    _validate_config(config)
    transport = contract["transport"]
    client = opener or urllib.request.build_opener(_NoRedirect())
    stocks = [deepcopy(stock) for stock in config["stocks"]]
    if stock_order is not None:
        by_id = {str(stock["film_stock_id"]): stock for stock in stocks}
        if set(stock_order) != set(by_id):
            raise OpenverseStockSourceError("stock order does not match frozen stocks")
        stocks = [by_id[value] for value in stock_order]

    categories: list[dict[str, Any]] = []
    request_count = 0
    source_error = False
    for stock_index, stock in enumerate(stocks):
        rows: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        observed_page_count: int | None = None
        for page in range(1, int(config["query"]["maximum_pages_per_stock"]) + 1):
            if observed_page_count is not None and page > observed_page_count:
                break
            if request_count >= int(config["query"]["maximum_total_requests"]):
                raise OpenverseStockSourceError("frozen request ceiling exceeded")
            params = urllib.parse.urlencode(
                {
                    "q": str(stock["query"]),
                    "page_size": int(config["query"]["page_size"]),
                    "page": page,
                }
            )
            payload, evidence = _bounded_get(
                f"{config['source']['api_endpoint']}?{params}",
                transport=transport,
                opener=client,
            )
            request_count += 1
            evidence["page"] = page
            requests.append(evidence)
            if not evidence.get("bounded_json_valid"):
                source_error = True
                break
            results = payload.get("results")
            if not isinstance(results, list):
                evidence["bounded_json_valid"] = False
                evidence["error"] = "missing_results"
                source_error = True
                break
            observed_page_count = min(
                int(payload.get("page_count") or 0),
                int(config["query"]["maximum_pages_per_stock"]),
            )
            rows.extend(
                _normalize_result(row, stock, config)
                for row in results
                if isinstance(row, Mapping)
            )
            if page < observed_page_count:
                sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
        categories.append(
            {
                "film_stock_id": str(stock["film_stock_id"]),
                "query": str(stock["query"]),
                "observed_page_count": observed_page_count,
                "requests": requests,
                "rows": rows,
            }
        )
        if stock_index + 1 < len(stocks):
            sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
    return {
        "schema_version": 1,
        "audit_id": str(contract["audit_id"]),
        "source": dict(config["source"]),
        "transport": str(transport["implementation"]),
        "request_count": request_count,
        "source_error": source_error,
        "raw_response_retained": False,
        "image_urls_retained": False,
        "categories": categories,
        "image_payload_download_allowed": False,
        "photo_page_access_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": str(contract["claim_ceiling"]),
    }


_VOLATILE_REQUEST_FIELDS = {
    "request_utc",
    "rate_limit_limit",
    "rate_limit_remaining",
    "rate_limit_reset",
}


def scientific_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize acquisition order and remove only frozen volatile fields."""
    value = deepcopy(dict(snapshot))
    categories = value.get("categories", [])
    for category in categories:
        category["rows"] = sorted(category["rows"], key=lambda row: str(row["id"]))
        for request in category["requests"]:
            for field in _VOLATILE_REQUEST_FIELDS:
                request.pop(field, None)
    value["categories"] = sorted(
        categories, key=lambda item: str(item["film_stock_id"])
    )
    return value
