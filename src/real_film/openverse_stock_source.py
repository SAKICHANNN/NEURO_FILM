"""Bounded Openverse metadata acquisition and offline stock-source audit."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from itertools import combinations
from typing import Any
from urllib.parse import urlparse

import requests


class OpenverseStockSourceError(ValueError):
    """Raised when the frozen SF2.1A contract fails closed."""


def _normal_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _normal_tag(value: object) -> str:
    return "".join(character for character in _normal_text(value) if character.isalnum())


def _creator_group(row: Mapping[str, Any]) -> str:
    return f"{_normal_text(row['source'])}|{_normal_text(row['creator_url'])}"


def _validate_config(config: Mapping[str, Any]) -> None:
    endpoint = urlparse(str(config["source"]["api_endpoint"]))
    if endpoint.scheme != "https" or endpoint.hostname != "api.openverse.org" or endpoint.path != "/v1/images/":
        raise OpenverseStockSourceError("API endpoint is outside the frozen Openverse route")
    query = config["query"]
    maximum = int(query["maximum_pages_per_stock"]) * len(config["stocks"])
    if maximum != int(query["maximum_total_requests"]):
        raise OpenverseStockSourceError("request ceiling does not match stocks times pages")
    if int(config["request_limits"]["request_retries"]) != 1:
        raise OpenverseStockSourceError("network retries would exceed the frozen request ceiling")
    gates = config["connectivity_gate"]
    if gates.get("require_same_source_shared_creator_edges") is not True:
        raise OpenverseStockSourceError("same-source shared-creator edges are required")
    if gates.get("cross_stock_openverse_id_overlap_must_be_zero") is not True:
        raise OpenverseStockSourceError("cross-stock Openverse ID overlap must fail closed")
    if gates.get("cross_stock_landing_url_overlap_must_be_zero") is not True:
        raise OpenverseStockSourceError("cross-stock landing URL overlap must fail closed")
    for flag in (
        "image_payload_download_allowed",
        "photo_page_access_allowed",
        "operator_fitting_allowed",
        "training_allowed",
        "latent_mode_study_allowed",
    ):
        if config.get(flag) is not False:
            raise OpenverseStockSourceError(f"{flag} must remain false")
    for flag in (
        "image_payload_requests_allowed",
        "thumbnail_requests_allowed",
        "detail_requests_allowed",
        "landing_page_requests_allowed",
        "related_requests_allowed",
        "raw_response_retained",
    ):
        if query.get(flag) is not False:
            raise OpenverseStockSourceError(f"query.{flag} must remain false")
    stock_ids = [str(stock["film_stock_id"]) for stock in config["stocks"]]
    if len(stock_ids) != len(set(stock_ids)):
        raise OpenverseStockSourceError("film stock IDs must be unique")


def _bounded_json_response(response: requests.Response, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    maximum_bytes = int(config["query"]["maximum_response_bytes"])
    body = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if chunk:
            body.extend(chunk)
        if len(body) > maximum_bytes:
            break
    content_type = str(response.headers.get("Content-Type", ""))
    bounded = len(body) <= maximum_bytes
    valid = (
        int(response.status_code) == int(config["query"]["required_status"])
        and content_type.casefold().startswith(str(config["query"]["required_content_type_prefix"]).casefold())
        and bounded
    )
    evidence = {
        "status": int(response.status_code),
        "final_url": str(response.url),
        "content_type": content_type,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "request_utc": datetime.now(timezone.utc).isoformat(),
        "bounded_json_valid": valid,
        "rate_limit_limit": response.headers.get("X-RateLimit-Limit"),
        "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
        "rate_limit_reset": response.headers.get("X-RateLimit-Reset"),
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


def _label_aliases(stock: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    titles = {_normal_text(value) for value in stock["title_aliases"]}
    tags = {_normal_tag(value) for value in stock["tag_aliases"]}
    return titles, tags


def _normalize_result(row: Mapping[str, Any], stock: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    title_aliases, tag_aliases = _label_aliases(stock)
    title = str(row.get("title") or "")
    raw_tags = row.get("tags") if isinstance(row.get("tags"), list) else []
    tag_names = sorted(
        {
            _normal_tag(tag.get("name"))
            for tag in raw_tags
            if isinstance(tag, Mapping) and _normal_tag(tag.get("name"))
        }
    )
    title_matches = sorted(alias for alias in title_aliases if alias in _normal_text(title))
    tag_matches = sorted(alias for alias in tag_aliases if alias in set(tag_names))
    maximum_tags = int(config["query"]["maximum_retained_tags_per_row"])
    maximum_chars = int(config["query"]["maximum_retained_tag_characters"])
    preferred = tag_matches + [value for value in tag_names if value not in tag_matches]
    retained_tags = [value[:maximum_chars] for value in preferred[:maximum_tags]]
    fields_matched = sorted(str(value) for value in row.get("fields_matched", []) if isinstance(value, str))
    return {
        "id": str(row.get("id") or ""),
        "film_stock_id": str(stock["film_stock_id"]),
        "title": title,
        "indexed_on": str(row.get("indexed_on") or ""),
        "foreign_landing_url": str(row.get("foreign_landing_url") or ""),
        "creator": str(row.get("creator") or ""),
        "creator_url": str(row.get("creator_url") or ""),
        "license": str(row.get("license") or "").casefold(),
        "license_version": str(row.get("license_version") or ""),
        "license_url": str(row.get("license_url") or ""),
        "provider": str(row.get("provider") or ""),
        "source": str(row.get("source") or ""),
        "category": row.get("category"),
        "tags": retained_tags,
        "fields_matched": fields_matched,
        "mature": bool(row.get("mature", False)),
        "height": int(row.get("height") or 0),
        "width": int(row.get("width") or 0),
        "title_alias_matches": title_matches,
        "tag_alias_matches": tag_matches,
    }


def fetch_snapshot(
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch only the frozen Openverse search pages and omit all pixel URLs."""
    _validate_config(config)
    client = session or requests.Session()
    client.headers["User-Agent"] = str(config["request_limits"]["user_agent"])
    categories: list[dict[str, Any]] = []
    request_count = 0
    source_error = False
    for stock_index, stock in enumerate(config["stocks"]):
        rows: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        observed_page_count: int | None = None
        for page in range(1, int(config["query"]["maximum_pages_per_stock"]) + 1):
            if observed_page_count is not None and page > observed_page_count:
                break
            if request_count >= int(config["query"]["maximum_total_requests"]):
                raise OpenverseStockSourceError("frozen request ceiling exceeded")
            response: requests.Response | None = None
            try:
                response = client.get(
                    str(config["source"]["api_endpoint"]),
                    params={"q": str(stock["query"]), "page_size": int(config["query"]["page_size"]), "page": page},
                    timeout=float(config["request_limits"]["timeout_seconds"]),
                    stream=True,
                    allow_redirects=False,
                )
                request_count += 1
                payload, item = _bounded_json_response(response, config)
                final = urlparse(str(item.get("final_url") or ""))
                if final.scheme != "https" or final.hostname != "api.openverse.org" or final.path != "/v1/images/":
                    item["bounded_json_valid"] = False
                    item["error"] = "query_contract_mismatch"
                    payload = {}
            except requests.RequestException as exc:
                request_count += 1
                payload = {}
                item = {"page": page, "bounded_json_valid": False, "error": str(exc)}
            finally:
                if response is not None:
                    response.close()
            item["page"] = page
            evidence.append(item)
            if not item.get("bounded_json_valid"):
                source_error = True
                break
            if not isinstance(payload.get("results"), list):
                item["bounded_json_valid"] = False
                item["error"] = "missing_results"
                source_error = True
                break
            observed_page_count = min(
                int(payload.get("page_count") or 0), int(config["query"]["maximum_pages_per_stock"])
            )
            for raw in payload["results"]:
                if isinstance(raw, Mapping):
                    rows.append(_normalize_result(raw, stock, config))
            if page < observed_page_count:
                sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
        categories.append(
            {
                "film_stock_id": str(stock["film_stock_id"]),
                "query": str(stock["query"]),
                "observed_page_count": observed_page_count,
                "requests": evidence,
                "rows": rows,
            }
        )
        if stock_index + 1 < len(config["stocks"]):
            sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
    return {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "source": dict(config["source"]),
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
        "claim_ceiling": config["claim_ceiling"],
    }


def _connected_components(nodes: Sequence[str], edges: Mapping[tuple[str, str], Sequence[str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[str]] = []
    unseen = set(nodes)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component))
        unseen -= component
        components.append(sorted(component))
    return components


def audit_snapshot(snapshot: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit one immutable metadata snapshot without network or pixels."""
    _validate_config(config)
    categories = snapshot.get("categories")
    if not isinstance(categories, list):
        raise OpenverseStockSourceError("snapshot categories are missing")
    configured = {str(stock["film_stock_id"]): stock for stock in config["stocks"]}
    strict_codes = set(str(value) for value in config["license_policy"]["strict_derivative_license_codes"])
    strict_versions = set(str(value) for value in config["license_policy"]["strict_license_versions"])
    flag_patterns = [re.compile(str(value), re.IGNORECASE) for value in config["label_policy"]["flag_title_patterns"]]
    gates = config["connectivity_gate"]
    results: dict[str, Any] = {}
    strict_rows_by_stock: dict[str, list[Mapping[str, Any]]] = {}
    ids: dict[str, set[str]] = {}
    landings: dict[str, set[str]] = {}
    query_errors: list[str] = []
    contract_errors: list[str] = []
    for category in categories:
        stock_id = str(category.get("film_stock_id"))
        if stock_id not in configured or stock_id in results:
            raise OpenverseStockSourceError(f"unexpected or duplicate stock category: {stock_id}")
        rows = category.get("rows")
        if not isinstance(rows, list):
            raise OpenverseStockSourceError(f"missing rows: {stock_id}")
        row_ids = [str(row.get("id") or "") for row in rows]
        row_landings = [str(row.get("foreign_landing_url") or "") for row in rows]
        if len([value for value in row_ids if value]) != len(set(value for value in row_ids if value)):
            contract_errors.append(f"{stock_id}:duplicate-openverse-id")
        if len([value for value in row_landings if value]) != len(set(value for value in row_landings if value)):
            contract_errors.append(f"{stock_id}:duplicate-landing-url")
        for request in category.get("requests", []):
            if not request.get("bounded_json_valid"):
                location = f"{stock_id}:page-{request.get('page')}"
                if request.get("error") in {"query_contract_mismatch", "missing_results", "invalid_json", "non_object_json"}:
                    contract_errors.append(location)
                else:
                    query_errors.append(location)
        strict_rows: list[Mapping[str, Any]] = []
        flagged = 0
        for row in rows:
            row_id = str(row.get("id") or "")
            landing = str(row.get("foreign_landing_url") or "")
            if row_id:
                ids.setdefault(row_id, set()).add(stock_id)
            if landing:
                landings.setdefault(landing, set()).add(stock_id)
            fields = {str(value).casefold() for value in row.get("fields_matched", [])}
            alias_match = bool(row.get("title_alias_matches") or row.get("tag_alias_matches"))
            field_match = "title" in fields or any(value.startswith("tag") for value in fields)
            complete = all(
                str(row.get(key) or "").strip()
                for key in ("id", "foreign_landing_url", "creator", "creator_url", "source", "provider", "license", "license_version", "license_url")
            )
            title_flagged = any(pattern.search(str(row.get("title") or "")) for pattern in flag_patterns)
            flagged += int(title_flagged)
            strict = (
                alias_match
                and field_match
                and complete
                and str(row.get("license")) in strict_codes
                and str(row.get("license_version")) in strict_versions
                and not bool(row.get("mature"))
                and int(row.get("width") or 0) > 0
                and int(row.get("height") or 0) > 0
            )
            if strict:
                strict_rows.append(row)
        groups = Counter(_creator_group(row) for row in strict_rows)
        largest_share = max(groups.values(), default=0) / max(len(strict_rows), 1)
        eligible = (
            len(strict_rows) >= int(gates["minimum_strict_rows_per_eligible_stock"])
            and len(groups) >= int(gates["minimum_unique_creators_per_eligible_stock"])
            and largest_share <= float(gates["maximum_largest_creator_share_per_eligible_stock"])
        )
        strict_rows_by_stock[stock_id] = strict_rows
        results[stock_id] = {
            "rows": len(rows),
            "strict_rows": len(strict_rows),
            "strict_creator_groups": len(groups),
            "largest_strict_creator_share": largest_share,
            "title_flagged_rows": flagged,
            "eligible": eligible,
            "source_counts": dict(sorted(Counter(str(row["source"]) for row in strict_rows).items())),
            "license_counts": dict(sorted(Counter(f"{row['license']}-{row['license_version']}" for row in rows).items())),
        }
    missing = sorted(set(configured) - set(results))
    if missing:
        raise OpenverseStockSourceError(f"configured stocks missing: {missing}")
    id_overlaps = sorted(value for value, stocks in ids.items() if len(stocks) > 1)
    landing_overlaps = sorted(value for value, stocks in landings.items() if len(stocks) > 1)
    eligible_stocks = sorted(stock for stock, result in results.items() if result["eligible"])
    retained_edges: dict[tuple[str, str], list[str]] = {}
    edge_rows: list[dict[str, Any]] = []
    for left, right in combinations(eligible_stocks, 2):
        left_groups = {_creator_group(row) for row in strict_rows_by_stock[left]}
        right_groups = {_creator_group(row) for row in strict_rows_by_stock[right]}
        shared = sorted(left_groups & right_groups)
        retained = len(shared) >= int(gates["minimum_shared_creators_per_graph_edge"])
        edge_rows.append({"left": left, "right": right, "shared_creator_groups": len(shared), "retained": retained})
        if retained:
            retained_edges[(left, right)] = shared
    component_results: list[dict[str, Any]] = []
    for component in _connected_components(eligible_stocks, retained_edges):
        component_edges = {
            edge: groups for edge, groups in retained_edges.items() if edge[0] in component and edge[1] in component
        }
        shared_groups = sorted({group for groups in component_edges.values() for group in groups})
        sources = sorted({group.split("|", 1)[0] for group in shared_groups})
        passes = (
            len(component) >= int(gates["minimum_connected_stocks"])
            and len(shared_groups) >= int(gates["minimum_shared_creators_total"])
            and len(sources) >= int(gates["minimum_distinct_sources_in_connected_component"])
        )
        component_results.append(
            {
                "stocks": component,
                "shared_creator_groups": len(shared_groups),
                "sources": sources,
                "strict_rows": sum(int(results[stock]["strict_rows"]) for stock in component),
                "passes": passes,
            }
        )
    passing_components = [item for item in component_results if item["passes"]]
    passing_components.sort(
        key=lambda item: (-len(item["stocks"]), -int(item["shared_creator_groups"]), -int(item["strict_rows"]), item["stocks"])
    )
    if contract_errors:
        decision_name = "query_contract_mismatch"
    elif query_errors or snapshot.get("source_error"):
        decision_name = "source_unavailable"
    elif id_overlaps or landing_overlaps:
        decision_name = "cross_stock_identity_overlap"
    elif passing_components:
        decision_name = "open_bounded_live_rights_label_preflight"
    else:
        decision_name = "insufficient_shared_creator_connectivity"
    if decision_name not in config["allowed_decisions"]:
        raise OpenverseStockSourceError("decision is outside the frozen branch set")
    decision = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "decision": decision_name,
        "selected_component": passing_components[0] if passing_components else None,
        "query_errors": query_errors,
        "contract_errors": contract_errors,
        "cross_stock_openverse_id_overlap": id_overlaps,
        "cross_stock_landing_url_overlap": landing_overlaps,
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "request_count": int(snapshot.get("request_count") or 0),
        "stock_results": results,
        "edges": edge_rows,
        "components": component_results,
        "summary": decision,
        "raw_response_retained": False,
        "image_urls_retained": False,
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, decision
