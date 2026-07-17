"""Keyless NASA/JSC exact-stock result-table snapshot for frozen SF2.0B0."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


class NasaStsStockSnapshotError(ValueError):
    """Raised when the SF2.0B0 query or result contract fails closed."""


class _TableRowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() == "tr":
            self._row = []
        elif tag.casefold() in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"th", "td"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _validate_config(config: Mapping[str, Any]) -> None:
    query = config["query"]
    if int(query["maximum_total_requests"]) != (
        int(query["maximum_post_requests"]) + int(query["maximum_result_get_requests"])
    ):
        raise NasaStsStockSnapshotError("request ceiling is internally inconsistent")
    if int(query["maximum_post_requests"]) < len(config["stocks"]):
        raise NasaStsStockSnapshotError("POST ceiling is below the frozen stock count")
    if int(query["maximum_result_get_requests"]) < len(config["stocks"]):
        raise NasaStsStockSnapshotError("GET ceiling is below the frozen stock count")
    for flag in (
        "image_payload_download_allowed",
        "photo_page_access_allowed",
        "operator_fitting_allowed",
        "training_allowed",
        "latent_mode_study_allowed",
    ):
        if config.get(flag) is not False:
            raise NasaStsStockSnapshotError(f"{flag} must remain false")
    if query.get("one_exact_film_code_per_query") is not True:
        raise NasaStsStockSnapshotError("one exact film code per query is required")
    if query.get("photo_page_requests_allowed") is not False:
        raise NasaStsStockSnapshotError("photo page requests must remain forbidden")
    if query.get("image_payload_requests_allowed") is not False or query.get("raw_html_retained") is not False:
        raise NasaStsStockSnapshotError("image requests and raw HTML retention must remain false")
    codes = [str(stock["film_code"]) for stock in config["stocks"]]
    ids = [str(stock["film_stock_id"]) for stock in config["stocks"]]
    if len(set(codes)) != len(codes) or len(set(ids)) != len(ids):
        raise NasaStsStockSnapshotError("stock codes and IDs must be unique")


def parse_result_table(body: bytes, config: Mapping[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Parse the one frozen NASA/JSC result table without retaining its HTML."""
    parser = _TableRowParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    required = [str(value) for value in config["result_contract"]["required_headers"]]
    header_index: int | None = None
    for index, row in enumerate(parser.rows):
        if row[: len(required)] == required:
            header_index = index
            break
    if header_index is None:
        raise NasaStsStockSnapshotError("required result headers are missing")
    pattern = re.compile(str(config["result_contract"]["photo_id_regex"]))
    records: list[dict[str, str]] = []
    for row in parser.rows[header_index + 1 :]:
        if len(row) < len(required) or not pattern.fullmatch(row[0]):
            continue
        records.append(dict(zip(required, row[: len(required)], strict=True)))
    if not records:
        raise NasaStsStockSnapshotError("result table contains no photo records")
    return required, records


def _bounded_response(
    response: requests.Response,
    *,
    maximum_bytes: int,
    required_content_type: str,
) -> tuple[bytes, dict[str, Any]]:
    body = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if chunk:
            body.extend(chunk)
        if len(body) > maximum_bytes:
            break
    bounded = len(body) <= maximum_bytes
    content_type = str(response.headers.get("Content-Type", ""))
    valid = (
        int(response.status_code) == 200
        and content_type.casefold().startswith(required_content_type.casefold())
        and bounded
    )
    evidence = {
        "status": int(response.status_code),
        "final_url": str(response.url),
        "content_type": content_type,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "request_utc": datetime.now(timezone.utc).isoformat(),
        "bounded_html_valid": valid,
    }
    return (bytes(body) if bounded else b""), evidence


def _request_with_retries(
    method: str,
    client: requests.Session,
    url: str,
    config: Mapping[str, Any],
    *,
    data: Mapping[str, str] | None,
    maximum_bytes: int,
    sleep_fn: Callable[[float], None],
) -> tuple[bytes, dict[str, Any]]:
    limits = config["request_limits"]
    retries = int(limits["request_retries"])
    last_error = ""
    for attempt_index in range(retries):
        response: requests.Response | None = None
        try:
            if method == "POST":
                response = client.post(
                    url,
                    data=data,
                    timeout=float(limits["timeout_seconds"]),
                    stream=True,
                    allow_redirects=True,
                )
            else:
                response = client.get(
                    url,
                    timeout=float(limits["timeout_seconds"]),
                    stream=True,
                    allow_redirects=True,
                )
            if response.status_code in {429, 500, 502, 503, 504} and attempt_index + 1 < retries:
                last_error = f"transient HTTP {response.status_code}"
                sleep_fn(float(limits["retry_backoff_seconds"]) * (2**attempt_index))
                continue
            body, evidence = _bounded_response(
                response,
                maximum_bytes=maximum_bytes,
                required_content_type=str(config["result_contract"]["required_content_type_prefix"]),
            )
            evidence.update({"method": method, "requested_url": url, "attempt_index": attempt_index})
            return body, evidence
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt_index + 1 < retries:
                sleep_fn(float(limits["retry_backoff_seconds"]) * (2**attempt_index))
        finally:
            if response is not None:
                response.close()
    return b"", {
        "method": method,
        "requested_url": url,
        "status": None,
        "final_url": None,
        "content_type": None,
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "request_utc": datetime.now(timezone.utc).isoformat(),
        "bounded_html_valid": False,
        "attempt_index": retries - 1,
        "error": last_error,
    }


def _is_nasa_endpoint(url: str, suffix: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and (parsed.hostname or "").casefold() == "eol.jsc.nasa.gov"
        and parsed.path.casefold().endswith(suffix.casefold())
    )


def _query_stock(
    client: requests.Session,
    stock: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    sleep_fn: Callable[[float], None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_url = str(config["source"]["query_endpoint"])
    if not _is_nasa_endpoint(query_url, "/Technical.pl"):
        raise NasaStsStockSnapshotError("query endpoint is outside NASA/JSC")
    form = {str(k): str(v) for k, v in config["query"]["form_fields"].items()}
    form["film"] = str(stock["film_code"])
    post_body, post_evidence = _request_with_retries(
        "POST",
        client,
        query_url,
        config,
        data=form,
        maximum_bytes=4096,
        sleep_fn=sleep_fn,
    )
    if not post_evidence["bounded_html_valid"] or not _is_nasa_endpoint(
        str(post_evidence["final_url"] or ""), "/Technical.pl"
    ):
        return {"stock": dict(stock), "query": post_evidence, "result": None, "error": "query_contract_mismatch"}, []
    match = re.search(
        rb'window\.location\.href="(ShowQueryResults-TextTable\.pl\?results=\d+)"',
        post_body,
    )
    if not match:
        return {"stock": dict(stock), "query": post_evidence, "result": None, "error": "query_contract_mismatch"}, []
    result_url = urljoin(str(config["source"]["result_base_url"]), match.group(1).decode("ascii"))
    if not _is_nasa_endpoint(result_url, "/ShowQueryResults-TextTable.pl"):
        raise NasaStsStockSnapshotError("result URL is outside NASA/JSC")
    sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
    result_body, result_evidence = _request_with_retries(
        "GET",
        client,
        result_url,
        config,
        data=None,
        maximum_bytes=int(config["result_contract"]["maximum_result_bytes"]),
        sleep_fn=sleep_fn,
    )
    if not result_evidence["bounded_html_valid"] or not _is_nasa_endpoint(
        str(result_evidence["final_url"] or ""), "/ShowQueryResults-TextTable.pl"
    ):
        return {
            "stock": dict(stock),
            "query": post_evidence,
            "result": result_evidence,
            "error": "source_unavailable",
        }, []
    try:
        headers, all_rows = parse_result_table(result_body, config)
    except NasaStsStockSnapshotError as exc:
        return {
            "stock": dict(stock),
            "query": post_evidence,
            "result": result_evidence,
            "error": "query_contract_mismatch",
            "detail": str(exc),
        }, []
    prefix = str(config["result_contract"].get("target_mission_prefix", ""))
    target_rows: list[dict[str, Any]] = []
    for row in all_rows:
        if row["Photo ID"].startswith(prefix):
            mission, roll, frame = row["Photo ID"].split("-", 2)
            target_rows.append(
                {
                    "film_code": str(stock["film_code"]),
                    "film_stock_id": str(stock["film_stock_id"]),
                    "mission": mission,
                    "roll": roll,
                    "frame": frame,
                    "photo_id": row["Photo ID"],
                    "photo_date": row["Photo Date"],
                    "latitude": row["Lat"],
                    "longitude": row["Lon"],
                    "geographic_name": row["Geographic Name"],
                    "features_manual": row["Features Identified Manually"],
                    "features_ml": row["Features Identified by Machine Learning"],
                    "focal_length_mm": row["Focal Length (mm)"],
                    "record_type": row["Record Type"],
                }
            )
    target_rows.sort(key=lambda row: (row["roll"], row["frame"], row["photo_id"]))
    evidence = {
        "stock": dict(stock),
        "query": post_evidence,
        "result": result_evidence,
        "required_headers": headers,
        "all_mission_rows": len(all_rows),
        "target_mission_rows": len(target_rows),
        "target_rolls": sorted({row["roll"] for row in target_rows}),
    }
    return evidence, target_rows


def decide_snapshot(
    query_evidence: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen B0 support/integrity decision only."""
    stock_results: dict[str, Any] = {}
    id_to_stocks: dict[str, set[str]] = {}
    for stock in config["stocks"]:
        stock_id = str(stock["film_stock_id"])
        rows = [row for row in records if str(row["film_stock_id"]) == stock_id]
        rolls = sorted({str(row["roll"]) for row in rows})
        stock_results[stock_id] = {
            "film_code": str(stock["film_code"]),
            "rows": len(rows),
            "rolls": rolls,
            "roll_count": len(rolls),
            "minimum_target_rows": int(stock["minimum_target_rows"]),
            "minimum_target_rolls": int(stock["minimum_target_rolls"]),
            "support_passed": len(rows) >= int(stock["minimum_target_rows"])
            and len(rolls) >= int(stock["minimum_target_rolls"]),
            "auxiliary_only": bool(stock.get("auxiliary_only", False)),
        }
        for row in rows:
            id_to_stocks.setdefault(str(row["photo_id"]), set()).add(stock_id)
    overlaps = sorted(photo_id for photo_id, stocks in id_to_stocks.items() if len(stocks) > 1)
    errors = [str(value.get("error")) for value in query_evidence if value.get("error")]
    if any(error == "query_contract_mismatch" for error in errors):
        decision = "query_contract_mismatch"
    elif errors:
        decision = "source_unavailable"
    elif overlaps:
        decision = "cross_stock_id_overlap"
    elif not all(result["support_passed"] for result in stock_results.values()):
        decision = "insufficient_stock_roll_support"
    else:
        decision = "snapshot_complete_open_connectivity_design"
    if decision not in config["allowed_decisions"]:
        raise NasaStsStockSnapshotError("decision is outside the frozen branch set")
    return {
        "schema_version": 1,
        "snapshot_id": config["snapshot_id"],
        "decision": decision,
        "stock_results": stock_results,
        "cross_stock_photo_id_overlap": overlaps,
        "query_errors": errors,
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def run_snapshot(
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run exactly one keyless table query per frozen film code."""
    _validate_config(config)
    client = session or requests.Session()
    client.headers["User-Agent"] = str(config["request_limits"]["user_agent"])
    evidence: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, stock in enumerate(config["stocks"]):
        item, rows = _query_stock(client, stock, config, sleep_fn=sleep_fn)
        evidence.append(item)
        records.extend(rows)
        if index + 1 < len(config["stocks"]):
            sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
    requests_made = sum(1 + int(item.get("result") is not None) for item in evidence)
    if requests_made > int(config["query"]["maximum_total_requests"]):
        raise NasaStsStockSnapshotError("frozen request ceiling exceeded")
    decision = decide_snapshot(evidence, records, config)
    report = {
        "schema_version": 1,
        "snapshot_id": config["snapshot_id"],
        "source": dict(config["source"]),
        "requests_made": requests_made,
        "query_evidence": evidence,
        "records": sorted(records, key=lambda row: (row["film_code"], row["roll"], row["frame"])),
        "summary": decision,
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, decision


def aggregate_cross_mission_records(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reduce transient frame rows to the only persistent mission/roll evidence."""
    gate = config["support_gate"]
    primary_ids = [str(value) for value in gate["primary_stock_ids"]]
    stock_ids = [str(stock["film_stock_id"]) for stock in config["stocks"]]
    minimum_rows_per_roll = int(gate["minimum_rows_per_supported_roll"])
    minimum_supported_rolls = int(gate["minimum_supported_rolls_per_primary_stock"])
    minimum_total_rows = int(gate["minimum_total_rows_per_primary_stock"])

    id_to_stocks: dict[str, set[str]] = {}
    mission_counts: dict[str, dict[str, dict[str, int]]] = {}
    for row in records:
        stock_id = str(row["film_stock_id"])
        mission = str(row["mission"])
        roll = str(row["roll"])
        photo_id = str(row["photo_id"])
        id_to_stocks.setdefault(photo_id, set()).add(stock_id)
        stock_counts = mission_counts.setdefault(mission, {}).setdefault(stock_id, {})
        stock_counts[roll] = stock_counts.get(roll, 0) + 1

    overlaps = sorted(photo_id for photo_id, values in id_to_stocks.items() if len(values) > 1)
    mission_aggregates: list[dict[str, Any]] = []
    for mission in sorted(mission_counts):
        stock_results: dict[str, Any] = {}
        for stock_id in stock_ids:
            roll_counts = mission_counts[mission].get(stock_id, {})
            rolls = [
                {
                    "roll": roll,
                    "rows": count,
                    "supported": count >= minimum_rows_per_roll,
                }
                for roll, count in sorted(roll_counts.items())
            ]
            total_rows = sum(roll_counts.values())
            supported_roll_count = sum(bool(item["supported"]) for item in rolls)
            stock_results[stock_id] = {
                "rows": total_rows,
                "roll_count": len(rolls),
                "supported_roll_count": supported_roll_count,
                "rolls": rolls,
            }
        primary_pass = {
            stock_id: stock_results[stock_id]["supported_roll_count"] >= minimum_supported_rolls
            and stock_results[stock_id]["rows"] >= minimum_total_rows
            for stock_id in primary_ids
        }
        eligible = all(primary_pass.values())
        item = {
            "mission": mission,
            "stocks": stock_results,
            "primary_pass": primary_pass,
            "eligible": eligible,
        }
        mission_aggregates.append(item)
    return mission_aggregates, overlaps


def decide_cross_mission_aggregates(
    query_evidence: Sequence[Mapping[str, Any]],
    mission_aggregates: Sequence[Mapping[str, Any]],
    cross_stock_photo_id_overlap: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen C0 branch using only persistent aggregate evidence."""
    gate = config["support_gate"]
    primary_ids = [str(value) for value in gate["primary_stock_ids"]]
    candidates: list[dict[str, Any]] = []
    for aggregate in mission_aggregates:
        if not bool(aggregate["eligible"]):
            continue
        stocks = aggregate["stocks"]
        candidates.append(
            {
                "mission": str(aggregate["mission"]),
                "minimum_primary_supported_roll_count": min(
                    int(stocks[stock_id]["supported_roll_count"]) for stock_id in primary_ids
                ),
                "minimum_primary_row_count": min(int(stocks[stock_id]["rows"]) for stock_id in primary_ids),
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(item["minimum_primary_supported_roll_count"]),
            -int(item["minimum_primary_row_count"]),
            str(item["mission"]),
        )
    )
    errors = [str(value.get("error")) for value in query_evidence if value.get("error")]
    if any(error == "query_contract_mismatch" for error in errors):
        decision_name = "query_contract_mismatch"
    elif errors:
        decision_name = "source_unavailable"
    elif cross_stock_photo_id_overlap:
        decision_name = "cross_stock_id_overlap"
    elif candidates:
        decision_name = "candidate_mission_found_open_metadata_connectivity"
    else:
        decision_name = "no_candidate_mission"
    if decision_name not in config["allowed_decisions"]:
        raise NasaStsStockSnapshotError("cross-mission decision is outside the frozen branch set")
    return {
        "schema_version": 1,
        "census_id": config["census_id"],
        "decision": decision_name,
        "candidate_missions": candidates,
        "selected_mission": candidates[0]["mission"] if candidates else None,
        "cross_stock_photo_id_overlap": sorted(str(value) for value in cross_stock_photo_id_overlap),
        "query_errors": errors,
        "support_gate": dict(gate),
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def run_cross_mission_census(
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate exact-code result tables by mission and roll without retaining frames."""
    _validate_config(config)
    if config["query"].get("frame_rows_retained") is not False:
        raise NasaStsStockSnapshotError("cross-mission census must not retain frame rows")
    primary_ids = [str(value) for value in config["support_gate"]["primary_stock_ids"]]
    known_ids = {str(stock["film_stock_id"]) for stock in config["stocks"]}
    if len(primary_ids) < 2 or not set(primary_ids).issubset(known_ids):
        raise NasaStsStockSnapshotError("primary stock IDs are invalid")

    client = session or requests.Session()
    client.headers["User-Agent"] = str(config["request_limits"]["user_agent"])
    evidence: list[dict[str, Any]] = []
    transient_records: list[dict[str, Any]] = []
    for index, stock in enumerate(config["stocks"]):
        item, rows = _query_stock(client, stock, config, sleep_fn=sleep_fn)
        evidence.append(item)
        transient_records.extend(rows)
        if index + 1 < len(config["stocks"]):
            sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
    requests_made = sum(1 + int(item.get("result") is not None) for item in evidence)
    if requests_made > int(config["query"]["maximum_total_requests"]):
        raise NasaStsStockSnapshotError("frozen request ceiling exceeded")

    mission_aggregates, overlaps = aggregate_cross_mission_records(transient_records, config)
    query_summary = [
        {
            "stock": item["stock"],
            "query": item["query"],
            "result": item.get("result"),
            "error": item.get("error"),
            "required_headers": item.get("required_headers"),
            "all_mission_rows": item.get("all_mission_rows", 0),
        }
        for item in evidence
    ]
    decision = decide_cross_mission_aggregates(query_summary, mission_aggregates, overlaps, config)
    report = {
        "schema_version": 1,
        "census_id": config["census_id"],
        "source": dict(config["source"]),
        "requests_made": requests_made,
        "query_evidence": query_summary,
        "mission_aggregates": mission_aggregates,
        "summary": decision,
        "frame_rows_retained": False,
        "raw_html_retained": False,
        "photo_page_access_allowed": False,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, decision
