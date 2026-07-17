"""Bounded NASA/JSC HTML metadata audit for the frozen SF2.0A Apollo leaf."""

from __future__ import annotations

import hashlib
import html
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


class ApolloMetadataFeasibilityError(ValueError):
    """Raised when the frozen SF2.0A contract fails closed."""


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _plain(value: str) -> str:
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value))).strip()


def _field(body: str, label: str) -> str:
    pattern = re.compile(
        rf"<b>\s*{re.escape(label)}\s*:?</b>.*?</td>\s*<td[^>]*>(.*?)</td>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(body)
    return _plain(match.group(1)) if match else ""


def _caption(body: str) -> str:
    match = re.search(
        r"<b>\s*Image Caption\s*</b>\s*</em>\s*:\s*(.*?)</div>",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    return _plain(match.group(1)) if match else ""


def _offered_images(body: str, base_url: str) -> list[dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for href in re.findall(r"href=[\"']([^\"']+\.(?:jpe?g|png|tiff?))(?:\?[^\"']*)?[\"']", body, re.IGNORECASE):
        absolute = urljoin(base_url, html.unescape(href))
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        values[absolute] = {"url": absolute, "suffix": parsed.path.rsplit(".", 1)[-1].casefold()}
    return [values[key] for key in sorted(values)]


def build_sample(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build and validate the exact deterministic 63-page sample."""
    selection = config["selection"]
    if selection.get("image_urls_requested") is not False or selection.get("html_bodies_retained") is not False:
        raise ApolloMetadataFeasibilityError("sample violates the page-only/no-body contract")
    for flag in (
        "image_payload_download_allowed",
        "operator_fitting_allowed",
        "training_allowed",
        "latent_mode_study_allowed",
    ):
        if config.get(flag) is not False:
            raise ApolloMetadataFeasibilityError(f"{flag} must remain false")
    count = int(selection["frames_per_magazine"])
    if count < 2:
        raise ApolloMetadataFeasibilityError("frames_per_magazine must be at least two")
    template = str(config["source"]["photo_page_template"])
    mission = str(config["source"]["mission"])
    rows: list[dict[str, Any]] = []
    for magazine in config["magazines"]:
        start = int(magazine["frame_start"])
        end = int(magazine["frame_end"])
        if end < start:
            raise ApolloMetadataFeasibilityError("invalid frame range")
        frames = [start + (index * (end - start)) // (count - 1) for index in range(count)]
        if len(set(frames)) != count:
            raise ApolloMetadataFeasibilityError("deterministic frame sample is not unique")
        for frame in frames:
            roll = str(magazine["photo_roll"])
            url = template.format(mission=mission, roll=roll, frame=frame)
            parsed = urlparse(url)
            if parsed.scheme != "https" or (parsed.hostname or "").casefold() != "eol.jsc.nasa.gov":
                raise ApolloMetadataFeasibilityError("photo page is outside the frozen NASA/JSC host")
            rows.append(
                {
                    **dict(magazine),
                    "mission": mission,
                    "frame": frame,
                    "nasa_photo_id": f"{mission}-{roll}-{frame}",
                    "page_url": url,
                }
            )
    maximum = int(selection["maximum_page_requests"])
    expected = int(selection["expected_page_requests"])
    if len(rows) != expected or len(rows) > maximum:
        raise ApolloMetadataFeasibilityError("sample does not match the frozen request count")
    if len({row["page_url"] for row in rows}) != len(rows):
        raise ApolloMetadataFeasibilityError("sample contains duplicate page URLs")
    return rows


def parse_photo_page(body: bytes, row: Mapping[str, Any], final_url: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only preregistered metadata from one bounded HTML response."""
    text = body.decode("utf-8", errors="replace")
    photo_match = re.search(
        r"NASA Photo ID(?:\s|&nbsp;|<[^>]+>)*([A-Z]{2}\d{2}-\d+-\d+)",
        text,
        re.IGNORECASE,
    )
    if not photo_match:
        photo_match = re.search(r"<h1[^>]*>.*?([A-Z]{2}\d{2}-\d+-\d+).*?</h1>", text, re.IGNORECASE | re.DOTALL)
    observed_id = photo_match.group(1).upper() if photo_match else ""
    format_text = _field(text, "Format")
    film_match = re.match(r"\s*([A-Z0-9-]+)\s*:", format_text, re.IGNORECASE)
    observed_film = film_match.group(1).upper().replace("-", "") if film_match else ""
    expected_film = str(row["film_code"]).upper().replace("-", "")
    geographic_name = _field(text, "Country or Geographic Name")
    features = _field(text, "Features")
    caption = _caption(text)
    evidence = " ".join((geographic_name, features, caption)).casefold()
    tags = [
        str(tag)
        for tag, keywords in config["content_tags"].items()
        if any(str(keyword).casefold() in evidence for keyword in keywords)
    ]
    return {
        "observed_nasa_photo_id": observed_id,
        "observed_film_code": observed_film,
        "format_text": format_text,
        "film_exposure": _field(text, "Film Exposure") or "unknown",
        "geographic_name": geographic_name,
        "features": features,
        "caption": caption,
        "content_tags": sorted(tags),
        "offered_image_metadata_only": _offered_images(text, final_url),
        "photo_id_match": observed_id == str(row["nasa_photo_id"]).upper(),
        "film_code_match": observed_film == expected_film,
    }


def _request_page(
    client: requests.Session,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    limits = config["request_limits"]
    page_contract = config["page_contract"]
    retries = int(limits["request_retries"])
    maximum_bytes = int(page_contract["maximum_html_bytes"])
    last_error = ""
    for attempt_index in range(retries):
        response: requests.Response | None = None
        try:
            response = client.get(
                str(row["page_url"]),
                timeout=float(limits["timeout_seconds"]),
                stream=True,
                allow_redirects=True,
            )
            status = int(response.status_code)
            content_type = str(response.headers.get("Content-Type", ""))
            final_url = str(response.url)
            final_parsed = urlparse(final_url)
            final_is_nasa_page = (
                final_parsed.scheme == "https"
                and (final_parsed.hostname or "").casefold() == "eol.jsc.nasa.gov"
                and final_parsed.path.casefold().endswith("/photo.pl")
            )
            if status in {429, 500, 502, 503, 504} and attempt_index + 1 < retries:
                last_error = f"transient HTTP {status}"
                sleep_fn(float(limits["retry_backoff_seconds"]) * (2**attempt_index))
                continue
            body = bytearray()
            if status == int(page_contract["required_status"]) and content_type.casefold().startswith(
                str(page_contract["required_content_type_prefix"]).casefold()
            ):
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        body.extend(chunk)
                    if len(body) > maximum_bytes:
                        break
            bounded = len(body) <= maximum_bytes
            parsed: dict[str, Any] = {}
            if status == 200 and content_type.casefold().startswith("text/html") and bounded and final_is_nasa_page:
                parsed = parse_photo_page(bytes(body), row, final_url, config)
            valid = bool(parsed.get("photo_id_match") and parsed.get("film_code_match"))
            return {
                **dict(row),
                "final_url": final_url,
                "page_status": status,
                "content_type": content_type,
                "html_bytes": len(body),
                "page_sha256": hashlib.sha256(body).hexdigest(),
                "request_utc": datetime.now(timezone.utc).isoformat(),
                "attempt_index": attempt_index,
                "valid_metadata_page": valid,
                "request_decision": "retain_metadata" if valid else "reject_metadata",
                "final_is_nasa_photo_page": final_is_nasa_page,
                **parsed,
            }
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt_index + 1 < retries:
                sleep_fn(float(limits["retry_backoff_seconds"]) * (2**attempt_index))
        finally:
            if response is not None:
                response.close()
    return {
        **dict(row),
        "final_url": None,
        "page_status": None,
        "content_type": None,
        "html_bytes": 0,
        "page_sha256": hashlib.sha256(b"").hexdigest(),
        "request_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_index": retries - 1,
        "valid_metadata_page": False,
        "request_decision": "reject_request_failure",
        "reason": last_error,
        "content_tags": [],
        "offered_image_metadata_only": [],
    }


def decide_metadata_feasibility(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the frozen metadata support/connectivity gates."""
    gates = config["decision_gates"]
    valid = [row for row in records if row.get("valid_metadata_page") is True]
    mismatches = [
        str(row["nasa_photo_id"])
        for row in records
        if row.get("page_status") == 200
        and (row.get("photo_id_match") is False or row.get("film_code_match") is False)
    ]
    stocks = sorted({str(row["film_stock_id"]) for row in config["magazines"]})
    rows_by_stock = {stock: sum(str(row["film_stock_id"]) == stock for row in valid) for stock in stocks}
    rows_by_magazine = {
        str(mag["physical_magazine"]): sum(
            str(row["physical_magazine"]) == str(mag["physical_magazine"]) for row in valid
        )
        for mag in config["magazines"]
    }
    magazines_by_stock = {
        stock: sorted({str(row["physical_magazine"]) for row in valid if str(row["film_stock_id"]) == stock})
        for stock in stocks
    }
    filter_free_rows = {
        stock: sum(
            str(row["film_stock_id"]) == stock and str(row["filter"]) == "none" for row in valid
        )
        for stock in stocks
    }
    tag_support: dict[str, dict[str, dict[str, Any]]] = {}
    for tag in config["content_tags"]:
        tag_support[str(tag)] = {}
        for stock in stocks:
            tagged = [
                row
                for row in valid
                if str(row["film_stock_id"]) == stock and str(tag) in row.get("content_tags", [])
            ]
            tag_support[str(tag)][stock] = {
                "rows": len(tagged),
                "magazines": sorted({str(row["physical_magazine"]) for row in tagged}),
            }
    shared_tags = [
        tag
        for tag, support in tag_support.items()
        if all(
            support[stock]["rows"] >= int(gates["minimum_rows_per_stock_per_shared_tag"])
            and len(support[stock]["magazines"]) >= int(gates["minimum_magazines_per_stock_per_shared_tag"])
            for stock in stocks
        )
    ]
    all_exposure_states = sorted(
        {str(row.get("film_exposure") or "unknown") for row in valid}, key=str.casefold
    )
    exposure_states = [value for value in all_exposure_states if value.casefold() != "unknown"]
    support_pass = (
        all(value >= int(gates["minimum_valid_pages_per_stock"]) for value in rows_by_stock.values())
        and all(value >= int(gates["minimum_valid_pages_per_magazine"]) for value in rows_by_magazine.values())
        and all(
            len(value) >= int(gates["minimum_independent_magazines_per_stock"])
            for value in magazines_by_stock.values()
        )
        and len(exposure_states) >= int(gates["minimum_distinct_reported_exposure_states"])
    )
    filter_pass = all(
        value >= int(gates["minimum_filter_free_rows_per_stock"])
        for value in filter_free_rows.values()
    )
    content_pass = len(shared_tags) >= int(gates["minimum_shared_content_tags"])
    if mismatches:
        decision = "metadata_mismatch"
    elif not valid:
        decision = "source_unavailable"
    elif not support_pass:
        decision = "insufficient_connectivity"
    elif not filter_pass:
        decision = "stock_filter_confounded"
    elif not content_pass:
        decision = "content_confounded"
    else:
        decision = "metadata_connectivity_candidate"
    if decision not in config["allowed_decisions"]:
        raise ApolloMetadataFeasibilityError("decision is outside the frozen branch set")
    return {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "decision": decision,
        "valid_pages": len(valid),
        "metadata_mismatch_photo_ids": sorted(mismatches),
        "rows_by_stock": rows_by_stock,
        "rows_by_magazine": rows_by_magazine,
        "magazines_by_stock": magazines_by_stock,
        "filter_free_rows_by_stock": filter_free_rows,
        "exposure_states": exposure_states,
        "unknown_exposure_pages": sum(
            str(row.get("film_exposure") or "unknown").casefold() == "unknown" for row in valid
        ),
        "content_tag_support": tag_support,
        "shared_content_tags": sorted(shared_tags),
        "support_gate_passed": support_pass,
        "filter_bridge_gate_passed": filter_pass,
        "content_connectivity_gate_passed": content_pass,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }


def run_metadata_feasibility_audit(
    config: Mapping[str, Any],
    *,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Request only the frozen HTML sample and return report plus decision."""
    sample = build_sample(config)
    client = session or requests.Session()
    client.headers["User-Agent"] = str(config["request_limits"]["user_agent"])
    records: list[dict[str, Any]] = []
    for row in sample:
        records.append(_request_page(client, row, config, sleep_fn=sleep_fn))
        sleep_fn(float(config["request_limits"]["request_interval_seconds"]))
    decision = decide_metadata_feasibility(records, config)
    report = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "source": dict(config["source"]),
        "page_requests": len(records),
        "records": records,
        "summary": decision,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "latent_mode_study_allowed": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    return report, decision
