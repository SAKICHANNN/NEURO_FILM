"""Zero-media Hershenson-Allen Velvia archive source-readiness audit."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class HershensonAllenSourceError(ValueError):
    """Raised when an official source violates the frozen contract."""


FetchResult = tuple[int, bytes]
Fetcher = Callable[[str], FetchResult]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _default_fetcher(url: str) -> FetchResult:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "K-MCFM-source-audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


class _VisibleTextAndLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._suppressed_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._suppressed_depth += 1
        if tag == "a":
            row = {key.lower(): value or "" for key, value in attrs}
            if row.get("href"):
                self.links.append(row["href"])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._suppressed_depth:
            self._suppressed_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self.text_parts.append(data)


def _parse_html(payload: bytes) -> tuple[str, list[str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HershensonAllenSourceError("official HTML is not UTF-8") from error
    parser = _VisibleTextAndLinks()
    parser.feed(text)
    return " ".join(" ".join(parser.text_parts).split()), sorted(set(parser.links))


def _canonical_html_identity(text: str, links: list[str]) -> str:
    stable_links = sorted(
        link
        for link in links
        if not link.startswith("/cdn-cgi/l/email-protection#")
    )
    return _json_sha256({"visible_text": text, "links": stable_links})


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _normalized_robots(text: str) -> str:
    return _normalized(
        " ".join(line.lstrip("# ").strip() for line in text.splitlines())
    )


def run_hershenson_allen_source_audit(
    config_path: Path,
    *,
    reverse: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Run the frozen official-text-only archive source-readiness audit."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    fetch = fetcher or _default_fetcher
    roles = ["home", "collection", "explore", "prospectus", "robots"]
    if reverse:
        roles.reverse()
    urls = {
        "home": source["home_url"],
        "collection": source["collection_url"],
        "explore": source["explore_url"],
        "prospectus": source["prospectus_page_url"],
        "robots": source["robots_url"],
    }
    fetched: dict[str, FetchResult] = {}
    for role in roles:
        fetched[role] = fetch(urls[role])

    if any(status != 200 for status, _ in fetched.values()):
        raise HershensonAllenSourceError("an official text source returned non-200")

    parsed = {
        role: _parse_html(payload)
        for role, (_, payload) in fetched.items()
        if role != "robots"
    }
    try:
        robots_text = fetched["robots"][1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise HershensonAllenSourceError("robots source is not UTF-8") from error
    expected = source["expected_responses"]
    observed_identities = {
        role: (
            _sha256(fetched[role][1])
            if expected[role]["identity_kind"] == "raw_bytes"
            else _canonical_html_identity(*parsed[role])
        )
        for role in sorted(expected)
    }
    response_identity_gates = {
        role: observed_identities[role] == expected[role]["sha256"]
        and (
            expected[role]["identity_kind"] != "raw_bytes"
            or len(fetched[role][1]) == int(expected[role]["bytes"])
        )
        for role in sorted(expected)
    }

    def phrases_present(role: str, phrases: list[str]) -> bool:
        text = parsed[role][0]
        return all(_normalized(phrase) in text for phrase in phrases)

    prospectus_links = parsed["prospectus"][1]
    prospectus_path = urlsplit(source["prospectus_pdf_url"]).path
    operation_counts = dict(config["operation_limits"])
    audit_gates = {
        "official_response_identities_exact": all(response_identity_gates.values()),
        "archive_counts_and_physical_workflow_exact": phrases_present(
            "home", source["required_home_phrases"]
        ),
        "collection_history_exact": phrases_present(
            "collection", source["required_collection_phrases"]
        ),
        "explore_inventory_statement_exact": phrases_present(
            "explore", source["required_explore_phrases"]
        ),
        "prospectus_link_exact_without_pdf_read": (
            source["prospectus_pdf_url"] in prospectus_links
            or prospectus_path in prospectus_links
        )
        and int(operation_counts["prospectus_pdf_requests"]) == 0,
        "robots_reference_only_and_no_ai_training_exact": all(
            _normalized(phrase) in _normalized_robots(robots_text)
            for phrase in source["required_robots_phrases"]
        ),
        "zero_database_media_pixel_and_model_reads": all(
            int(operation_counts[key]) == 0
            for key in (
                "prospectus_pdf_requests",
                "database_queries",
                "poster_image_requests",
                "transparency_requests",
                "thumbnail_requests",
                "media_body_requests",
                "pixel_decodes",
                "fit_calls",
                "render_calls",
                "score_calls",
            )
        ),
    }
    evidence = config["data_evidence"]
    minimums = config["admission_minimums"]
    admission_gates = {
        key: int(evidence[key]) >= int(minimums[key]) for key in sorted(minimums)
    }
    passed = all(audit_gates.values()) and all(admission_gates.values())
    source_identity = {
        "archive_image_count": int(source["archive_image_count"]),
        "unique_film_count": int(source["unique_film_count"]),
        "stock_example": source["stock_example"],
        "urls": {role: urls[role] for role in sorted(urls)},
        "canonical_response_sha256": observed_identities,
        "prospectus_pdf_url": source["prospectus_pdf_url"],
    }
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3z-hershenson-allen-velvia-archive-source-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source": source_identity,
        "source_identity_sha256": _json_sha256(source_identity),
        "official_statuses": {role: fetched[role][0] for role in sorted(fetched)},
        "response_identity_gates": response_identity_gates,
        "data_evidence": evidence,
        "admission_minimums": minimums,
        "audit_gates": audit_gates,
        "admission_gates": admission_gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _json_sha256(report)
    return report
