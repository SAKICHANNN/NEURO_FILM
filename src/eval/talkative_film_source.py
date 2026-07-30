"""Metadata-only audit of the Talkative Photographer controlled film source."""

from __future__ import annotations

import hashlib
import html
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Any

from src.eval.filmmatch_paired_source import canonical_sha256


class TalkativeSourceAuditError(RuntimeError):
    """Raised when the bounded live metadata source violates its contract."""


def _fetch_html(url: str, config: Mapping[str, Any]) -> str:
    network = config["network_contract"]
    request = urllib.request.Request(
        url,
        headers={"User-Agent": str(network["user_agent"])},
        method="GET",
    )
    maximum = int(network["maximum_response_bytes_per_page"])
    with urllib.request.urlopen(  # noqa: S310 - exact HTTPS URLs are config-bound
        request, timeout=float(network["timeout_seconds"])
    ) as response:
        final_url = str(response.geturl())
        if urllib.parse.urlsplit(final_url).scheme != "https":
            raise TalkativeSourceAuditError("live source redirected off HTTPS")
        payload = response.read(maximum + 1)
        if len(payload) > maximum:
            raise TalkativeSourceAuditError("live HTML response exceeds byte cap")
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "text/" not in content_type and "xml" not in content_type:
            raise TalkativeSourceAuditError("live source returned non-text payload")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TalkativeSourceAuditError("live source is not UTF-8") from exc


def catalogue_page_urls(html_text: str, config: Mapping[str, Any]) -> list[str]:
    prefix = str(config["source"]["allowed_page_prefix"])
    root = urllib.parse.urlsplit(prefix)
    pattern = re.compile(
        rf"(?:{re.escape(root.scheme + '://' + root.netloc)})?"
        rf"{re.escape(root.path)}[a-z0-9-]+"
    )
    output = sorted(
        {
            urllib.parse.urljoin(prefix, match.group(0))
            for match in pattern.finditer(html_text)
        }
    )
    maximum = int(config["network_contract"]["maximum_catalogue_pages"])
    if not output or len(output) > maximum:
        raise TalkativeSourceAuditError("catalogue page count escapes bound")
    if any(not url.startswith(prefix) for url in output):
        raise TalkativeSourceAuditError("catalogue emitted an unapproved page URL")
    return output


def _canonical_image_urls(html_text: str) -> list[str]:
    pattern = re.compile(
        r"https://images\.squarespace-cdn\.com/content/v1/"
        r"[^\"'<>\\\s]+",
        flags=re.IGNORECASE,
    )
    output = set()
    for match in pattern.finditer(html_text):
        value = html.unescape(match.group(0))
        parsed = urllib.parse.urlsplit(value)
        output.add(
            urllib.parse.urlunsplit(
                (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "")
            )
        )
    return sorted(output)


def parse_film_page(
    page_url: str, html_text: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    title_match = re.search(
        r"<title[^>]*>(.*?)</title>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    title = (
        re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip()
        if title_match
        else ""
    )
    controlled_pattern = re.compile(
        str(config["parser_contract"]["controlled_filename_pattern"])
    )
    scanner_patterns = {
        name: re.compile(pattern)
        for name, pattern in config["parser_contract"][
            "scanner_variant_patterns"
        ].items()
    }
    controlled: list[dict[str, Any]] = []
    for url in _canonical_image_urls(html_text):
        filename = urllib.parse.unquote(
            PurePosixPath(urllib.parse.urlsplit(url).path).name
        )
        if not controlled_pattern.search(filename):
            continue
        slot_match = re.search(r"(?:LS600_)?([1-5])\.[^.]+$", filename, re.I)
        scanner_variant = next(
            (
                name
                for name, pattern in scanner_patterns.items()
                if pattern.search(filename)
            ),
            "unclassified",
        )
        controlled.append(
            {
                "url": url,
                "filename": filename,
                "slot": int(slot_match.group(1)) if slot_match else None,
                "scanner_variant": scanner_variant,
            }
        )
    licence_urls = sorted(
        set(
            re.findall(
                r"https://creativecommons\.org/licenses/[^\"'<>\\\s]+",
                html.unescape(html_text),
                flags=re.IGNORECASE,
            )
        )
    )
    by_variant: dict[str, list[int]] = {}
    for row in controlled:
        if row["slot"] is not None:
            by_variant.setdefault(row["scanner_variant"], []).append(row["slot"])
    expected = list(config["parser_contract"]["expected_bracket_slots"])
    complete_variants = sorted(
        name
        for name, slots in by_variant.items()
        if sorted(set(slots)) == expected
    )
    return {
        "page_url": page_url,
        "stock_slug": page_url.rstrip("/").rsplit("/", 1)[-1],
        "page_title": title,
        "controlled_images": controlled,
        "controlled_image_count": len(controlled),
        "scanner_variant_slots": {
            name: sorted(slots) for name, slots in sorted(by_variant.items())
        },
        "complete_scanner_variants": complete_variants,
        "explicit_creative_commons_licence_urls": licence_urls,
    }


def _generic_robots_allows_film_pages(robots_text: str) -> bool:
    active = False
    disallowed: list[str] = []
    for raw_line in robots_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key.lower() == "user-agent":
            active = value == "*"
        elif active and key.lower() == "disallow" and value:
            disallowed.append(value)
    return not any(
        path == "/" or "/film-samples".startswith(path) for path in disallowed
    )


def audit_live_source(
    config: Mapping[str, Any],
    *,
    fetcher: Callable[[str, Mapping[str, Any]], str] = _fetch_html,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    robots = fetcher(str(config["source"]["robots_url"]), config)
    if not _generic_robots_allows_film_pages(robots):
        raise TalkativeSourceAuditError(
            "generic robots policy disallows film sample pages"
        )
    catalogue = fetcher(str(config["source"]["catalogue_url"]), config)
    page_urls = catalogue_page_urls(catalogue, config)
    pages = []
    delay = float(config["network_contract"]["minimum_delay_seconds"])
    for index, page_url in enumerate(page_urls):
        if index:
            sleeper(delay)
        pages.append(parse_film_page(page_url, fetcher(page_url, config), config))

    controlled_urls = [
        row["url"]
        for page in pages
        for row in page["controlled_images"]
    ]
    duplicate_urls = sorted(
        url for url, count in Counter(controlled_urls).items() if count > 1
    )
    complete_pages = [
        page for page in pages if page["complete_scanner_variants"]
    ]
    two_scanner_pages = [
        page
        for page in pages
        if len(page["complete_scanner_variants"]) >= 2
    ]
    gates = config["structure_gates"]
    structure_checks = {
        "minimum_catalogue_pages": len(pages)
        >= int(gates["minimum_catalogue_pages"]),
        "minimum_stocks_with_complete_five_frame_bracket": len(complete_pages)
        >= int(gates["minimum_stocks_with_complete_five_frame_bracket"]),
        "minimum_stocks_with_two_complete_scanner_variants": len(
            two_scanner_pages
        )
        >= int(gates["minimum_stocks_with_two_complete_scanner_variants"]),
        "maximum_duplicate_canonical_image_urls": len(duplicate_urls)
        <= int(gates["maximum_duplicate_canonical_image_urls"]),
    }
    licensed_complete_pages = [
        page
        for page in complete_pages
        if page["explicit_creative_commons_licence_urls"]
    ]
    rights_passed = bool(
        complete_pages and len(licensed_complete_pages) == len(complete_pages)
    )
    structure_passed = bool(all(structure_checks.values()))
    if not structure_passed:
        decision = "structure_failed_no_pixel_access"
    elif not rights_passed:
        decision = "structure_passed_rights_blocked_metadata_only"
    else:
        decision = "structure_and_rights_passed_separate_pixel_contract_required"
    report = {
        "schema_version": "u5-r2bg0-talkative-controlled-film-source-report-v1",
        "experiment_id": config["experiment_id"],
        "network_observation": {
            "html_get_requests": 2 + len(pages),
            "image_requests": 0,
            "api_requests": 0,
            "generic_robots_allows_film_pages": True,
            "robots_sha256": hashlib.sha256(
                robots.encode("utf-8")
            ).hexdigest(),
        },
        "catalogue_page_count": len(pages),
        "controlled_image_url_count": len(controlled_urls),
        "stocks_with_complete_five_frame_bracket": [
            page["stock_slug"] for page in complete_pages
        ],
        "stocks_with_two_complete_scanner_variants": [
            page["stock_slug"] for page in two_scanner_pages
        ],
        "duplicate_controlled_image_urls": duplicate_urls,
        "pages": pages,
        "structure_checks": structure_checks,
        "structure_passed": structure_passed,
        "rights": {
            "complete_bracket_stocks_with_explicit_cc_licence": [
                page["stock_slug"] for page in licensed_complete_pages
            ],
            "future_open_source_intent_counted_as_licence": False,
            "public_display_counted_as_licence": False,
            "passed": rights_passed,
        },
        "decision": decision,
        "pixel_access_opened": bool(structure_passed and rights_passed),
        "operator_fitting_opened": False,
        "training_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = [
    "TalkativeSourceAuditError",
    "audit_live_source",
    "catalogue_page_urls",
    "parse_film_page",
]
