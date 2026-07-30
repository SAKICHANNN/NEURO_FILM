from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.talkative_film_source import (
    TalkativeSourceAuditError,
    audit_live_source,
    catalogue_page_urls,
    parse_film_page,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2bg0_talkative_controlled_film_source_v1.json"
    ).read_text(encoding="utf-8")
)


def _page(stock: str, *, second_scanner: bool = False, licensed: bool = False) -> str:
    rows = []
    for slot in range(1, 6):
        rows.append(
            f"https://images.squarespace-cdn.com/content/v1/x/y/"
            f"{stock}_5FrameSample_{slot}.jpg?format=1000w"
        )
        if second_scanner:
            rows.append(
                f"https://images.squarespace-cdn.com/content/v1/x/z/"
                f"{stock}_5FrameSample_LS600_{slot}.jpg?format=100w"
            )
    licence = (
        "https://creativecommons.org/licenses/by/4.0/" if licensed else ""
    )
    return (
        f"<html><head><title>{stock} Sample Images</title></head><body>"
        + " ".join(rows)
        + licence
        + "</body></html>"
    )


def test_parser_deduplicates_query_derivatives_and_finds_two_scanners() -> None:
    text = _page("Stock", second_scanner=True)
    text += (
        " https://images.squarespace-cdn.com/content/v1/x/y/"
        "Stock_5FrameSample_1.jpg?format=2500w"
    )
    row = parse_film_page(
        "https://talkativephotographer.com/film-samples/stock", text, CONFIG
    )
    assert row["controlled_image_count"] == 10
    assert row["complete_scanner_variants"] == [
        "noritsu_ls600",
        "other_or_unlabelled",
    ]


def test_catalogue_rejects_page_count_over_bound() -> None:
    changed = json.loads(json.dumps(CONFIG))
    changed["network_contract"]["maximum_catalogue_pages"] = 1
    with pytest.raises(TalkativeSourceAuditError):
        catalogue_page_urls(
            '<a href="/film-samples/one"></a><a href="/film-samples/two"></a>',
            changed,
        )


def test_live_audit_separates_structure_from_rights_without_image_requests() -> None:
    changed = json.loads(json.dumps(CONFIG))
    changed["structure_gates"]["minimum_catalogue_pages"] = 2
    changed["structure_gates"][
        "minimum_stocks_with_complete_five_frame_bracket"
    ] = 2
    catalogue = (
        '<a href="/film-samples/one"></a>'
        '<a href="/film-samples/two"></a>'
    )
    payloads = {
        changed["source"]["robots_url"]: "User-agent: *\nDisallow: /api/\n",
        changed["source"]["catalogue_url"]: catalogue,
        "https://talkativephotographer.com/film-samples/one": _page(
            "One", second_scanner=True
        ),
        "https://talkativephotographer.com/film-samples/two": _page("Two"),
    }
    requests: list[str] = []

    def fetch(url: str, _config: dict) -> str:
        requests.append(url)
        return payloads[url]

    report = audit_live_source(
        changed, fetcher=fetch, sleeper=lambda _seconds: None
    )
    assert report["structure_passed"]
    assert not report["rights"]["passed"]
    assert not report["pixel_access_opened"]
    assert report["decision"] == "structure_passed_rights_blocked_metadata_only"
    assert all("images.squarespace-cdn.com" not in url for url in requests)
