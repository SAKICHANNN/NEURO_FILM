from __future__ import annotations

import json

import pytest

from src.eval.flickr_analog_digital_source import (
    FlickrPairAuditError,
    PoolRow,
    candidate_adjacent_pairs,
    parse_photo_page,
    parse_pool_page,
    summarize_audit,
)


def _photo_html(
    photo_id: str,
    *,
    title: str,
    description: str,
    license_id: int,
    keywords: str = "",
) -> str:
    model = {
        "photoModel": {
            "id": photo_id,
            "title": title,
            "description": description,
            "license": license_id,
            "owner": {"nsid": "owner@N00"},
            "sizes": {"o": {"width": 100, "height": 80}},
            "oWidth": 100,
            "oHeight": 80,
        }
    }
    return (
        f'<meta name="keywords" content="{keywords}">'
        '"dateTaken":"2020-01-02 03:04:05" '
        f"params: {json.dumps(model)}"
    )


def test_pool_parser_preserves_order_and_deduplicates_render_links() -> None:
    html = """
    <a href="/photos/a/101/in/pool-lomoplusdigital" title="Film &amp; one">
    <a href="/photos/a/101/in/pool-lomoplusdigital" title="duplicate">
    <a href="/photos/a/102/in/pool-lomoplusdigital" title="Digital">
    """
    rows = parse_pool_page(html, page=1)
    assert [row.photo_id for row in rows] == ["101", "102"]
    assert rows[0].pool_title == "Film & one"
    assert [row.position for row in rows] == [0, 1]


def test_photo_parser_classifies_explicit_media_and_rights() -> None:
    film = parse_photo_page(
        _photo_html(
            "101",
            title="I shot film",
            description="Scan Kodak Gold 200",
            license_id=4,
        ),
        PoolRow(1, 0, "owner", "101", "I shot film"),
    )
    digital = parse_photo_page(
        _photo_html(
            "102",
            title="digital variant",
            description="Digitalfoto",
            license_id=0,
            keywords="nikon, d800, dslr",
        ),
        PoolRow(1, 1, "owner", "102", "digital variant"),
    )
    assert film["film_explicit"] is True
    assert film["derivative_rights_eligible"] is True
    assert digital["digital_explicit"] is True
    assert digital["digital_camera_metadata"] is True
    assert digital["derivative_rights_eligible"] is False


def test_pairing_requires_same_owner_page_and_explicit_opposite_media() -> None:
    base = {
        "page": 1,
        "owner_nsid": "owner",
        "license_id": 4,
        "derivative_rights_eligible": True,
        "digital_camera_metadata": False,
        "film_metadata": False,
    }
    records = [
        {
            **base,
            "position": 0,
            "photo_id": "film",
            "film_explicit": True,
            "digital_explicit": False,
        },
        {
            **base,
            "position": 1,
            "photo_id": "digital",
            "film_explicit": False,
            "digital_explicit": True,
        },
        {
            **base,
            "position": 2,
            "photo_id": "ambiguous",
            "film_explicit": False,
            "digital_explicit": False,
        },
    ]
    pairs = candidate_adjacent_pairs(records)
    assert pairs == [
        {
            "owner_nsid": "owner",
            "film_photo_id": "film",
            "digital_photo_id": "digital",
            "both_derivative_rights_eligible": True,
        }
    ]
    summary = summarize_audit(records)
    assert summary["rights_eligible_explicit_pair_count"] == 1
    assert summary["decision"] == "open_bounded_pixel_preflight"


def test_summary_closes_when_rights_and_pair_evidence_do_not_intersect() -> None:
    records = [
        {
            "page": 1,
            "position": 0,
            "owner_nsid": "rights-owner",
            "photo_id": "digital-only",
            "license_id": 2,
            "derivative_rights_eligible": True,
            "film_explicit": False,
            "digital_explicit": False,
            "digital_camera_metadata": True,
            "film_metadata": False,
        },
        {
            "page": 1,
            "position": 1,
            "owner_nsid": "pair-owner",
            "photo_id": "film",
            "license_id": 0,
            "derivative_rights_eligible": False,
            "film_explicit": True,
            "digital_explicit": False,
            "digital_camera_metadata": False,
            "film_metadata": True,
        },
        {
            "page": 1,
            "position": 2,
            "owner_nsid": "pair-owner",
            "photo_id": "digital",
            "license_id": 0,
            "derivative_rights_eligible": False,
            "film_explicit": False,
            "digital_explicit": True,
            "digital_camera_metadata": False,
            "film_metadata": False,
        },
    ]
    summary = summarize_audit(records)
    assert summary["explicit_adjacent_candidate_pair_count"] == 1
    assert summary["rights_eligible_explicit_pair_count"] == 0
    assert (
        summary["decision"]
        == "close_current_visible_pool_no_rights_eligible_explicit_pair"
    )


def test_parser_rejects_drift_and_unbounded_pages() -> None:
    with pytest.raises(FlickrPairAuditError, match="pages 1-2"):
        parse_pool_page("<a>", page=3)
    with pytest.raises(FlickrPairAuditError, match="photo model missing"):
        parse_photo_page(
            "<html></html>",
            PoolRow(1, 0, "owner", "101", "title"),
        )
