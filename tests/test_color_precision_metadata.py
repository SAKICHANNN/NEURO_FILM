from __future__ import annotations

from src.real_film.color_precision_metadata import (
    extract_comparison_image_urls,
    inventory_comparison_metadata,
    normalize_dynamic_html,
    parse_comparison_url,
)


HOST = "https://colorprecision-cdn.s3.us-east-1.amazonaws.com"


def test_dynamic_shopify_token_normalization_is_narrow() -> None:
    left = b'<meta name="shopify-y" content="abc"><p>stable</p>'
    right = b'<meta name="shopify-y" content="xyz"><p>stable</p>'
    assert normalize_dynamic_html(left) == normalize_dynamic_html(right)
    assert b"<p>stable</p>" in normalize_dynamic_html(left)


def test_extract_urls_deduplicates_without_fetching() -> None:
    url = (
        f"{HOST}/Pictures-for-Slider-Comparison/Charts/"
        "Gold-0EV-Frontier.png"
    )
    payload = f'<option value="{url}"></option><img src="{url}">'.encode()
    assert extract_comparison_image_urls(payload) == [url]


def test_parse_url_canonicalizes_stock_and_scanner() -> None:
    url = (
        f"{HOST}/Exterior-Film-Scans-Tool/Night/"
        "Velvia-50-plus1EV-Noritsu.jpg"
    )
    record = parse_comparison_url(url)
    assert record["stock_id"] == "Velvia50"
    assert record["scanner"] == "Noritsu"
    assert record["scene"] == "Night"
    assert record["condition_key"] == "velvia-50-plus1ev"


def test_inventory_counts_complete_and_incomplete_filename_pairs() -> None:
    urls = [
        (
            f"{HOST}/Pictures-for-Slider-Comparison/Charts/"
            "Gold-0EV-Frontier.png"
        ),
        (
            f"{HOST}/Pictures-for-Slider-Comparison/Charts/"
            "Gold-0EV-Noritsu.png"
        ),
        (
            f"{HOST}/Pictures-for-Slider-Comparison/Charts/"
            "Ektar-plus1EV-Frontier.png"
        ),
    ]
    payload = "".join(f'<option value="{url}"></option>' for url in urls).encode()
    inventory = inventory_comparison_metadata(payload)
    assert inventory["embedded_image_url_count"] == 3
    assert inventory["filename_implied_condition_count"] == 2
    assert inventory["filename_implied_complete_scanner_pair_count"] == 1
    assert inventory["filename_implied_incomplete_scanner_pair_count"] == 1
    assert inventory["stock_label_counts"] == {"Ektar": 1, "Gold": 2}


def test_alias_typos_remain_separate_pair_candidates() -> None:
    urls = [
        (
            f"{HOST}/Exterior-Film-Scans-Tool/Night/"
            "NHG-II-800-plus1EV-Frontier.jpg"
        ),
        (
            f"{HOST}/Exterior-Film-Scans-Tool/Night/"
            "NGH-II-800-plus1EV-Noritsu.jpg"
        ),
    ]
    payload = "".join(f'<option value="{url}"></option>' for url in urls).encode()
    inventory = inventory_comparison_metadata(payload)
    assert inventory["stock_label_counts"] == {"NHGII800": 2}
    assert inventory["filename_implied_complete_scanner_pair_count"] == 0
    assert inventory["filename_implied_incomplete_scanner_pair_count"] == 2
