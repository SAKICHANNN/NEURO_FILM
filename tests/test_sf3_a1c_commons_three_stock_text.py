from __future__ import annotations

from src.real_film.commons_three_stock_text import SNAPSHOT_SCHEMA, audit


def _contract() -> dict:
    return {
        "experiment_id": "test",
        "stock_queries": [
            {"film_stock_id": "fujifilm_velvia_50", "exact_search_phrase": "Velvia 50"},
            {"film_stock_id": "kodak_portra_400", "exact_search_phrase": "Kodak Portra 400"},
            {"film_stock_id": "kodak_ektar_100", "exact_search_phrase": "Kodak Ektar 100"},
        ],
        "query_limits": {"maximum_results_per_stock": 8},
        "license_policy": {"permissive_license_short_names": ["CC BY 4.0"]},
        "metadata_exclusions": ["filmstrip"],
        "metadata_gates": {
            "minimum_search_rows_per_stock": 2,
            "minimum_eligible_rows_per_stock": 2,
            "minimum_unique_authors_per_stock": 2,
            "maximum_largest_author_share": 0.5,
            "minimum_short_dimension": 512,
            "maximum_cross_stock_ambiguous_rows": 0,
        },
        "decision_if_pass": "pass",
        "decision_if_fail": "fail",
        "claim_ceiling": "metadata only",
    }


def _row(page_id: int, author: str) -> dict:
    return {
        "page_id": page_id,
        "title": f"File:scene-{page_id}.jpg",
        "description_raw_html": "ordinary scene",
        "categories": [],
        "license_short_name": "CC BY 4.0",
        "author_raw_html": author,
        "file_page_url": f"https://commons.wikimedia.org/wiki/File:{page_id}",
        "original_url": f"https://upload.wikimedia.org/{page_id}.jpg",
        "derivative_1600_url": f"https://upload.wikimedia.org/thumb/{page_id}.jpg",
        "width": 1600,
        "height": 1200,
    }


def _snapshot() -> dict:
    stocks = ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]
    return {
        "schema": SNAPSHOT_SCHEMA,
        "network_requests": 3,
        "image_payloads_downloaded_or_decoded": False,
        "stock_results": [
            {
                "film_stock_id": stock,
                "exact_search_phrase": stock,
                "reported_total_hits": 2,
                "rows": [_row(index * 10, "a"), _row(index * 10 + 1, "b")],
            }
            for index, stock in enumerate(stocks)
        ],
    }


def test_balanced_three_stock_metadata_passes_without_pixels() -> None:
    report = audit(_snapshot(), _contract())
    assert report["automatic_pass"] is True
    assert report["decision"] == "pass"
    assert report["pixel_decodes"] == report["operator_fits"] == 0


def test_cross_stock_page_is_excluded_and_fails_ambiguity_gate() -> None:
    snapshot = _snapshot()
    snapshot["stock_results"][1]["rows"][0]["page_id"] = 0
    report = audit(snapshot, _contract())
    assert report["automatic_pass"] is False
    assert report["ambiguity_gate_passed"] is False
    assert report["cross_stock_ambiguous_row_count"] == 1


def test_non_scene_and_missing_author_rows_are_excluded() -> None:
    snapshot = _snapshot()
    snapshot["stock_results"][0]["rows"][0]["title"] = "File:filmstrip.jpg"
    snapshot["stock_results"][0]["rows"][1]["author_raw_html"] = ""
    report = audit(snapshot, _contract())
    first = report["stock_results"][0]
    assert first["eligible_rows"] == 0
    assert first["metadata_gate_passed"] is False
