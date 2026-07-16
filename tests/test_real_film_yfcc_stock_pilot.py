from __future__ import annotations

from src.real_film.yfcc_stock_pilot import (
    balanced_candidate_order,
    candidate_image_urls,
    eligible_rows,
)


def _config() -> dict:
    return {
        "target_stock_id": "fujifilm_velvia_50",
        "process_contamination_exclusions": ["cross[ -]?process|xpro", "\\bhdr\\b"],
        "selection": {
            "maximum_files_per_uid": 2,
            "preflight_expected": {"eligible_rows": 4, "author_uids": 3, "largest_author_share": 0.5},
        },
    }


def test_prospective_filter_and_uid_round_robin_are_stable() -> None:
    rows = [
        {"stock_id": "fujifilm_velvia_50", "photoid": 3, "uid": "a", "title": "Velvia 50", "description": "", "usertags": ""},
        {"stock_id": "fujifilm_velvia_50", "photoid": 1, "uid": "a", "title": "Velvia 50", "description": "", "usertags": ""},
        {"stock_id": "fujifilm_velvia_50", "photoid": 2, "uid": "b", "title": "Velvia 50", "description": "", "usertags": ""},
        {"stock_id": "fujifilm_velvia_50", "photoid": 4, "uid": "c", "title": "Velvia 50", "description": "", "usertags": ""},
        {"stock_id": "fujifilm_velvia_50", "photoid": 5, "uid": "d", "title": "Velvia xpro", "description": "", "usertags": ""},
    ]
    report = {"parquet_sha256": {"x": "y"}, "matches": rows}
    ordered = balanced_candidate_order(eligible_rows(report, _config()), _config())
    assert [(row["uid"], row["photoid"]) for row in ordered] == [("a", 1), ("b", 2), ("c", 4), ("a", 3)]


def test_large_flickr_url_precedes_frozen_fallback() -> None:
    urls = candidate_image_urls("http://farm1.staticflickr.com/2/3_abcd.jpg")
    assert urls == [
        "https://farm1.staticflickr.com/2/3_abcd_b.jpg",
        "https://farm1.staticflickr.com/2/3_abcd.jpg",
    ]
