from __future__ import annotations

from copy import deepcopy

from src.real_film.commons_stock_source import audit_snapshot, normalize_file_page


def _api_page(index: int, *, license_name: str = "CC BY 4.0") -> dict:
    return {
        "pageid": index,
        "title": f"File:scene-{index}.jpg",
        "categories": [{"title": "Category:Test"}],
        "imageinfo": [{
            "url": f"https://upload.example/{index}.jpg",
            "descriptionurl": f"https://commons.example/File:{index}",
            "thumburl": f"https://upload.example/thumb/{index}.jpg",
            "sha1": f"sha{index}", "size": 1000, "width": 1024, "height": 768,
            "mime": "image/jpeg", "mediatype": "BITMAP", "user": f"user{index % 5}",
            "timestamp": "2026-01-01T00:00:00Z",
            "extmetadata": {
                "Artist": {"value": f"artist{index % 5}"},
                "LicenseShortName": {"value": license_name},
                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
            },
        }],
    }


def _config() -> dict:
    categories = [
        {"film_stock_id": stock, "category": stock, "label_scope": "exact_stock_community_category"}
        for stock in ("a", "b", "c")
    ]
    categories.append({"film_stock_id": "family", "category": "family", "label_scope": "family_only_control_not_exact_stock"})
    return {
        "categories": categories,
        "license_policy": {"research_free_licenses": ["CC BY 4.0", "CC BY-SA 4.0"], "permissive_candidate_licenses": ["CC BY 4.0"]},
        "metadata_flags": {"non_scene_title_patterns": ["film strip"]},
        "gates_for_exact_stock_pilot": {
            "minimum_files": 5, "minimum_unique_uploaders": 5,
            "maximum_largest_uploader_share": 0.3, "minimum_permissive_candidate_files": 5,
            "minimum_fraction_minimum_dimension_512": 1.0,
            "maximum_non_scene_title_flag_fraction": 0.1,
        },
    }


def test_normalize_file_page_preserves_rights_and_integrity() -> None:
    row = normalize_file_page(_api_page(1))
    assert row["title"] == "File:scene-1.jpg"
    assert row["license_short_name"] == "CC BY 4.0"
    assert row["api_sha1_base36"] == "sha1"
    assert row["derivative_1600_url"].endswith("1.jpg")


def test_three_exact_stocks_pass_and_family_does_not_count() -> None:
    categories = []
    for configured in _config()["categories"]:
        categories.append({
            **configured,
            "files": [normalize_file_page(_api_page(index)) for index in range(5)],
        })
    result = audit_snapshot({"categories": categories}, _config())
    assert result["exact_stock_metadata_passes"] == 3
    assert result["conditional_pixel_pilot_allowed"] is True
    assert result["category_results"][-1]["exact_stock_pilot_eligible"] is False


def test_unknown_license_fails_closed() -> None:
    config = _config()
    categories = []
    for configured in config["categories"]:
        pages = [_api_page(index) for index in range(5)]
        if configured["film_stock_id"] == "a":
            pages[0] = _api_page(0, license_name="UNKNOWN")
        categories.append({**configured, "files": [normalize_file_page(page) for page in pages]})
    result = audit_snapshot({"categories": categories}, config)
    row = next(item for item in result["category_results"] if item["film_stock_id"] == "a")
    assert row["checks"]["every_row_free_license"] is False
    assert result["conditional_pixel_pilot_allowed"] is False


def test_non_scene_title_fraction_fails() -> None:
    config = _config()
    categories = []
    for configured in config["categories"]:
        files = [normalize_file_page(_api_page(index)) for index in range(5)]
        if configured["film_stock_id"] == "b":
            files = deepcopy(files)
            files[0]["title"] = "File:film strip example.jpg"
        categories.append({**configured, "files": files})
    result = audit_snapshot({"categories": categories}, config)
    row = next(item for item in result["category_results"] if item["film_stock_id"] == "b")
    assert row["checks"]["non_scene_title_fraction"] is False


def test_true_author_and_strict_derivative_gates_fail_closed() -> None:
    config = _config()
    config["gates_for_exact_stock_pilot"].update({
        "minimum_unique_normalized_authors": 3,
        "maximum_largest_normalized_author_share": 0.6,
        "minimum_strict_derivative_rights_rows": 4,
    })
    categories = []
    for configured in config["categories"]:
        pages = [_api_page(index) for index in range(5)]
        files = [normalize_file_page(page) for page in pages]
        if configured["film_stock_id"] == "a":
            for row in files:
                row["author_raw_html"] = "same author"
            files[0]["derivative_1600_url"] = files[0]["original_url"]
        categories.append({**configured, "files": files})
    result = audit_snapshot({"categories": categories}, config)
    row = next(item for item in result["category_results"] if item["film_stock_id"] == "a")
    assert row["checks"]["minimum_unique_normalized_authors"] is False
    assert row["checks"]["largest_normalized_author_share"] is False
    assert row["strict_derivative_rights_files"] == 4
    assert row["metadata_gate_passed"] is False
