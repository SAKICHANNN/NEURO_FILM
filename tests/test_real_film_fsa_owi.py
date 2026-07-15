from __future__ import annotations

from src.real_film.fsa_owi import (
    build_canonical_subset,
    evaluate_metadata_gate,
    extract_loc_id,
    merge_api_pages,
    parse_page,
)


def _page(page_id: int = 7) -> dict:
    return {
        "pageid": page_id,
        "title": "File:Example.jpg",
        "imageinfo": [
            {
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Example.jpg",
                "url": "https://upload.wikimedia.org/example.jpg",
                "thumburl": "https://upload.wikimedia.org/example-1280.jpg",
                "width": 2000,
                "height": 1000,
                "thumbwidth": 1280,
                "thumbheight": 640,
                "size": 100000,
                "sha1": "abc",
                "mime": "image/jpeg",
                "extmetadata": {
                    "Credit": {"value": '<a href="https://hdl.loc.gov/loc.pnp/fsac.1A34354">source</a>'},
                    "Artist": {"value": "<b>Jane Doe</b>"},
                    "DateTimeOriginal": {"value": "June 1942"},
                    "ObjectName": {"value": "A &amp; B"},
                    "LicenseShortName": {"value": "Public domain"},
                    "UsageTerms": {"value": "Public domain"},
                    "AttributionRequired": {"value": "false"},
                    "Categories": {
                        "value": "PD US FSA/OWI|Images from the Library of Congress"
                    },
                },
            }
        ],
    }


def _config() -> dict:
    return {
        "phase_a_metadata": {
            "minimum_unique_records": 1,
            "minimum_unique_photographers": 1,
            "minimum_loc_identifier_coverage": 1.0,
            "minimum_public_domain_coverage": 1.0,
            "minimum_loc_source_coverage": 1.0,
        },
        "license_gate": {
            "required_license_short_name": "Public domain",
            "required_commons_category": "PD US FSA/OWI",
            "required_source_category": "Images from the Library of Congress",
        },
        "canonical_recovery": {
            "minimum_unique_loc_records": 1,
            "minimum_creator_group_coverage": 0.0,
            "minimum_creators_with_eight_records": 0,
            "creator_category_prefix": "Farm Security Administration photographs by ",
            "forbidden_title_tokens": ["cropped"],
        },
    }


def test_extract_and_parse_loc_metadata() -> None:
    assert extract_loc_id("https://hdl.loc.gov/loc.pnp/fsac.1A34354") == "fsac.1a34354"
    assert extract_loc_id(
        "https://cdn.loc.gov/service/pnp/fsac/1a34000/1a34200/1a34297v.jpg"
    ) == "fsac.1a34297"
    record = parse_page(_page())
    assert record["loc_fsac_id"] == "fsac.1a34354"
    assert record["creator"] == "Jane Doe"
    assert record["description"] == "A & B"


def test_merge_api_pages_retains_split_imageinfo() -> None:
    merged: dict[int, dict] = {}
    merge_api_pages(merged, [{"pageid": 7, "title": "File:Example.jpg"}])
    merge_api_pages(merged, [_page()])
    assert merged[7]["imageinfo"][0]["sha1"] == "abc"


def test_metadata_gate_passes_complete_public_domain_record() -> None:
    result = evaluate_metadata_gate([parse_page(_page())], _config())
    assert result["decision"] == "pilot_allowed"
    assert result["unique_loc_identifiers"] == 1


def test_metadata_gate_rejects_duplicate_loc_ids() -> None:
    left = parse_page(_page(7))
    right = parse_page(_page(8))
    right["commons_title"] = "File:Other.jpg"
    result = evaluate_metadata_gate([left, right], _config())
    assert result["decision"] == "metadata_ineligible"
    assert result["checks"]["unique_loc_identifiers"] is False


def test_canonical_subset_prefers_loc_service_file_and_removes_duplicates() -> None:
    direct = parse_page(_page(7))
    direct["commons_title"] = "File:Example 1a34354v.jpg"
    direct["categories"].append("Farm Security Administration photographs by Jane Doe")
    crop = dict(direct)
    crop["commons_page_id"] = 8
    crop["commons_title"] = "File:Example cropped.jpg"
    selected, audit = build_canonical_subset([crop, direct], _config())
    assert audit["decision"] == "canonical_pilot_allowed"
    assert audit["duplicate_representations_removed"] == 1
    assert selected[0]["commons_page_id"] == 7
    assert selected[0]["creator_group"] == "Jane Doe"
