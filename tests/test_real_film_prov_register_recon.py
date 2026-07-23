from __future__ import annotations

from src.real_film.prov_register_recon import (
    canonical_json,
    summarize_catalogues,
)


def _config() -> dict:
    return {
        "expected_discovery_bounds": {
            "negative_register_minimum_items": 1,
            "negative_register_maximum_items": 100,
            "digitised_collection_minimum_items": 1000,
        }
    }


def test_normalization_removes_qtime_only() -> None:
    first = {"nested": {"responseHeader": {"status": 0, "QTime": 1}}, "response": {"numFound": 0, "docs": []}}
    second = {"nested": {"responseHeader": {"status": 0, "QTime": 99}}, "response": {"numFound": 0, "docs": []}}
    assert canonical_json(first) == canonical_json(second)


def test_physical_register_closes_before_pixels() -> None:
    register = {
        "response": {
            "numFound": 1,
            "docs": [
                {
                    "identifier.PROV_ACM.id": "VPRS 17690/P0001/1",
                    "title": "1913-1945",
                    "consignment_id": "P0001",
                    "format": "Physical",
                    "description.aggregate": "Neg. no. 13-1 to 45-6369",
                    "rights_status": ["Open"],
                }
            ],
        }
    }
    collection = {
        "response": {"numFound": 6832, "docs": []},
        "facet_counts": {"facet_fields": {"format": ["Digital", 6716, "Physical", 116]}},
    }
    result = summarize_catalogues(register, collection, _config())
    assert result["decision"] == "closed_register_catalogue_is_physical_only"
    assert not result["register_contents_machine_accessible"]
    assert result["digitised_negative_collection"]["format_counts"]["Digital"] == 6716


def test_digital_register_opens_only_separate_join_audit() -> None:
    register = {
        "response": {
            "numFound": 1,
            "docs": [
                {
                    "identifier.PROV_ACM.id": "VPRS 17690/P0001/1",
                    "title": "1913-1945",
                    "consignment_id": "P0001",
                    "format": "Digital",
                    "iiif-manifest": "https://example.invalid/manifest",
                }
            ],
        }
    }
    collection = {
        "response": {"numFound": 6832, "docs": []},
        "facet_counts": {"facet_fields": {"format": ["Digital", 6716, "Physical", 116]}},
    }
    result = summarize_catalogues(register, collection, _config())
    assert result["decision"] == "open_separate_bounded_register_content_join_audit"
