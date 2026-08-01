from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.flickr_single_author_pair_source import (
    FlickrAlbumAuditError,
    SCHEMA,
    audit_payload,
    classify_title,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/u5_r2bo0_flickr_single_author_pair_source_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _row(photo_id: int, title: str, *, license_id: int = 1) -> dict:
    return {
        "id": str(photo_id),
        "title": title,
        "license": str(license_id),
        "ispublic": 1,
        "media": "photo",
        "media_status": "ready",
        "url_l": f"https://example.invalid/{photo_id}_b.jpg",
        "width_l": 1024,
        "height_l": 768,
    }


def test_title_classifier_keeps_only_exact_series_rows() -> None:
    families = _config()["pair_families"]
    digital = classify_title(
        "Digital vs. Film Shootout: Scene 12 Digital - Nikon D70", families
    )
    film = classify_title(
        "Film (Nikon) vs. Digital (Leica) Scene 7 (Film) - Nikon F90X", families
    )
    composite = classify_title(
        "Digital (left) Olympus Camedia C-70 vs. Analog (right) Yashica Auto Focus Motor - 7 (of 23)",
        families,
    )
    assert (digital.family_id, digital.scene_id, digital.role) == (
        "nikon_d70_to_ricoh_rw1",
        12,
        "digital",
    )
    assert (film.family_id, film.scene_id, film.role) == (
        "lumix_lc40_to_nikon_f90x_fuji200",
        7,
        "film",
    )
    assert composite is None


def test_audit_forms_complete_pairs_and_enforces_rights() -> None:
    config = _config()
    config["source"]["expected_public_photo_count"] = 6
    config["feasibility_gates"].update(
        {
            "minimum_complete_rights_eligible_pairs": 2,
            "minimum_independent_capture_families": 2,
            "maximum_largest_family_pair_share": 0.5,
        }
    )
    rows = [
        _row(1, "Digital vs. Film Shootout: Scene 1 Digital - Nikon D70"),
        _row(2, "Digital vs. Film Shootout: Scene 1 Film - Ricoh RW-1"),
        _row(3, "Film (Nikon) vs. Digital (Leica) Scene 2 (Digital) - Lumix"),
        _row(4, "Film (Nikon) vs. Digital (Leica) Scene 2 (Film) - Nikon"),
        _row(5, "unmatched album row"),
        _row(6, "Digital vs. Film Shootout: Scene 9 Film - Ricoh RW-1", license_id=0),
    ]
    payload = {
        "stat": "ok",
        "photoset": {
            "id": config["source"]["album_id"],
            "owner": config["source"]["owner_nsid"],
            "total": "6",
            "photo": rows,
        },
    }
    report = audit_payload(payload, config)
    assert report["automatic_pass"] is True
    assert report["metrics"]["title_matched_complete_pairs"] == 2
    assert report["metrics"]["eligible_complete_pairs"] == 2
    assert report["metrics"]["ambiguous_or_incomplete_series_rows"] == [
        {
            "family_id": "nikon_d70_to_ricoh_rw1",
            "scene_id": 9,
            "digital_count": 0,
            "film_count": 1,
        }
    ]
    assert report["operator_fitting_allowed"] is False
    assert report["training_allowed"] is False


def test_audit_rejects_identity_count_and_duplicate_drift() -> None:
    config = _config()
    payload = {
        "stat": "ok",
        "photoset": {
            "id": "wrong",
            "owner": config["source"]["owner_nsid"],
            "total": "0",
            "photo": [],
        },
    }
    with pytest.raises(FlickrAlbumAuditError, match="identity"):
        audit_payload(payload, config)


def test_frozen_contract_preserves_metadata_only_claim_boundary() -> None:
    config = _config()
    assert config["schema"] == SCHEMA
    assert config["source"]["image_payloads_allowed"] is False
    assert config["source"]["allowed_license_ids"] == [1]
    assert config["feasibility_gates"]["minimum_complete_rights_eligible_pairs"] == 45
    assert any("commercial" in item for item in config["forbidden"])
    assert "weak-pair" in config["claim_ceiling"]
