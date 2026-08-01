from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.flickr_bw_composite_pair_source import FlickrBwCompositeSourceError, audit_payload


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(
        (ROOT / "configs/u5_r2bo9_flickr_bw_composite_pair_source_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _payload() -> dict:
    config = _config()
    rows = []
    for item in config["eligible_composites"]:
        rows.append(
            {
                "id": item["photo_id"],
                "title": f"#{item['scene_id']} 'Film vs. Digital B&W'",
                "license": "4",
                "ispublic": 1,
                "media": "photo",
                "description": {
                    "_content": "Above: Minolta XG-2. Below: Sony SLT-77; making the same shots"
                },
                "url_l": f"https://example.invalid/{item['photo_id']}.jpg",
                "width_l": 765,
                "height_l": 1024,
            }
        )
    for item in config["explicit_exclusions"]:
        rows.append(
            {
                "id": item["photo_id"],
                "title": "excluded",
                "license": "4",
                "ispublic": 1,
                "media": "photo",
                "description": {"_content": "single view"},
            }
        )
    return {"stat": "ok", "photos": {"total": "9", "photo": rows}}


def test_audit_accepts_exact_seven_composites() -> None:
    report = audit_payload(_payload(), _config())
    assert report["automatic_pass"] is True
    assert report["metrics"]["complete_composites"] == 7
    assert report["metrics"]["excluded_single_view_rows"] == 2
    assert report["operator_fitting_allowed"] is False


def test_audit_rejects_role_or_dimension_drift() -> None:
    payload = _payload()
    payload["photos"]["photo"][0]["description"]["_content"] = "ambiguous comparison"
    report = audit_payload(payload, _config())
    assert report["automatic_pass"] is False
    assert report["checks"]["explicit_role_text"] is False

    payload = _payload()
    payload["photos"]["photo"][0]["width_l"] = 764
    report = audit_payload(payload, _config())
    assert report["automatic_pass"] is False
    assert report["checks"]["frozen_dimensions"] is False


def test_audit_fails_closed_on_search_count_drift() -> None:
    payload = _payload()
    payload["photos"]["total"] = "10"
    with pytest.raises(FlickrBwCompositeSourceError, match="count drift"):
        audit_payload(payload, _config())
