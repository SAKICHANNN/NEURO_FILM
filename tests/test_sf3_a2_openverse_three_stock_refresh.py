from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.real_film.openverse_stock_source import OpenverseStockSourceError
from src.real_film.openverse_three_stock_refresh import audit_fresh_snapshot


def _config() -> dict:
    value = json.loads(
        Path("configs/sf3_a2_openverse_three_stock_refresh_v1.json").read_text(
            encoding="utf-8"
        )
    )
    value["connectivity_gate"].update(
        minimum_strict_rows_per_eligible_stock=1,
        minimum_unique_creators_per_eligible_stock=1,
        maximum_largest_creator_share_per_eligible_stock=1.0,
        minimum_shared_creators_total=1,
        minimum_shared_creators_per_graph_edge=1,
    )
    return value


def _row(stock: dict, row_id: str, creator: str = "shared") -> dict:
    return {
        "id": row_id,
        "film_stock_id": stock["film_stock_id"],
        "title": f"Photo on {stock['title_aliases'][0]}",
        "foreign_landing_url": f"https://example.test/{row_id}",
        "creator": creator,
        "creator_url": f"https://example.test/creator/{creator}",
        "license": "by",
        "license_version": "2.0",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "provider": "flickr",
        "source": "flickr",
        "fields_matched": ["title"],
        "title_alias_matches": [stock["title_aliases"][0]],
        "tag_alias_matches": [],
        "mature": False,
        "width": 1200,
        "height": 800,
    }


def _snapshot(config: dict) -> dict:
    return {
        "request_count": 3,
        "source_error": False,
        "categories": [
            {
                "film_stock_id": stock["film_stock_id"],
                "requests": [{"page": 1, "bounded_json_valid": True}],
                "rows": [_row(stock, f"fresh-{index}")],
            }
            for index, stock in enumerate(config["stocks"])
        ],
    }


def _bind_prior(tmp_path: Path, config: dict, rows: list[dict] | None = None) -> None:
    inputs = {
        "openverse_snapshot": {"categories": [{"rows": rows or []}]},
        "yfcc_triangle_report": {"ok": True},
        "commons_connectivity_report": {"ok": True},
    }
    for key, value in inputs.items():
        path = tmp_path / f"{key}.json"
        payload = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(payload)
        config["prior_inputs"][key]["path"] = path.name
        config["prior_inputs"][key]["sha256"] = hashlib.sha256(payload).hexdigest()


def test_refresh_excludes_prior_and_deduplicates_identical_pages(
    tmp_path: Path,
) -> None:
    config = _config()
    prior = _row(config["stocks"][0], "prior")
    _bind_prior(tmp_path, config, [prior])
    snapshot = _snapshot(config)
    snapshot["categories"][0]["rows"] = [
        prior,
        snapshot["categories"][0]["rows"][0],
        deepcopy(snapshot["categories"][0]["rows"][0]),
    ]
    report, decision = audit_fresh_snapshot(snapshot, config, root=tmp_path)
    assert decision["decision"] == "open_bounded_live_rights_label_preflight"
    assert report["freshness"]["prior_rows_excluded"] == 1
    assert report["freshness"]["repeated_page_rows_deduplicated"] == 1
    assert (
        report["freshness"]["fresh_rows_by_stock"][config["stocks"][0]["film_stock_id"]]
        == 1
    )


def test_conflicting_duplicate_fails_closed(tmp_path: Path) -> None:
    config = _config()
    _bind_prior(tmp_path, config)
    snapshot = _snapshot(config)
    conflict = deepcopy(snapshot["categories"][0]["rows"][0])
    conflict["creator_url"] = "https://example.test/creator/other"
    snapshot["categories"][0]["rows"].append(conflict)
    _, decision = audit_fresh_snapshot(snapshot, config, root=tmp_path)
    assert decision["decision"] == "query_contract_mismatch"


def test_bound_prior_hash_drift_rejects(tmp_path: Path) -> None:
    config = _config()
    _bind_prior(tmp_path, config)
    config["prior_inputs"]["openverse_snapshot"]["sha256"] = "0" * 64
    with pytest.raises(OpenverseStockSourceError, match="bound input hash mismatch"):
        audit_fresh_snapshot(_snapshot(config), config, root=tmp_path)


def test_tracked_contract_is_metadata_only_and_bounded() -> None:
    config = json.loads(
        Path("configs/sf3_a2_openverse_three_stock_refresh_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert [stock["film_stock_id"] for stock in config["stocks"]] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert config["query"]["maximum_total_requests"] == 15
    assert config["query"]["raw_response_retained"] is False
    assert config["image_payload_download_allowed"] is False
    assert config["photo_page_access_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
