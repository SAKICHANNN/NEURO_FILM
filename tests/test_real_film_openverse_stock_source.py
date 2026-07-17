from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from src.real_film.openverse_stock_source import audit_snapshot, fetch_snapshot


def _config() -> dict:
    config = json.loads(Path("configs/real_film_openverse_shared_creator_v1.json").read_text(encoding="utf-8"))
    config["stocks"] = config["stocks"][:3]
    config["query"]["maximum_pages_per_stock"] = 1
    config["query"]["maximum_total_requests"] = 3
    config["request_limits"]["request_interval_seconds"] = 0
    config["connectivity_gate"].update(
        {
            "minimum_strict_rows_per_eligible_stock": 3,
            "minimum_unique_creators_per_eligible_stock": 3,
            "maximum_largest_creator_share_per_eligible_stock": 0.4,
            "minimum_connected_stocks": 3,
            "minimum_shared_creators_total": 2,
            "minimum_shared_creators_per_graph_edge": 1,
        }
    )
    return config


def _row(stock: dict, suffix: str, creator: str) -> dict:
    alias = stock["title_aliases"][0]
    return {
        "id": f"id-{stock['film_stock_id']}-{suffix}",
        "title": f"Scene photographed on {alias}",
        "indexed_on": "2026-01-01T00:00:00Z",
        "foreign_landing_url": f"https://example.test/{stock['film_stock_id']}/{suffix}",
        "url": f"https://images.example.test/{suffix}.jpg",
        "thumbnail": f"https://api.openverse.org/v1/images/{suffix}/thumb/",
        "creator": creator,
        "creator_url": f"https://example.test/creator/{creator}",
        "license": "by",
        "license_version": "2.0",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "provider": "flickr",
        "source": "flickr",
        "category": "photograph",
        "tags": [{"name": stock["tag_aliases"][0]}],
        "fields_matched": ["title"],
        "mature": False,
        "height": 800,
        "width": 1200,
    }


def _rows(config: dict) -> dict[str, list[dict]]:
    first, second, third = config["stocks"]
    return {
        first["query"]: [
            _row(first, "ab", "shared-ab"),
            _row(first, "ac", "shared-ac"),
            _row(first, "a", "unique-a"),
        ],
        second["query"]: [
            _row(second, "ab", "shared-ab"),
            _row(second, "b1", "unique-b1"),
            _row(second, "b2", "unique-b2"),
        ],
        third["query"]: [
            _row(third, "ac", "shared-ac"),
            _row(third, "c1", "unique-c1"),
            _row(third, "c2", "unique-c2"),
        ],
    }


class _Response:
    def __init__(self, url: str, payload: dict) -> None:
        self.url = url
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self.body = json.dumps(payload).encode()
        self.closed = False

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index : index + chunk_size]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, rows: dict[str, list[dict]]) -> None:
        self.headers: dict[str, str] = {}
        self.rows = rows
        self.requests: list[tuple[str, dict]] = []
        self.responses: list[_Response] = []

    def get(self, url: str, params: dict, **kwargs) -> _Response:
        del kwargs
        self.requests.append((url, params))
        response = _Response(url, {"result_count": len(self.rows[params["q"]]), "page_count": 1, "results": self.rows[params["q"]]})
        self.responses.append(response)
        return response


def test_tracked_contract_is_metadata_only_and_bounded() -> None:
    config = json.loads(Path("configs/real_film_openverse_shared_creator_v1.json").read_text(encoding="utf-8"))
    assert config["query"]["maximum_total_requests"] == 48
    assert config["query"]["raw_response_retained"] is False
    assert config["query"]["thumbnail_requests_allowed"] is False
    assert config["query"]["landing_page_requests_allowed"] is False
    assert config["image_payload_download_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
    assert config["latent_mode_study_allowed"] is False


def test_fetch_omits_pixel_urls_and_offline_graph_passes() -> None:
    config = _config()
    session = _Session(_rows(config))
    snapshot = fetch_snapshot(config, session=session, sleep_fn=lambda _: None)
    serialized = json.dumps(snapshot)
    assert snapshot["request_count"] == 3
    assert '"url"' not in serialized
    assert '"thumbnail"' not in serialized
    assert "images.example.test" not in serialized
    assert all(urlparse(url).netloc == "api.openverse.org" for url, _ in session.requests)
    assert all(response.closed for response in session.responses)
    report, decision = audit_snapshot(snapshot, config)
    assert decision["decision"] == "open_bounded_live_rights_label_preflight"
    assert decision["selected_component"]["stocks"] == sorted(stock["film_stock_id"] for stock in config["stocks"])
    assert report["raw_response_retained"] is False
    assert audit_snapshot(snapshot, config) == (report, decision)


def test_identity_overlap_fails_closed() -> None:
    config = _config()
    rows = _rows(config)
    rows[config["stocks"][1]["query"]][0]["id"] = rows[config["stocks"][0]["query"]][0]["id"]
    snapshot = fetch_snapshot(config, session=_Session(rows), sleep_fn=lambda _: None)
    _, decision = audit_snapshot(snapshot, config)
    assert decision["decision"] == "cross_stock_identity_overlap"


def test_insufficient_shared_creator_graph_closes() -> None:
    config = _config()
    rows = _rows(config)
    for query_rows in rows.values():
        for index, row in enumerate(query_rows):
            row["creator"] = f"{row['id']}-{index}"
            row["creator_url"] = f"https://example.test/creator/{row['creator']}"
    snapshot = fetch_snapshot(config, session=_Session(rows), sleep_fn=lambda _: None)
    _, decision = audit_snapshot(snapshot, config)
    assert decision["decision"] == "insufficient_shared_creator_connectivity"


def test_duplicate_within_stock_is_a_contract_failure() -> None:
    config = _config()
    rows = _rows(config)
    rows[config["stocks"][0]["query"]][1]["id"] = rows[config["stocks"][0]["query"]][0]["id"]
    snapshot = fetch_snapshot(config, session=_Session(rows), sleep_fn=lambda _: None)
    _, decision = audit_snapshot(snapshot, config)
    assert decision["decision"] == "query_contract_mismatch"
