from __future__ import annotations

import json
from email.message import Message
from pathlib import Path

from src.real_film.openverse_transport_recovery import (
    fetch_snapshot_urllib,
    load_effective_config,
    scientific_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


class _Response:
    status = 200

    def __init__(self, url: str, payload: dict[str, object]) -> None:
        self._url = url
        self._body = json.dumps(payload).encode()
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return self._url

    def read(self, amount: int) -> bytes:
        return self._body[:amount]

    def close(self) -> None:
        return None


class _Opener:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def open(self, request, timeout: float):
        del timeout
        self.urls.append(request.full_url)
        query = dict(
            __import__("urllib.parse").parse.parse_qsl(
                request.full_url.split("?", 1)[1]
            )
        )
        stock = query["q"].casefold()
        row_id = (
            "velvia"
            if "velvia" in stock
            else "portra"
            if "portra" in stock
            else "ektar"
        )
        return _Response(
            request.full_url,
            {
                "page_count": 1,
                "results": [
                    {
                        "id": row_id,
                        "title": stock,
                        "foreign_landing_url": f"https://example.test/{row_id}",
                        "url": f"https://forbidden.test/{row_id}.jpg",
                        "creator": "creator",
                        "creator_url": "https://example.test/creator",
                        "license": "by",
                        "license_version": "4.0",
                        "provider": "provider",
                        "source": "source",
                        "tags": [],
                        "fields_matched": ["title"],
                        "height": 10,
                        "width": 10,
                    }
                ],
            },
        )


def _contract() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs" / "sf3_a3l_openverse_transport_recovery_v1.json").read_text()
    )


def test_bound_contract_loads_and_preserves_frozen_science() -> None:
    base = load_effective_config(ROOT, _contract())
    assert base["query"]["maximum_total_requests"] == 15
    assert base["query"]["image_payload_requests_allowed"] is False
    assert base["connectivity_gate"]["minimum_connected_stocks"] == 3


def test_urllib_snapshot_omits_pixel_urls_and_is_order_canonical() -> None:
    contract = _contract()
    base = load_effective_config(ROOT, contract)
    order = [str(stock["film_stock_id"]) for stock in base["stocks"]]
    left = fetch_snapshot_urllib(
        base, contract, stock_order=order, opener=_Opener(), sleep_fn=lambda _: None
    )
    right = fetch_snapshot_urllib(
        base,
        contract,
        stock_order=list(reversed(order)),
        opener=_Opener(),
        sleep_fn=lambda _: None,
    )
    assert left["request_count"] == 3
    assert left["source_error"] is False
    assert "forbidden.test" not in json.dumps(left)
    assert scientific_snapshot(left) == scientific_snapshot(right)


def test_transport_contract_forbids_pixel_and_postscore_rescue() -> None:
    policy = _contract()["formal_policy"]
    assert (
        policy["image_thumbnail_detail_landing_and_related_requests_allowed"] is False
    )
    assert policy["operator_fitting_training_and_mode_study_allowed"] is False
    assert (
        policy["post_result_query_alias_page_gate_or_transport_rescue_allowed"] is False
    )
