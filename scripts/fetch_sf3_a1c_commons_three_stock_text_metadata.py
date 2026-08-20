"""Fetch and audit a bounded Commons three-stock exact-text metadata snapshot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_source import normalize_file_page
from src.real_film.commons_three_stock_text import SNAPSHOT_SCHEMA, audit, load_contract


def _request(session: requests.Session, endpoint: str, params: dict[str, str], contract: dict) -> dict:
    last: Exception | None = None
    for delay in contract["query_limits"]["retry_delays_seconds"]:
        if delay:
            time.sleep(float(delay))
        try:
            response = session.get(
                endpoint,
                params=params,
                timeout=float(contract["query_limits"]["timeout_seconds"]),
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last = exc
    raise RuntimeError(f"Commons API request failed: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/sf3_a1c_commons_three_stock_text_metadata_v1.json",
    )
    args = parser.parse_args()
    _, contract = load_contract(args.contract)
    session = requests.Session()
    session.headers["User-Agent"] = str(contract["user_agent"])
    blocks = []
    request_count = 0
    limits = contract["query_limits"]
    for query in contract["stock_queries"]:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": f'"{query["exact_search_phrase"]}"',
            "gsrwhat": "text",
            "gsrnamespace": "6",
            "gsrlimit": str(limits["results_per_request"]),
            "prop": "imageinfo|categories",
            "iiprop": "url|size|sha1|user|timestamp|mime|mediatype|extmetadata",
            "iilimit": "1",
            "iiurlwidth": str(limits["thumbnail_width"]),
            "cllimit": "max",
        }
        rows: dict[int, dict] = {}
        total_hits: int | None = None
        requests_for_stock = 0
        while len(rows) < int(limits["maximum_results_per_stock"]):
            requests_for_stock += 1
            request_count += 1
            if requests_for_stock > int(limits["maximum_requests_per_stock"]):
                raise RuntimeError("per-stock Commons request ceiling exceeded")
            payload = _request(session, str(contract["api_endpoint"]), params, contract)
            total_hits = int(payload.get("query", {}).get("searchinfo", {}).get("totalhits", total_hits or 0))
            for page in payload.get("query", {}).get("pages", []):
                if len(rows) >= int(limits["maximum_results_per_stock"]):
                    break
                if len(page.get("imageinfo", [])) == 1:
                    rows[int(page["pageid"])] = normalize_file_page(page)
            continuation = payload.get("continue")
            if not continuation:
                break
            allowed = {"continue", "gsroffset"}
            if set(continuation) - allowed:
                raise RuntimeError("unexpected Commons search continuation")
            params.update({key: str(value) for key, value in continuation.items()})
        blocks.append(
            {
                "film_stock_id": query["film_stock_id"],
                "exact_search_phrase": query["exact_search_phrase"],
                "reported_total_hits": int(total_hits or 0),
                "api_requests": requests_for_stock,
                "rows": list(rows.values()),
            }
        )
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "network_requests": request_count,
        "image_payloads_downloaded_or_decoded": False,
        "stock_results": blocks,
    }
    report = audit(snapshot, contract)
    for key, value in (("snapshot_output", snapshot), ("report_output", report)):
        path = ROOT / str(contract[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "automatic_pass": report["automatic_pass"],
                "network_requests": report["network_requests"],
                "stocks": [
                    {
                        "film_stock_id": row["film_stock_id"],
                        "retained_search_rows": row["retained_search_rows"],
                        "eligible_rows": row["eligible_rows"],
                        "unique_authors": row["unique_authors"],
                    }
                    for row in report["stock_results"]
                ],
            },
            indent=2,
        )
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
