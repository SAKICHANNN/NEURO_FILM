"""Fetch and freeze the SF0.4 Wikimedia Commons category metadata snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.commons_stock_source import (  # noqa: E402
    CommonsStockSourceError,
    normalize_file_page,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _request(endpoint: str, parameters: dict[str, str], user_agent: str) -> dict:
    url = endpoint + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for delay in (0, 2, 5, 10):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or delay == 10:
                raise
    raise CommonsStockSourceError("unreachable API retry state")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "real_film_commons_stock_source_audit_v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = _load(args.config)
    output = args.output or ROOT / config["snapshot_output"]
    categories: list[dict] = []
    for configured in config["categories"]:
        category_title = "Category:" + configured["category"]
        category_response = _request(config["api_endpoint"], {
            "action": "query", "format": "json", "formatversion": "2",
            "titles": category_title, "prop": "categoryinfo|revisions",
            "rvprop": "ids|timestamp",
        }, config["user_agent"])
        category_pages = category_response.get("query", {}).get("pages", [])
        if len(category_pages) != 1 or category_pages[0].get("missing"):
            raise CommonsStockSourceError(f"missing Commons category: {category_title}")
        category_page = category_pages[0]
        member_response = _request(config["api_endpoint"], {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "categorymembers", "gcmtitle": category_title,
            "gcmtype": "file", "gcmlimit": "500", "gcmsort": "sortkey",
            "prop": "imageinfo|categories", "cllimit": "max",
            "iiprop": "url|size|sha1|user|timestamp|mime|mediatype|extmetadata",
            "iiurlwidth": str(config["api_query"]["thumbnail_url_width_for_future_pilot"]),
        }, config["user_agent"])
        if "continue" in member_response:
            raise CommonsStockSourceError(f"category exceeds one frozen page: {category_title}")
        pages = member_response.get("query", {}).get("pages", [])
        normalized = sorted((normalize_file_page(page) for page in pages), key=lambda row: row["title"])
        categories.append({
            "film_stock_id": configured["film_stock_id"],
            "category": configured["category"],
            "label_scope": configured["label_scope"],
            "category_page_id": int(category_page["pageid"]),
            "category_revision": category_page["revisions"][0],
            "category_info": category_page["categoryinfo"],
            "files": normalized,
        })
    snapshot = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "api_endpoint": config["api_endpoint"],
        "requests_per_category": 2,
        "image_payloads_downloaded_or_decoded": False,
        "categories": categories,
    }
    encoded = (json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    print(json.dumps({
        "snapshot": str(output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "categories": len(categories),
        "files": sum(len(row["files"]) for row in categories),
        "image_payloads_downloaded_or_decoded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
