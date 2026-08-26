"""Acquire then audit the bounded Commons Ektar companion-source route."""

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

from src.real_film.commons_ektar_companion import (
    CommonsEktarCompanionError,
    audit_snapshot,
    canonical_json,
    flickr_identity,
    load_contract,
    normalize_search_page,
    select_metadata_candidates,
    sha256_bytes,
)


def _request(session: requests.Session, endpoint: str, params: dict[str, str], contract: dict) -> dict:
    last: Exception | None = None
    minimum_interval = float(contract["query"]["minimum_request_interval_seconds"])
    for delay in contract["query"]["retry_delays_seconds"]:
        time.sleep(max(minimum_interval, float(delay)))
        try:
            response = session.get(endpoint, params=params, timeout=float(contract["query"]["timeout_seconds"]))
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdecimal():
                    time.sleep(float(retry_after))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
    raise CommonsEktarCompanionError(f"Commons API request failed: {last}")


def _acquire(contract_path: Path, contract: dict) -> dict:
    manifest = json.loads((ROOT / str(contract["input_selection_manifest"])).read_text(encoding="utf-8"))
    sources = [row for row in manifest["rows"] if row["film_stock_id"] == contract["film_stock_id"]]
    if len(sources) != int(contract["expected_target_rows"]):
        raise CommonsEktarCompanionError("unexpected Ektar target row count")
    identities = sorted({value for row in sources if (value := flickr_identity(row))})
    session = requests.Session()
    session.headers["User-Agent"] = str(contract["user_agent"])
    pages_by_identity: dict[str, list[dict]] = {}
    requests_used = 0
    for identity in identities:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": f'insource:"{identity}"',
            "gsrlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|size|sha1|timestamp|mime|mediatype|metadata|extmetadata",
            "iilimit": "1",
            "iiurlwidth": str(contract["query"]["thumbnail_width"]),
        }
        rows: dict[int, dict] = {}
        for _ in range(int(contract["query"]["maximum_requests_per_identity"])):
            requests_used += 1
            if requests_used > int(contract["query"]["maximum_total_requests"]):
                raise CommonsEktarCompanionError("total request ceiling exceeded")
            payload = _request(session, str(contract["api_endpoint"]), params, contract)
            for page in payload.get("query", {}).get("pages", []):
                normalized = normalize_search_page(page)
                rows[normalized["page_id"]] = normalized
            if len(rows) >= int(contract["query"]["maximum_results_per_identity"]):
                break
            continuation = payload.get("continue")
            if not continuation:
                break
            if set(continuation) - {"continue", "gsroffset"}:
                raise CommonsEktarCompanionError("unexpected search continuation")
            params.update({key: str(value) for key, value in continuation.items()})
        pages_by_identity[identity] = sorted(rows.values(), key=lambda row: row["page_id"])
    selected = select_metadata_candidates(sources, pages_by_identity, contract)
    target_root = ROOT / str(contract["candidate_thumbnail_root"])
    target_root.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for row in selected:
        candidate = row["candidate"]
        url = candidate["thumbnail_url"]
        if not url:
            raise CommonsEktarCompanionError("selected candidate lacks thumbnail URL")
        response = session.get(url, timeout=float(contract["query"]["timeout_seconds"]))
        response.raise_for_status()
        payload = response.content
        total_bytes += len(payload)
        if total_bytes > int(contract["pixel_preflight"]["maximum_total_thumbnail_bytes"]):
            raise CommonsEktarCompanionError("thumbnail byte ceiling exceeded")
        name = f"{row['source_page_id']}__{candidate['page_id']}.img"
        path = target_root / name
        with path.open("xb") as handle:
            handle.write(payload)
        row["candidate_local_name"] = name
        row["candidate_thumbnail_bytes"] = len(payload)
        row["candidate_thumbnail_sha256"] = sha256_bytes(payload)
    snapshot = {
        "schema": "neuro-film.sf3-a3e-commons-ektar-companion-discovery-snapshot.v1",
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_bytes(contract_path.read_bytes()),
        "source_rows": sorted(sources, key=lambda row: row["page_id"]),
        "pages_by_identity": pages_by_identity,
        "network_requests": requests_used + len(selected),
        "selected_candidates": selected,
        "thumbnail_bytes": total_bytes,
    }
    output = ROOT / str(contract["metadata_snapshot_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json(snapshot))
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "configs/sf3_a3e_commons_ektar_companion_discovery_v1.json")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path)
    if args.acquire:
        snapshot = _acquire(contract_path, contract)
    else:
        snapshot = json.loads((ROOT / str(contract["metadata_snapshot_output"])).read_text(encoding="utf-8"))
    if args.reverse:
        snapshot = dict(snapshot)
        snapshot["source_rows"] = list(reversed(snapshot["source_rows"]))
        snapshot["selected_candidates"] = list(reversed(snapshot["selected_candidates"]))
    report = audit_snapshot(ROOT, contract, snapshot)
    report_path = args.report or (ROOT / str(contract["report_output"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as handle:
        handle.write(canonical_json(report))
    print(json.dumps({key: report[key] for key in ("decision", "metadata_candidates", "pixel_candidates_passed", "stable_evidence_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
