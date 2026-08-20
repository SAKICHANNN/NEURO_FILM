"""Compile a bounded three-stock YFCC metadata triangle before page or pixel access."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus, urlparse

CONTRACT_SCHEMA = "neuro-film.sf3-a0y-yfcc-three-stock-triangle-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0y-yfcc-three-stock-triangle-report.v1"


class YfccThreeStockTriangleError(ValueError):
    """Raised when the frozen metadata input or contract drifts."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise YfccThreeStockTriangleError(f"expected JSON object: {path}")
    return raw, value


def _evidence_text(row: Mapping[str, Any]) -> str:
    return unquote_plus(
        " ".join(str(row.get(key) or "") for key in ("title", "description", "usertags"))
    ).casefold()


def _is_flickr_page(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").casefold()
    return parsed.scheme in {"http", "https"} and (
        host == "flickr.com" or host.endswith(".flickr.com")
    )


def evaluate(contract_path: Path, *, root: Path) -> dict[str, Any]:
    """Derive the exact three-stock candidate matrix without network or pixels."""

    contract_raw, contract = _read_object(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise YfccThreeStockTriangleError("unsupported SF3.A0Y contract")
    stocks = [str(value) for value in contract.get("stock_ids", [])]
    if stocks != ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]:
        raise YfccThreeStockTriangleError("three-stock order drifted")
    report_path = root / str(contract["input_report"])
    report_raw, source = _read_object(report_path)
    if _sha256(report_raw) != contract["input_report_sha256"]:
        raise YfccThreeStockTriangleError("input metadata report hash drifted")
    if source.get("image_payloads_downloaded_or_decoded") is not False:
        raise YfccThreeStockTriangleError("input report is not metadata-only")

    expected = contract["expected_stock_support"]
    for stock in stocks:
        actual = source.get("candidate_results", {}).get(stock)
        if not isinstance(actual, Mapping):
            raise YfccThreeStockTriangleError(f"missing stock support: {stock}")
        if int(actual.get("rows", -1)) != int(expected[stock]["rows"]):
            raise YfccThreeStockTriangleError(f"row support drifted: {stock}")
        if int(actual.get("author_uids", -1)) != int(expected[stock]["author_uids"]):
            raise YfccThreeStockTriangleError(f"author support drifted: {stock}")

    rows = [
        dict(row)
        for row in source.get("matches", [])
        if str(row.get("film_stock_id")) in stocks
    ]
    identities = [(int(row["photoid"]), str(row["film_stock_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise YfccThreeStockTriangleError("duplicate stock/photo identity")
    ambiguous_ids = {int(row["photoid"]) for row in source.get("ambiguous_multi_stock_rows", [])}
    if any(photoid in ambiguous_ids for photoid, _ in identities):
        raise YfccThreeStockTriangleError("ambiguous multi-stock row entered matches")

    uid_sets = {
        stock: {str(row["uid"]) for row in rows if row["film_stock_id"] == stock}
        for stock in stocks
    }
    pair_support: dict[str, int] = {}
    for left, right in combinations(stocks, 2):
        key = f"{left}__{right}"
        pair_support[key] = len(uid_sets[left] & uid_sets[right])
        if pair_support[key] != int(contract["expected_pair_shared_uids"][key]):
            raise YfccThreeStockTriangleError(f"pair support drifted: {key}")
    triple_uids = sorted(set.intersection(*(uid_sets[stock] for stock in stocks)), key=str.casefold)

    exclusions = [
        re.compile(str(pattern), re.IGNORECASE)
        for pattern in contract["process_contamination_exclusions"]
    ]
    required_license = str(contract["required_snapshot_license_url"])
    cap = int(contract["metadata_gates"]["maximum_candidates_per_stock_per_uid"])
    matrix: dict[str, dict[str, list[dict[str, Any]]]] = {}
    eligible_uids: list[str] = []
    for uid in triple_uids:
        matrix[uid] = {}
        for stock in stocks:
            candidates = [
                row
                for row in rows
                if str(row["uid"]) == uid
                and str(row["film_stock_id"]) == stock
                and str(row.get("licenseurl")) == required_license
                and not any(pattern.search(_evidence_text(row)) for pattern in exclusions)
            ]
            candidates.sort(key=lambda row: int(row["photoid"]))
            selected = candidates[:cap]
            if any(not _is_flickr_page(str(row.get("pageurl") or "")) for row in selected):
                raise YfccThreeStockTriangleError("candidate page URL is not Flickr")
            matrix[uid][stock] = [
                {
                    "photoid": int(row["photoid"]),
                    "uid": uid,
                    "film_stock_id": stock,
                    "pageurl": str(row["pageurl"]),
                    "downloadurl": str(row["downloadurl"]),
                    "licenseurl": str(row["licenseurl"]),
                }
                for row in selected
            ]
        if all(matrix[uid][stock] for stock in stocks):
            eligible_uids.append(uid)

    page_candidates = sum(
        len(stock_rows)
        for uid_rows in matrix.values()
        for stock_rows in uid_rows.values()
    )
    gates_cfg = contract["metadata_gates"]
    gates = {
        "stock_row_support": all(
            int(expected[stock]["rows"]) >= int(gates_cfg["minimum_rows_per_stock"])
            for stock in stocks
        ),
        "stock_author_support": all(
            int(expected[stock]["author_uids"])
            >= int(gates_cfg["minimum_author_uids_per_stock"])
            for stock in stocks
        ),
        "pair_connectivity": all(
            count >= int(gates_cfg["minimum_shared_uids_per_pair"])
            for count in pair_support.values()
        ),
        "triple_connectivity": len(triple_uids)
        >= int(gates_cfg["minimum_triple_shared_uids"]),
        "eligible_triple_connectivity": len(eligible_uids)
        >= int(gates_cfg["minimum_eligible_triple_uids"]),
        "bounded_page_candidate_count": page_candidates
        <= int(gates_cfg["maximum_page_candidates"]),
        "every_eligible_uid_has_three_stock_arms": all(
            all(matrix[uid][stock] for stock in stocks) for uid in eligible_uids
        ),
    }
    automatic_pass = all(gates.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _sha256(contract_raw),
        "input_report_sha256": _sha256(report_raw),
        "stock_support": {stock: dict(expected[stock]) for stock in stocks},
        "pair_shared_uids": pair_support,
        "triple_shared_uids": triple_uids,
        "triple_shared_uid_count": len(triple_uids),
        "eligible_triple_uids": eligible_uids,
        "eligible_triple_uid_count": len(eligible_uids),
        "page_candidate_count": page_candidates,
        "candidate_matrix": matrix,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "network_requests": 0,
        "image_payload_downloads": 0,
        "pixel_decodes": 0,
        "operator_fits": 0,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}
