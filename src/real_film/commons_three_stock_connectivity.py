"""Evaluate cross-stock author connectivity without requesting image pixels."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = (
    "neuro-film.sf3-a1f-commons-three-stock-author-connectivity-contract.v1"
)
REPORT_SCHEMA = "neuro-film.sf3-a1f-commons-three-stock-author-connectivity-report.v1"
INPUT_SCHEMA = "neuro-film.sf3-a1c-commons-three-stock-text-report.v1"


class CommonsThreeStockConnectivityError(ValueError):
    """Raised when a frozen connectivity input or identity is invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _author_identity(author_html: str) -> str:
    value = html.unescape(author_html)
    match = re.search(r"flickr\.com/people/([^\"'/?#<]+)", value, re.IGNORECASE)
    if match:
        return f"flickr:{match.group(1).casefold()}"
    match = re.search(r"500px\.com/([^\"'/?#<]+)", value, re.IGNORECASE)
    if match:
        return f"500px:{match.group(1).casefold()}"
    match = re.search(r"wiki/User:([^\"'#<]+)", value, re.IGNORECASE)
    if match:
        return f"commons:{match.group(1).replace('_', ' ').strip().casefold()}"
    plain = re.sub(r"<[^>]+>", " ", value)
    plain = " ".join(plain.split()).casefold()
    if not plain:
        raise CommonsThreeStockConnectivityError("candidate author identity is empty")
    return f"name:{plain}"


def _alias_map(groups: object) -> dict[str, str]:
    if not isinstance(groups, list):
        raise CommonsThreeStockConnectivityError("identity alias groups are invalid")
    result: dict[str, str] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, list) or len(group) < 2:
            raise CommonsThreeStockConnectivityError("identity alias group is invalid")
        normalized = [str(value).casefold() for value in group]
        if len(normalized) != len(set(normalized)):
            raise CommonsThreeStockConnectivityError("duplicate identity within alias group")
        canonical = f"alias:{index}:" + "|".join(sorted(normalized))
        for identity in normalized:
            if identity in result:
                raise CommonsThreeStockConnectivityError("identity appears in multiple aliases")
            result[identity] = canonical
    return result


def evaluate(contract_path: Path, *, root: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise CommonsThreeStockConnectivityError("unsupported SF3.A1F contract")
    stocks = [str(value) for value in contract.get("stock_ids", [])]
    if stocks != ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]:
        raise CommonsThreeStockConnectivityError("three-stock order drifted")
    input_path = root / str(contract["input_report"])
    raw = input_path.read_bytes()
    if _sha256(raw) != str(contract["input_report_sha256"]):
        raise CommonsThreeStockConnectivityError("input report hash drifted")
    source = json.loads(raw)
    if source.get("schema") != INPUT_SCHEMA:
        raise CommonsThreeStockConnectivityError("input report schema drifted")
    if source.get("stable_evidence_id") != contract["input_stable_evidence_id"]:
        raise CommonsThreeStockConnectivityError("input evidence identity drifted")
    if source.get("image_payload_downloads") != 0 or source.get("pixel_decodes") != 0:
        raise CommonsThreeStockConnectivityError("input is not metadata-only")
    blocks = source.get("stock_results")
    if not isinstance(blocks, list) or [row.get("film_stock_id") for row in blocks] != stocks:
        raise CommonsThreeStockConnectivityError("input stock inventory drifted")

    aliases = _alias_map(contract.get("identity_alias_groups"))
    identities_by_stock: dict[str, set[str]] = {}
    rows_by_stock_identity: dict[str, dict[str, list[int]]] = {}
    for block in blocks:
        stock = str(block["film_stock_id"])
        candidates = block.get("eligible_candidates")
        if not isinstance(candidates, list):
            raise CommonsThreeStockConnectivityError("eligible candidate inventory is invalid")
        grouped: dict[str, list[int]] = defaultdict(list)
        seen_pages: set[int] = set()
        for row in candidates:
            page_id = int(row["page_id"])
            if page_id in seen_pages:
                raise CommonsThreeStockConnectivityError("duplicate candidate page identity")
            seen_pages.add(page_id)
            identity = _author_identity(str(row.get("author_raw_html") or ""))
            identity = aliases.get(identity, identity)
            grouped[identity].append(page_id)
        identities_by_stock[stock] = set(grouped)
        rows_by_stock_identity[stock] = dict(grouped)

    pair_counts: dict[str, int] = {}
    pair_identities: dict[str, list[str]] = {}
    for left, right in combinations(stocks, 2):
        key = f"{left}__{right}"
        shared = sorted(identities_by_stock[left] & identities_by_stock[right])
        pair_counts[key] = len(shared)
        pair_identities[key] = shared
    triple = sorted(set.intersection(*(identities_by_stock[stock] for stock in stocks)))
    connected_row_counts = {
        stock: sum(len(rows_by_stock_identity[stock][identity]) for identity in triple)
        for stock in stocks
    }
    gates_cfg = contract["metadata_gates"]
    gates = {
        "minimum_shared_authors_per_pair": all(
            count >= int(gates_cfg["minimum_shared_authors_per_pair"])
            for count in pair_counts.values()
        ),
        "minimum_shared_authors_all_three_stocks": len(triple)
        >= int(gates_cfg["minimum_shared_authors_all_three_stocks"]),
        "minimum_rows_per_stock_in_connected_pool": all(
            count >= int(gates_cfg["minimum_rows_per_stock_in_connected_pool"])
            for count in connected_row_counts.values()
        ),
    }
    automatic_pass = all(gates.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "input_report_sha256": _sha256(raw),
        "stock_ids": stocks,
        "pair_shared_author_counts": pair_counts,
        "pair_shared_author_identities": pair_identities,
        "triple_shared_author_count": len(triple),
        "triple_shared_author_identities": triple,
        "connected_row_counts_by_stock": connected_row_counts,
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

