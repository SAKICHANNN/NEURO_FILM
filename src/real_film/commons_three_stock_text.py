"""Audit a bounded Commons exact-text stock snapshot before pixel access."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.real_film.commons_stock_source import normalize_author

CONTRACT_SCHEMA = "neuro-film.sf3-a1c-commons-three-stock-text-metadata-contract.v1"
SNAPSHOT_SCHEMA = "neuro-film.sf3-a1c-commons-three-stock-text-snapshot.v1"
REPORT_SCHEMA = "neuro-film.sf3-a1c-commons-three-stock-text-report.v1"


class CommonsThreeStockTextError(ValueError):
    """Raised when the frozen query or returned metadata is invalid."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_contract(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != CONTRACT_SCHEMA:
        raise CommonsThreeStockTextError("unsupported SF3.A1C contract")
    stocks = [str(row["film_stock_id"]) for row in value.get("stock_queries", [])]
    if stocks != ["fujifilm_velvia_50", "kodak_portra_400", "kodak_ektar_100"]:
        raise CommonsThreeStockTextError("three-stock query order drifted")
    phrases = [str(row["exact_search_phrase"]) for row in value["stock_queries"]]
    if len(phrases) != len({phrase.casefold() for phrase in phrases}):
        raise CommonsThreeStockTextError("stock search phrase drifted")
    return raw, value


def _row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("title", "description_raw_html", "categories")
    )


def audit(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Apply source, rights, author-diversity and ambiguity gates offline."""

    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise CommonsThreeStockTextError("unsupported SF3.A1C snapshot")
    if snapshot.get("image_payloads_downloaded_or_decoded") is not False:
        raise CommonsThreeStockTextError("snapshot is not metadata-only")
    stocks = [str(row["film_stock_id"]) for row in contract["stock_queries"]]
    blocks = snapshot.get("stock_results")
    if not isinstance(blocks, list) or [str(row.get("film_stock_id")) for row in blocks] != stocks:
        raise CommonsThreeStockTextError("snapshot stock inventory drifted")
    cap = int(contract["query_limits"]["maximum_results_per_stock"])
    page_stocks: dict[int, set[str]] = {}
    for block in blocks:
        rows = block.get("rows")
        if not isinstance(rows, list) or len(rows) > cap:
            raise CommonsThreeStockTextError("snapshot row count is invalid")
        ids = [int(row["page_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise CommonsThreeStockTextError("duplicate page identity within stock")
        for page_id in ids:
            page_stocks.setdefault(page_id, set()).add(str(block["film_stock_id"]))
    ambiguous = {page_id for page_id, labels in page_stocks.items() if len(labels) > 1}
    licenses = set(contract["license_policy"]["permissive_license_short_names"])
    exclusions = [re.compile(str(value), re.IGNORECASE) for value in contract["metadata_exclusions"]]
    gates_cfg = contract["metadata_gates"]
    results: list[dict[str, Any]] = []
    for block in blocks:
        stock = str(block["film_stock_id"])
        rows = list(block["rows"])
        eligible = []
        for row in rows:
            author = normalize_author(str(row.get("author_raw_html") or ""))
            valid = (
                int(row["page_id"]) not in ambiguous
                and str(row.get("license_short_name")) in licenses
                and bool(author)
                and bool(str(row.get("file_page_url") or ""))
                and bool(str(row.get("derivative_1600_url") or ""))
                and str(row.get("derivative_1600_url")) != str(row.get("original_url"))
                and min(int(row["width"]), int(row["height"]))
                >= int(gates_cfg["minimum_short_dimension"])
                and not any(pattern.search(_row_text(row)) for pattern in exclusions)
            )
            if valid:
                eligible.append({**dict(row), "normalized_author": author})
        author_counts = Counter(str(row["normalized_author"]) for row in eligible)
        largest_share = max(author_counts.values(), default=0) / max(len(eligible), 1)
        checks = {
            "minimum_search_rows": len(rows) >= int(gates_cfg["minimum_search_rows_per_stock"]),
            "minimum_eligible_rows": len(eligible) >= int(gates_cfg["minimum_eligible_rows_per_stock"]),
            "minimum_unique_authors": len(author_counts)
            >= int(gates_cfg["minimum_unique_authors_per_stock"]),
            "largest_author_share": largest_share
            <= float(gates_cfg["maximum_largest_author_share"]),
        }
        results.append(
            {
                "film_stock_id": stock,
                "exact_search_phrase": block["exact_search_phrase"],
                "reported_total_hits": int(block["reported_total_hits"]),
                "retained_search_rows": len(rows),
                "eligible_rows": len(eligible),
                "unique_authors": len(author_counts),
                "largest_author_share": largest_share,
                "checks": checks,
                "metadata_gate_passed": all(checks.values()),
                "eligible_candidates": eligible,
            }
        )
    ambiguity_gate = len(ambiguous) <= int(gates_cfg["maximum_cross_stock_ambiguous_rows"])
    automatic_pass = all(row["metadata_gate_passed"] for row in results) and ambiguity_gate
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "stock_results": results,
        "cross_stock_ambiguous_page_ids": sorted(ambiguous),
        "cross_stock_ambiguous_row_count": len(ambiguous),
        "ambiguity_gate_passed": ambiguity_gate,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "network_requests": int(snapshot["network_requests"]),
        "image_payload_downloads": 0,
        "pixel_decodes": 0,
        "operator_fits": 0,
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _sha256(_canonical(core))}
