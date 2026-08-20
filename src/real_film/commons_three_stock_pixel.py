"""Bind the SF3.A1C weak-label metadata result to a bounded pixel pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.real_film.commons_stock_pilot import build_selection_manifest

CONTRACT_SCHEMAS = {
    "neuro-film.sf3-a1d-commons-three-stock-pixel-integrity-contract.v1",
    "neuro-film.sf3-a1d2-commons-three-stock-pixel-integrity-contract.v1",
}
METADATA_REPORT_SCHEMA = "neuro-film.sf3-a1c-commons-three-stock-text-report.v1"


class CommonsThreeStockPixelError(ValueError):
    """Raised when the frozen metadata parent or pixel contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") not in CONTRACT_SCHEMAS:
        raise CommonsThreeStockPixelError("unsupported SF3.A1D contract")
    if value.get("allowed_stock_ids") != [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]:
        raise CommonsThreeStockPixelError("three-stock order drifted")
    if value.get("operator_fitting_allowed") is not False:
        raise CommonsThreeStockPixelError("pixel integrity leaf cannot fit operators")
    return value


def load_metadata_report(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(path) != contract["metadata_report_sha256"]:
        raise CommonsThreeStockPixelError("metadata report hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != METADATA_REPORT_SCHEMA or value.get("automatic_pass") is not True:
        raise CommonsThreeStockPixelError("metadata parent did not pass")
    if value.get("stable_evidence_id") != contract["required_metadata_stable_evidence_id"]:
        raise CommonsThreeStockPixelError("metadata stable identity drifted")
    return value


def build_selection(report: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic 24-author-per-stock selection manifest."""

    stock_ids = list(contract["allowed_stock_ids"])
    rows = report.get("stock_results")
    if not isinstance(rows, list) or [row.get("film_stock_id") for row in rows] != stock_ids:
        raise CommonsThreeStockPixelError("metadata stock inventory drifted")
    exclusions = {
        int(page_id) for page_id in contract["selection"].get("excluded_page_ids", {})
    }
    available_ids = {
        int(source["page_id"])
        for row in rows
        for source in row["eligible_candidates"]
    }
    if exclusions - available_ids:
        raise CommonsThreeStockPixelError("manual source exclusion is not in metadata parent")
    snapshot = {
        "categories": [
            {
                "film_stock_id": row["film_stock_id"],
                "label_scope": contract["allowed_label_scope"],
                "files": [
                    source
                    for source in row["eligible_candidates"]
                    if int(source["page_id"]) not in exclusions
                ],
            }
            for row in rows
        ]
    }
    pilot_config = {
        **contract,
        "metadata_snapshot_sha256": contract["metadata_report_sha256"],
    }
    return build_selection_manifest(snapshot, pilot_config)
