"""Retrospective, metadata-only YFCC three-stock connectivity adjudication."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus


class YfccThreeStockConnectivityError(ValueError):
    """Raised when the frozen SF3.A3Q source or contract fails closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _normalized_text(row: dict[str, Any], fields: Sequence[str]) -> str:
    values = [unquote_plus(str(row.get(field, ""))) for field in fields]
    return re.sub(r"\s+", " ", " ".join(values)).strip().casefold()


def _classify_portra_row(row: dict[str, Any], config: dict[str, Any]) -> str:
    generation = config["portra_generation"]
    text = _normalized_text(row, generation["text_fields"])
    if any(
        indicator.casefold() in text for indicator in generation["current_indicators"]
    ):
        return "current_explicit"
    if any(
        indicator.casefold() in text for indicator in generation["legacy_indicators"]
    ):
        return "legacy_explicit"
    return "generation_ambiguous"


def _validate_stock_order(
    order: Sequence[str], expected: Sequence[str]
) -> tuple[str, ...]:
    if len(order) != len(expected) or set(order) != set(expected):
        raise YfccThreeStockConnectivityError(
            "stock_order must contain each frozen target stock exactly once"
        )
    return tuple(order)


def _resolve_source_path(config_path: Path, configured_path: str) -> Path:
    source_path = Path(configured_path)
    if source_path.is_absolute():
        return source_path
    return config_path.resolve().parents[1] / source_path


def run_connectivity_audit(
    config_path: Path,
    *,
    stock_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Adjudicate current-Portra three-stock connectivity from one frozen report."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_stocks = tuple(config["target_stock_ids"])
    order = _validate_stock_order(stock_order or expected_stocks, expected_stocks)
    source_contract = config["source_report"]
    source_path = _resolve_source_path(config_path, source_contract["path"])
    source_payload = source_path.read_bytes()
    if len(source_payload) != int(source_contract["bytes"]):
        raise YfccThreeStockConnectivityError("source report byte size drifted")
    if _sha256(source_payload) != source_contract["sha256"]:
        raise YfccThreeStockConnectivityError("source report SHA-256 drifted")
    source = json.loads(source_payload)
    if source.get("dataset_id") != source_contract["required_dataset_id"]:
        raise YfccThreeStockConnectivityError("source report dataset identity drifted")
    if source.get("image_payloads_downloaded_or_decoded") is not False:
        raise YfccThreeStockConnectivityError("source report is not metadata-only")
    matches = source.get("matches")
    if not isinstance(matches, list):
        raise YfccThreeStockConnectivityError("source report matches are missing")

    rows_by_stock: dict[str, list[dict[str, Any]]] = {
        stock: [] for stock in expected_stocks
    }
    for stock in order:
        rows = [row for row in matches if row.get("film_stock_id") == stock]
        for row in rows:
            if not isinstance(row.get("uid"), str) or not row["uid"]:
                raise YfccThreeStockConnectivityError("target row lacks an author UID")
            if "photoid" not in row:
                raise YfccThreeStockConnectivityError(
                    "target row lacks a photo identity"
                )
        rows_by_stock[stock] = rows

    uid_sets = {
        stock: {str(row["uid"]) for row in rows_by_stock[stock]}
        for stock in expected_stocks
    }
    sorted_stocks = sorted(expected_stocks)
    pairwise: list[dict[str, Any]] = []
    for index, left in enumerate(sorted_stocks):
        for right in sorted_stocks[index + 1 :]:
            shared = sorted(uid_sets[left] & uid_sets[right])
            pairwise.append(
                {
                    "stock_ids": [left, right],
                    "shared_author_uid_count": len(shared),
                    "shared_author_uids": shared,
                }
            )
    all_three = sorted(
        set.intersection(*(uid_sets[stock] for stock in expected_stocks))
    )

    portra_stock = "kodak_portra_400"
    classifications: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_by_stock[portra_stock]:
        if row["uid"] not in all_three:
            continue
        classification = _classify_portra_row(row, config)
        classifications[classification].append(
            {"photoid": int(row["photoid"]), "uid": str(row["uid"])}
        )
    class_names = ("current_explicit", "legacy_explicit", "generation_ambiguous")
    classification_rows = {
        name: sorted(
            classifications[name], key=lambda row: (row["uid"], row["photoid"])
        )
        for name in class_names
    }
    classification_authors = {
        name: sorted({row["uid"] for row in classification_rows[name]})
        for name in class_names
    }
    qualifying_uids = classification_authors["current_explicit"]
    minimum = int(config["gates"]["minimum_current_portra_three_stock_author_uids"])
    operation_counts = {"source_report_reads": 1, **config["operation_limits"]}
    zero_operation_keys = tuple(config["operation_limits"])
    gates = {
        "source_identity_exact": True,
        "source_metadata_only": True,
        "all_target_stocks_present": all(
            rows_by_stock[stock] for stock in expected_stocks
        ),
        "minimum_current_portra_three_stock_author_uids": len(qualifying_uids)
        >= minimum,
        "zero_database_page_image_pixel_fit_render_score": all(
            operation_counts[key] == 0 for key in zero_operation_keys
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema": "neuro-film.sf3-a3q-yfcc-three-stock-generation-connectivity-result.v1",
        "experiment_id": config["experiment_id"],
        "decision": config["decision_if_pass"]
        if passed
        else config["decision_if_fail"],
        "source_report": {
            "path": source_contract["path"],
            "bytes": len(source_payload),
            "sha256": _sha256(source_payload),
            "dataset_id": source["dataset_id"],
        },
        "target_stock_rows": {
            stock: {
                "row_count": len(rows_by_stock[stock]),
                "author_uid_count": len(uid_sets[stock]),
            }
            for stock in sorted_stocks
        },
        "connectivity": {
            "pairwise": pairwise,
            "all_three_author_uid_count": len(all_three),
            "all_three_author_uids": all_three,
        },
        "portra_generation": {
            "classification_row_counts": {
                name: len(classification_rows[name]) for name in class_names
            },
            "classification_author_uids": classification_authors,
            "classification_rows": classification_rows,
            "qualifying_current_portra_three_stock_author_uid_count": len(
                qualifying_uids
            ),
            "qualifying_current_portra_three_stock_author_uids": qualifying_uids,
            "required_minimum": minimum,
        },
        "gates": gates,
        "operation_counts": operation_counts,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(report)
    return report
