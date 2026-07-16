"""Exact-scope preparation and decision logic for SF1.3A YFCC pixels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.real_film.commons_stock_pilot import audit_download_manifest
from src.real_film.yfcc_shared_author_rights import select_shared_author_candidates


class YfccSharedAuthorPixelError(ValueError):
    """Raised when the frozen shared-author pixel scope drifts."""


def prepare_shared_author_pixel_candidates(
    metadata_report: Mapping[str, Any],
    sf11_decision: Mapping[str, Any],
    rights_report: Mapping[str, Any],
    rights_config: Mapping[str, Any],
    pixel_config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact prospective rows for the eight rights-passing UIDs."""
    if rights_report.get("decision") != "pass_live_rights_feasibility":
        raise YfccSharedAuthorPixelError("SF1.2 did not pass live-rights feasibility")
    if rights_report.get("image_payload_download_allowed") is not False:
        raise YfccSharedAuthorPixelError("SF1.2 report unexpectedly allowed image payloads")
    if rights_report.get("operator_fitting_allowed") is not False:
        raise YfccSharedAuthorPixelError("SF1.2 report unexpectedly allowed operator fitting")
    expected_uids = sorted(
        (str(value) for value in pixel_config["expected_usable_shared_author_uids"]),
        key=str.casefold,
    )
    actual_uids = sorted(
        (str(value) for value in rights_report.get("usable_shared_author_uids", [])),
        key=str.casefold,
    )
    if actual_uids != expected_uids:
        raise YfccSharedAuthorPixelError("SF1.2 usable-author set drifted")
    matrix = select_shared_author_candidates(metadata_report, sf11_decision, rights_config)
    stocks = [str(value) for value in pixel_config["stock_ids"]]
    candidates: dict[str, list[dict[str, Any]]] = {stock: [] for stock in stocks}
    for uid in expected_uids:
        for stock in stocks:
            for row in matrix[uid][stock]:
                candidate = dict(row)
                candidate["candidate_rank"] = len(candidates[stock])
                candidates[stock].append(candidate)
    total = sum(len(rows) for rows in candidates.values())
    if total != int(pixel_config["selection"]["expected_candidate_rows"]):
        raise YfccSharedAuthorPixelError("bounded candidate row count drifted")
    cap = int(pixel_config["selection"]["maximum_files_per_uid_per_stock"])
    for stock, rows in candidates.items():
        counts = Counter(str(row["uid"]) for row in rows)
        if max(counts.values(), default=0) > cap:
            raise YfccSharedAuthorPixelError(f"per-UID cap drifted for {stock}")
    return candidates


def evaluate_shared_author_pixel_download(
    stock_results: Mapping[str, Mapping[str, Any]], pixel_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine per-stock bounded downloads and apply the frozen acquisition gate."""
    stocks = [str(value) for value in pixel_config["stock_ids"]]
    if set(stock_results) != set(stocks):
        raise YfccSharedAuthorPixelError("per-stock download result set drifted")
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    seen_photoids: set[int] = set()
    for stock in stocks:
        result = stock_results[stock]
        for row in result.get("rows", []):
            if str(row.get("film_stock_id")) != stock:
                raise YfccSharedAuthorPixelError("downloaded stock identity drifted")
            photoid = int(row["photoid"])
            if photoid in seen_photoids:
                raise YfccSharedAuthorPixelError("duplicate photoid across pixel pilot")
            seen_photoids.add(photoid)
            rows.append(dict(row))
        attempts.extend({"film_stock_id": stock, **dict(row)} for row in result.get("attempts", []))
    rows.sort(key=lambda row: (str(row["film_stock_id"]), str(row["author_uid"]).casefold(), int(row["photoid"])))
    total_bytes = sum(int(row["bytes"]) for row in rows)
    if len(rows) > int(pixel_config["selection"]["maximum_retained_files"]):
        raise YfccSharedAuthorPixelError("retained file ceiling exceeded")
    if total_bytes > int(pixel_config["download_limits"]["maximum_bytes_total"]):
        raise YfccSharedAuthorPixelError("aggregate byte ceiling exceeded")
    stock_counts = Counter(str(row["film_stock_id"]) for row in rows)
    authors_by_stock = {
        stock: {str(row["author_uid"]) for row in rows if str(row["film_stock_id"]) == stock}
        for stock in stocks
    }
    bilateral = sorted(set.intersection(*(authors_by_stock[stock] for stock in stocks)), key=str.casefold)
    gates = pixel_config["pixel_gate"]
    checks = {
        "minimum_files_each_stock": all(
            stock_counts[stock] >= int(gates["minimum_retained_files_per_stock"]) for stock in stocks
        ),
        "minimum_usable_shared_authors": len(bilateral) >= int(gates["minimum_usable_shared_authors"]),
        "within_file_ceiling": all(
            int(row["bytes"]) <= int(pixel_config["download_limits"]["maximum_bytes_per_file"])
            for row in rows
        ),
    }
    return {
        "schema_version": 1,
        "pilot_id": pixel_config["pilot_id"],
        "attempts": attempts,
        "rows": rows,
        "files": len(rows),
        "bytes": total_bytes,
        "stock_counts": {stock: stock_counts[stock] for stock in stocks},
        "usable_shared_author_uids": bilateral,
        "usable_shared_author_count": len(bilateral),
        "checks": checks,
        "acquisition_gate_passed": all(checks.values()),
        "next_stage": (
            "run_hash_decode_duplicate_content_and_visual_audit"
            if all(checks.values())
            else "close_or_narrow_shared_author_pixel_design_without_fitting"
        ),
        "operator_fitting_allowed": False,
        "training_allowed": False,
        "claim_ceiling": pixel_config["claim_ceiling"],
    }


def audit_shared_author_pixels(
    manifest: Mapping[str, Any], *, root: Path, pixel_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the frozen offline integrity audit and add shared-author support evidence."""
    if manifest.get("acquisition_gate_passed") is not True:
        raise YfccSharedAuthorPixelError("pixel acquisition gate did not pass")
    gates = pixel_config["pixel_gate"]
    adapter = {
        "pilot_id": pixel_config["pilot_id"],
        "allowed_stock_ids": list(pixel_config["stock_ids"]),
        "pixel_audit": {
            "minimum_short_dimension": gates["minimum_short_dimension"],
            "near_duplicate_hamming_threshold": gates["near_duplicate_hamming_threshold"],
            "minimum_retained_files_per_stock": gates["minimum_retained_files_per_stock"],
            "minimum_normalized_author_groups_for_learning": gates["minimum_usable_shared_authors"],
            "maximum_largest_normalized_author_share_for_learning": 1.0,
        },
        "claim_ceiling": pixel_config["claim_ceiling"],
    }
    report = audit_download_manifest(manifest, root=root, config=adapter)
    for stock_report in report["stock_source_gates"].values():
        stock_report["bounded_source_support_gate_passed"] = bool(
            stock_report.pop("learning_source_gate_passed")
        )
        stock_report["learning_source_gate_passed"] = False
        stock_report["learning_gate_reason"] = (
            "SF1.3A measures bounded source support only; stock identifiability has not passed"
        )
    stocks = [str(value) for value in pixel_config["stock_ids"]]
    support = {
        uid: {
            stock: sum(
                1
                for row in report["file_records"]
                if str(row["author_uid"]) == uid and str(row["film_stock_id"]) == stock
            )
            for stock in stocks
        }
        for uid in sorted(
            {str(row["author_uid"]) for row in report["file_records"]}, key=str.casefold
        )
    }
    bilateral = [uid for uid, counts in support.items() if all(counts[stock] > 0 for stock in stocks)]
    report.update(
        {
            "shared_author_stock_support": support,
            "bilateral_pixel_authors": bilateral,
            "bilateral_pixel_author_count": len(bilateral),
            "source_content_interpretation": (
                "UID/stock support only; content requires contact-sheet review and no exposure, "
                "process, scanner, roll or physical label is imputed"
            ),
            "operator_fitting_allowed": False,
            "training_allowed": False,
        }
    )
    return report
