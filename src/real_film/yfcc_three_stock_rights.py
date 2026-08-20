"""Bounded page-only rights verification for the SF3 three-stock triangle."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests

from src.real_film.yfcc_shared_author_rights import _request_page

CONTRACT_SCHEMA = "neuro-film.sf3-a0z-yfcc-three-stock-live-rights-contract.v1"
REPORT_SCHEMA = "neuro-film.sf3-a0z-yfcc-three-stock-live-rights-report.v1"


class YfccThreeStockRightsError(ValueError):
    """Raised when the frozen metadata handoff or request budget drifts."""


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
        raise YfccThreeStockRightsError(f"expected JSON object: {path}")
    return raw, value


def load_candidate_matrix(
    contract_path: Path, *, root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Load the exact metadata result and validate its bounded matrix."""

    contract_raw, contract = _read_object(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise YfccThreeStockRightsError("unsupported SF3.A0Z contract")
    report_path = root / str(contract["input_report"])
    report_raw, report = _read_object(report_path)
    if _sha256(report_raw) != contract["input_report_sha256"]:
        raise YfccThreeStockRightsError("input metadata report hash drifted")
    if report.get("stable_evidence_id") != contract["input_stable_evidence_id"]:
        raise YfccThreeStockRightsError("input stable evidence identity drifted")
    if report.get("decision") != contract["required_input_decision"]:
        raise YfccThreeStockRightsError("input metadata decision does not open rights review")
    if report.get("image_payload_download_allowed") is not False:
        raise YfccThreeStockRightsError("input metadata report authorizes pixels")
    stocks = [str(value) for value in contract["stock_ids"]]
    uids = [str(value) for value in contract["expected_eligible_uids"]]
    if len(uids) != len(set(uids)) or len(stocks) != 3 or len(stocks) != len(set(stocks)):
        raise YfccThreeStockRightsError("stock or UID inventory drifted")
    matrix = report.get("candidate_matrix")
    if not isinstance(matrix, Mapping) or sorted(matrix, key=str.casefold) != sorted(
        uids, key=str.casefold
    ):
        raise YfccThreeStockRightsError("candidate UID inventory drifted")
    normalized: dict[str, dict[str, list[dict[str, Any]]]] = {}
    photoids: set[int] = set()
    count = 0
    for uid in uids:
        arms = matrix.get(uid)
        if not isinstance(arms, Mapping) or set(arms) != set(stocks):
            raise YfccThreeStockRightsError("candidate stock-arm inventory drifted")
        normalized[uid] = {}
        for stock in stocks:
            rows = arms[stock]
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
                raise YfccThreeStockRightsError("candidate stock arm is empty")
            normalized[uid][stock] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise YfccThreeStockRightsError("candidate row is invalid")
                photoid = int(row["photoid"])
                if photoid in photoids:
                    raise YfccThreeStockRightsError("candidate photo identity is duplicated")
                if str(row["uid"]) != uid or str(row["film_stock_id"]) != stock:
                    raise YfccThreeStockRightsError("candidate row role drifted")
                photoids.add(photoid)
                normalized[uid][stock].append(dict(row))
                count += 1
    if count != int(contract["expected_page_candidates"]):
        raise YfccThreeStockRightsError("candidate page count drifted")
    if count > int(contract["selection"]["maximum_page_requests"]):
        raise YfccThreeStockRightsError("candidate matrix exceeds request budget")
    contract["contract_sha256"] = _sha256(contract_raw)
    return contract, normalized


def run(
    contract_path: Path,
    *,
    root: Path,
    session: requests.Session | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Verify current page rights, stopping each arm at its first valid page."""

    contract, matrix = load_candidate_matrix(contract_path, root=root)
    client = session or requests.Session()
    client.headers["User-Agent"] = str(contract["user_agent"])
    stocks = [str(value) for value in contract["stock_ids"]]
    attempts: list[dict[str, Any]] = []
    author_results: dict[str, Any] = {}
    for uid in contract["expected_eligible_uids"]:
        retained: dict[str, int | None] = {}
        arm_pass: dict[str, bool] = {}
        for stock in stocks:
            retained[stock] = None
            arm_pass[stock] = False
            for row in matrix[uid][stock]:
                attempt = _request_page(client, row, contract, sleep_fn=sleep_fn)
                attempts.append(attempt)
                sleep_fn(float(contract["request_limits"]["request_interval_seconds"]))
                if attempt["live_cc_by_2_confirmed"]:
                    retained[stock] = int(row["photoid"])
                    arm_pass[stock] = True
                    break
        author_results[uid] = {
            "stock_live_rights": arm_pass,
            "retained_photoids": retained,
            "usable_three_stock_uid": all(arm_pass.values()),
        }
    if len(attempts) > int(contract["selection"]["maximum_page_requests"]):
        raise YfccThreeStockRightsError("executed request count exceeds budget")
    usable = [
        uid
        for uid in contract["expected_eligible_uids"]
        if author_results[uid]["usable_three_stock_uid"]
    ]
    automatic_pass = len(usable) >= int(contract["minimum_usable_three_stock_uids"])
    stable_core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract["contract_sha256"],
        "input_report_sha256": contract["input_report_sha256"],
        "stock_ids": stocks,
        "author_results": author_results,
        "usable_three_stock_uids": usable,
        "usable_three_stock_uid_count": len(usable),
        "minimum_usable_three_stock_uids": int(contract["minimum_usable_three_stock_uids"]),
        "page_requests": len(attempts),
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "image_payload_download_allowed": False,
        "operator_fitting_allowed": False,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        **stable_core,
        "attempts": attempts,
        "stable_evidence_id": _sha256(_canonical(stable_core)),
    }
