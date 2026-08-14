"""Evaluate transfer of scalar scanner-setting effects across scanner hardware."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "neuro_film.u6_p6ar_chin_scanner_setting_transfer_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ar_chin_scanner_setting_transfer_result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("observations", [])
    keys = {
        (row.get("scanner"), row.get("software"), row.get("multi_exposure"), row.get("infrared"))
        for row in rows
    }
    expected = {
        (scanner, software, multi, infrared)
        for scanner in ("flatbed", "slide")
        for software in ("1", "2")
        for multi in (False, True)
        for infrared in (False, True)
    }
    if (
        value.get("schema") != SCHEMA
        or len(rows) != 16
        or len({row.get("test") for row in rows}) != 16
        or keys != expected
        or any(
            not math.isfinite(float(row.get("dynamic_range", math.nan)))
            for row in rows
        )
        or value.get("candidate", {}).get("post_score_refit_allowed") is not False
    ):
        raise ValueError("unsupported P6AR contract")
    return value


def _contrasts(
    bank: Mapping[tuple[str, str, bool, bool], float], setting: str
) -> list[dict[str, Any]]:
    rows = []
    for scanner in ("flatbed", "slide"):
        for software in ("1", "2"):
            for other in (False, True):
                if setting == "multi_exposure":
                    off = bank[(scanner, software, False, other)]
                    on = bank[(scanner, software, True, other)]
                    other_name = "infrared"
                else:
                    off = bank[(scanner, software, other, False)]
                    on = bank[(scanner, software, other, True)]
                    other_name = "multi_exposure"
                rows.append(
                    {
                        "scanner": scanner,
                        "software": software,
                        other_name: other,
                        "gain": round(on - off, 12),
                    }
                )
    return rows


def evaluate(contract: Mapping[str, Any]) -> dict[str, Any]:
    bank = {
        (
            str(row["scanner"]),
            str(row["software"]),
            bool(row["multi_exposure"]),
            bool(row["infrared"]),
        ): float(row["dynamic_range"])
        for row in contract["observations"]
    }
    contrasts = {
        setting: _contrasts(bank, setting)
        for setting in ("multi_exposure", "infrared")
    }
    aggregate = {
        setting: round(sum(row["gain"] for row in rows) / len(rows), 12)
        for setting, rows in contrasts.items()
    }
    transfers = []
    for train, held in (("flatbed", "slide"), ("slide", "flatbed")):
        for setting, rows in contrasts.items():
            train_gain = sum(row["gain"] for row in rows if row["scanner"] == train) / 4
            held_rows = [row for row in rows if row["scanner"] == held]
            errors = [abs(row["gain"] - train_gain) for row in held_rows]
            transfers.append(
                {
                    "direction": f"{train}-to-{held}",
                    "setting": setting,
                    "fitted_gain": round(train_gain, 12),
                    "held_gains": [row["gain"] for row in held_rows],
                    "maximum_absolute_error": round(max(errors), 12),
                }
            )
    checks = contract["published_aggregate_checks"]
    tolerance = float(checks["absolute_tolerance"])
    aggregate_replay = (
        abs(aggregate["multi_exposure"] - float(checks["multi_exposure_mean_gain"]))
        <= tolerance
        and abs(aggregate["infrared"] - float(checks["infrared_mean_gain"]))
        <= tolerance
    )
    gates = contract["gates"]
    gate_results = {
        "complete_factorial": len(bank) == 16,
        "published_aggregate_replay": aggregate_replay,
        "multi_exposure_direction": all(
            row["gain"] >= 0.0 for row in contrasts["multi_exposure"]
        ),
        "infrared_direction": all(row["gain"] >= 0.0 for row in contrasts["infrared"]),
        "cross_scanner_transfer": max(
            row["maximum_absolute_error"] for row in transfers
        )
        <= float(gates["maximum_cross_scanner_absolute_contrast_error"]),
        "both_directions": {row["direction"] for row in transfers}
        == {"flatbed-to-slide", "slide-to-flatbed"},
    }
    automatic_pass = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "source": contract["source"],
        "contrasts": contrasts,
        "aggregate_gains": aggregate,
        "transfers": transfers,
        "metrics": {
            "minimum_multi_exposure_gain": min(
                row["gain"] for row in contrasts["multi_exposure"]
            ),
            "maximum_multi_exposure_gain": max(
                row["gain"] for row in contrasts["multi_exposure"]
            ),
            "minimum_infrared_gain": min(row["gain"] for row in contrasts["infrared"]),
            "maximum_infrared_gain": max(row["gain"] for row in contrasts["infrared"]),
            "maximum_cross_scanner_absolute_contrast_error": max(
                row["maximum_absolute_error"] for row in transfers
            ),
        },
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": contract["decision_if_pass"] if automatic_pass else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
