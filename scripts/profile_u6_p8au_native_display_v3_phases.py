#!/usr/bin/env python3
"""Apply deterministic phase attribution to exact AO6 display v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.profile_u6_p8ar_native_fastpath_phases as p8ar  # noqa: E402
import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
from scripts.benchmark_u6_p8at_native_display_v3_resources import (  # noqa: E402
    _apply_display_v3_rows,
)
from src.eval.physical_native_ao6_display_v3_conformance import (  # noqa: E402
    _load_display_v3,
    build_msvc_native_ao6_display_v3_dll,
)


SCHEMA = "neuro_film.u6_p8au_native_display_v3_phases_contract.v1"
RESULT_SCHEMA = (
    "neuro_film.u6_p8au_native_display_v3_phases_result.v1"
)
OLD_PHASE = "ao6_native_display_v2_seconds"
NEW_PHASE = "ao6_native_display_v3_seconds"


def validate_contract(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != SCHEMA
        or int(config["tile_rows"]) != 32
        or int(config["scenario"]["repeats"]) != 2
    ):
        raise ValueError("unsupported P8AU contract")
    parent = p8ar._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if not str(parent.get("next_leaf", "")).startswith("U6.P8AU"):
        raise ValueError("P8AU parent decision drift")
    p8ar._load_exact_json(
        ROOT / config["resource_contract"],
        config["resource_contract_sha256"],
    )
    p8ar._load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )


def _patch_p8ar_runtime() -> None:
    p8ar.validate_contract = validate_contract
    p8aq.build_msvc_native_ao6_display_v2_dll = (
        build_msvc_native_ao6_display_v3_dll
    )
    p8ar._load_display_v2 = _load_display_v3
    p8ar._apply_display_v2_rows = _apply_display_v3_rows


def _rename_phase_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        (NEW_PHASE if key == OLD_PHASE else key): value
        for key, value in mapping.items()
    }


def _relabel_report(report: dict[str, Any]) -> dict[str, Any]:
    report["schema"] = RESULT_SCHEMA
    report["dominant_phase"] = (
        NEW_PHASE if report["dominant_phase"] == OLD_PHASE
        else report["dominant_phase"]
    )
    report["phase_median_seconds"] = _rename_phase_mapping(
        report["phase_median_seconds"]
    )
    for run in report["runs"]:
        run["dominant_phase"] = (
            NEW_PHASE if run["dominant_phase"] == OLD_PHASE
            else run["dominant_phase"]
        )
        run["phases"] = _rename_phase_mapping(run["phases"])
        run["phase_shares"] = _rename_phase_mapping(run["phase_shares"])
    stable = {
        key: value
        for key, value in report.items()
        if key not in {"runs", "stable_evidence_id"}
    }
    report["stable_evidence_id"] = hashlib.sha256(
        p8ar._canonical_bytes(stable)
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8au_native_display_v3_phases_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8au_native_display_v3_phases_v1",
    )
    arguments = parser.parse_args()
    _patch_p8ar_runtime()
    config = json.loads((ROOT / arguments.config).read_text())
    output_dir = ROOT / arguments.output_dir
    report = _relabel_report(p8ar.profile(config, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
