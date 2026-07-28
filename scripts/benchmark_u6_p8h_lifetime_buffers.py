#!/usr/bin/env python3
"""Benchmark P8G after exact U6.P8H lifetime and buffer reuse."""

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

from scripts.benchmark_u6_p8g_row_staged_context import (  # noqa: E402
    evaluate as evaluate_parent,
)
from src.eval.global_frontier import sha256_file  # noqa: E402
from src.film_physics.profile_compiler import _canonical_bytes  # noqa: E402


SCHEMA = "neuro_film.u6_p8h_lifetime_buffer_resources_contract.v1"


def _load_exact_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "scene_domain_copy_release": "after-input-hash-and-encode",
        "encoded_source_release": "after-source-context-build",
        "display_input_full_float32_copy": False,
        "display_output_reuses_consumed_float64_input": True,
        "array_hash_copy_allowed": False,
        "final_pixel_identity_change_allowed": False,
    }
    if (
        config.get("schema") != SCHEMA
        or config["implementation"] != expected
        or not config["execution"][
            "reuse_parent_scenarios_gates_and_safety_exactly"
        ]
        or not config["execution"][
            "timing_excluded_from_stable_evidence_id"
        ]
        or config["execution"]["post_result_retuning_allowed"]
    ):
        raise ValueError("unsupported U6.P8H contract")
    decision = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    parent = _load_exact_json(
        ROOT / config["parent_contract"],
        config["parent_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8H")
        or decision["performance_target_pass"]
        or decision["production_default_changed"]
    ):
        raise ValueError("U6.P8H parent decision drift")
    return parent


def evaluate(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    parent = validate_contract(config)
    parent_report = evaluate_parent(parent, output_dir)
    parent_stable = parent_report["stable_evidence"]
    stable = {
        "schema": (
            "neuro_film.u6_p8h_lifetime_buffer_stable_evidence.v1"
        ),
        "lifetime_mode": config["implementation"],
        "rows": parent_stable["rows"],
        "all_success": parent_stable["all_success"],
        "all_repeat_identity_exact": parent_stable[
            "all_repeat_identity_exact"
        ],
    }
    return {
        "schema": (
            "neuro_film.u6_p8h_lifetime_buffer_resources_report.v1"
        ),
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "parent_measurement": parent_report,
        "stable_evidence": stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "decision": parent_report["decision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u6_p8h_lifetime_buffer_resources_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(config, args.output.parent)
    raw = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={hashlib.sha256(raw).hexdigest()}")
    runs = report["parent_measurement"]["parent_measurement"]["runs"]
    for run in runs:
        print(
            f"{run['scenario_id']}/r{run['repeat']}: "
            f"success={run['success']} "
            f"wall={run['wall_seconds']:.3f}s "
            f"peak={run['peak_process_tree_rss_bytes'] / 2**30:.3f}GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
