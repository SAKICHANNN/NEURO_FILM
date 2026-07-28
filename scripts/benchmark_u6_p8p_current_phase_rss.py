#!/usr/bin/env python3
"""Re-attribute RSS through the exact current P8O production entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_u6_p8d_cpu_consumer import _monitor  # noqa: E402
from scripts.benchmark_u6_p8n_current_phase_rss import (  # noqa: E402
    _run_worker,
)
from src.eval.global_frontier import sha256_file  # noqa: E402
from src.film_physics.profile_compiler import _canonical_bytes  # noqa: E402


SCHEMA = "neuro_film.u6_p8p_current_phase_rss_contract.v1"


def _load_exact_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    measurement = config["measurement"]
    scenario = config["scenario"]
    if (
        config.get("schema") != SCHEMA
        or int(scenario["repeats"]) != 2
        or int(scenario["height"]) * int(scenario["width"]) != 12_000_000
        or int(scenario["tile_rows"]) <= 0
        or float(measurement["sample_interval_seconds"]) <= 0.0
        or not measurement["production_entry_required"]
        or not measurement[
            "timing_and_rss_excluded_from_stable_evidence_id"
        ]
        or measurement["timing_comparison_valid"]
        or measurement["post_result_retuning_allowed"]
        or len(measurement["expected_output_sha256"]) != 64
        or len(set(measurement["phase_order"]))
        != len(measurement["phase_order"])
    ):
        raise ValueError("unsupported U6.P8P contract")
    decision = _load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    profile = _load_exact_json(
        ROOT / config["profile_contract"],
        config["profile_contract_sha256"],
    )
    if (
        not decision["next_leaf"].startswith("U6.P8P")
        or not decision["candidate_retained"]
        or decision["timing_comparison_valid"]
        or decision["production_default_changed"]
    ):
        raise ValueError("U6.P8P parent decision drift")
    return profile


def evaluate(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_contract(config)
    available = int(psutil.virtual_memory().available)
    if available < int(
        config["safety"]["minimum_available_memory_bytes_before_launch"]
    ):
        raise RuntimeError("available memory is below P8P launch floor")
    scenario = config["scenario"]
    runs: list[dict[str, Any]] = []
    for repeat in range(int(scenario["repeats"])):
        worker_output = (
            output_dir / "workers" / f"current-phase-rss-r{repeat + 1}.json"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-output",
            str(worker_output),
            "--config",
            str(ROOT / "configs/u6_p8p_current_phase_rss_v1.json"),
        ]
        observed = _monitor(
            command,
            output=worker_output,
            timeout_seconds=int(scenario["timeout_seconds"]),
            maximum_rss=int(
                config["safety"]["maximum_process_tree_rss_bytes"]
            ),
            interval=float(
                config["measurement"]["sample_interval_seconds"]
            ),
        )
        runs.append({"repeat": repeat + 1, **observed})
        if not observed["success"]:
            break
    expected = config["measurement"]["expected_output_sha256"]
    success = len(runs) == 2 and all(row["success"] for row in runs)
    exact = success and all(
        row["worker"]["output_sha256"] == expected for row in runs
    )
    dominant = {
        row["worker"]["dominant_phase"]
        for row in runs
        if row["success"]
    }
    stable = {
        "schema": (
            "neuro_film.u6_p8p_current_phase_rss_stable_evidence.v1"
        ),
        "phase_order": config["measurement"]["phase_order"],
        "all_success": success,
        "all_output_identity_exact": exact,
        "dominant_phase_stable": len(dominant) == 1,
        "dominant_phase": next(iter(dominant)) if len(dominant) == 1 else None,
    }
    return {
        "schema": "neuro_film.u6_p8p_current_phase_rss_report.v1",
        "node": config["node"],
        "claim_ceiling": config["claim_ceiling"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "total_memory_bytes": int(psutil.virtual_memory().total),
            "available_memory_bytes_before_launch": available,
        },
        "runs": runs,
        "stable_evidence": stable,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(stable)
        ).hexdigest(),
        "decision": (
            "attribution-complete"
            if success and exact and len(dominant) == 1
            else "resource-identity-or-attribution-failure"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u6_p8p_current_phase_rss_v1.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.worker:
        if args.worker_output is None:
            parser.error("--worker-output is required")
        _run_worker(config, args.worker_output)
        return 0
    if args.output is None:
        parser.error("--output is required")
    report = evaluate(config, args.output.parent)
    raw = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"decision={report['decision']}")
    print(f"stable_evidence_id={report['stable_evidence_id']}")
    print(f"report_sha256={hashlib.sha256(raw).hexdigest()}")
    for run in report["runs"]:
        print(
            f"r{run['repeat']}: success={run['success']} "
            f"wall={run['wall_seconds']:.3f}s "
            f"tree_peak={run['peak_process_tree_rss_bytes'] / 2**30:.3f}GiB "
            f"phase={run['worker'].get('dominant_phase')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
