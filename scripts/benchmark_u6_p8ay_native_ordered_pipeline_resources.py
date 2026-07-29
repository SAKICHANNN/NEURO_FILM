#!/usr/bin/env python3
"""Apply the frozen 12MP resource gate to the selected ordered pipeline."""

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

import scripts.benchmark_u6_p8aq_native_fastpath_resources as p8aq  # noqa: E402
import scripts.benchmark_u6_p8aw_native_display_v4_resources as p8aw  # noqa: E402
import scripts.benchmark_u6_p8ax_native_ordered_pipeline_grid as p8ax  # noqa: E402


SCHEMA = (
    "neuro_film.u6_p8ay_native_ordered_pipeline_resources_contract.v1"
)
RESULT_SCHEMA = (
    "neuro_film.u6_p8ay_native_ordered_pipeline_resources_result.v1"
)


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or int(config["pipeline_workers"]) != 4
        or int(config["max_in_flight"]) != 4
        or int(config["tile_rows"]) != 32
        or int(config["scenario"]["repeats"]) != 2
    ):
        raise ValueError("unsupported P8AY contract")
    parent = p8aq._load_exact_json(
        ROOT / config["parent_decision"],
        config["parent_decision_sha256"],
    )
    if (
        not str(parent.get("next_leaf", "")).startswith("U6.P8AY")
        or int(parent["result"]["selected_pipeline_workers"]) != 4
        or int(parent["result"]["selected_max_in_flight"]) != 4
    ):
        raise ValueError("P8AY parent selection drift")
    frozen = p8aq._load_exact_json(
        ROOT / config["frozen_resource_contract"],
        config["frozen_resource_contract_sha256"],
    )
    if (
        frozen["scenario"] != config["scenario"]
        or frozen["tile_rows"] != config["tile_rows"]
        or frozen["safety"] != config["safety"]
        or frozen["gates"] != config["gates"]
        or frozen["profile_compiler_config"]
        != config["profile_compiler_config"]
        or frozen["profile_compiler_config_sha256"]
        != config["profile_compiler_config_sha256"]
        or frozen["expected_output_sha256"]
        != config["expected_output_sha256"]
    ):
        raise ValueError("P8AY frozen resource contract drift")
    p8aq._load_exact_json(
        ROOT / config["profile_compiler_config"],
        config["profile_compiler_config_sha256"],
    )
    for name in ("domains", "gaussian", "adjacency", "gauge", "context"):
        if config["component_dll_sha256"].get(name) != frozen[
            "component_dll_sha256"
        ].get(name):
            raise ValueError(f"P8AY frozen {name} identity drift")
    if (
        config["component_dll_sha256"].get("display")
        != "a8a4aede80996d96f72ee805f4ae38d23caef3df786c0f4138a2ee5a37571ce9"
    ):
        raise ValueError("P8AY display v4 identity drift")
    return parent


def _worker(**kwargs: Any) -> None:
    p8ax._worker(
        **kwargs,
        pipeline_workers=4,
        max_in_flight=4,
    )


def _patch_runtime() -> None:
    p8aw._patch_runtime()
    p8aq.validate_contract = validate_contract
    p8aq._worker = _worker
    p8aq.__file__ = str(Path(__file__).resolve())


def _relabel_report(report: dict[str, Any]) -> dict[str, Any]:
    report["schema"] = RESULT_SCHEMA
    report["pipeline_workers"] = 4
    report["max_in_flight"] = 4
    stable = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "worker_elapsed_seconds",
            "wall_seconds",
            "runs",
            "stable_evidence_id",
        }
    }
    report["stable_evidence_id"] = hashlib.sha256(
        p8aq._canonical_bytes(stable)
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/u6_p8ay_native_ordered_pipeline_resources_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/u6_p8ay_native_ordered_pipeline_resources_v1",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile-config")
    parser.add_argument("--domains-dll")
    parser.add_argument("--gaussian-dll")
    parser.add_argument("--adjacency-dll")
    parser.add_argument("--gauge-dll")
    parser.add_argument("--context-dll")
    parser.add_argument("--display-dll")
    parser.add_argument("--worker-output")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--tile-rows", type=int)
    arguments = parser.parse_args()
    _patch_runtime()
    if arguments.worker:
        _worker(
            profile_config=Path(arguments.profile_config),
            domains_dll=Path(arguments.domains_dll),
            gaussian_dll=Path(arguments.gaussian_dll),
            adjacency_dll=Path(arguments.adjacency_dll),
            gauge_dll=Path(arguments.gauge_dll),
            context_dll=Path(arguments.context_dll),
            display_dll=Path(arguments.display_dll),
            output=Path(arguments.worker_output),
            height=arguments.height,
            width=arguments.width,
            tile_rows=arguments.tile_rows,
        )
        return
    config = json.loads((ROOT / arguments.config).read_text())
    output_dir = ROOT / arguments.output_dir
    report = _relabel_report(p8aq.benchmark(config, output_dir))
    p8aq.write_report(output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
