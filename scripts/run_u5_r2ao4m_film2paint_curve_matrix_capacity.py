#!/usr/bin/env python
"""Run the frozen AO4M curve-then-matrix capacity baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.film2paint_curve_matrix_capacity import (  # noqa: E402
    evaluate_curve_matrix_capacity,
    load_ao4m_pairs,
)


CONFIG_SHA256 = "7d3be9c4fc6359ee404f75604a1ed491f45f0e9cb20db942edd9360403fe6d4a"
REPORT_SCHEMA = "neuro-film.u5.r2ao4m.curve-matrix-capacity-report.v1"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AO4M requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO4M config hash mismatch")
    config = json.loads(raw)
    parent = config["parent"]
    parent_raw = (ROOT / parent["decision_path"]).read_bytes()
    parent_decision = json.loads(parent_raw)["decision"]["code"]
    candidate = config["models"]["candidate_parameterization"]
    if (
        config["experiment_id"]
        != "u5.r2ao4m-film2paint-curve-matrix-capacity-v1"
        or _sha256(parent_raw) != parent["decision_sha256"]
        or parent_decision != parent["required_decision"]
        or config["external_method"]["source_code_obtained"]
        or config["external_method"]["dataset_obtained"]
        or config["models"]["candidate_parameter_count"] != 12
        or config["models"]["control_parameter_count"] != 12
        or candidate["hard_output_clipping"]
        or candidate["endpoint_normalization"]
        or config["training_allowed"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AO4M frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    paired = config["paired_source"]
    datasets = load_ao4m_pairs(
        ROOT / paired["chart_pairs_path"],
        ROOT / paired["palette_pairs_path"],
        ROOT / paired["loader_config_path"],
        config,
    )
    result = evaluate_curve_matrix_capacity(datasets, config)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "source_evidence_class": paired["evidence_class"],
        "external_method_implementation_status": config["external_method"][
            "implementation_status"
        ],
        "result": result,
        "automatic_pass": result["automatic_pass"],
        "decision": result["decision"],
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_write(output_path, _canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ao4m_film2paint_curve_matrix_capacity_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, expected_sha256=args.expected_config_sha256)
    report = run(config, args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": _sha256(_canonical_json(report)),
                "candidate_summary": report["result"]["candidate_summary"],
                "failed_checks": [
                    check["name"]
                    for check in report["result"]["automatic_checks"]
                    if not check["passed"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
