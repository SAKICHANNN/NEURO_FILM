#!/usr/bin/env python
"""Run the frozen AO5 combined Velvia operator fit."""

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

from src.real_film.combined_velvia_operator import (  # noqa: E402
    evaluate_combined_velvia_operator,
    load_combined_velvia_pairs,
)


CONFIG_SHA256 = "c3f94c07e5351a4cea10e96b6d3b1dd13ee43842441e650dac5bd3b19ddb93bf"
REPORT_SCHEMA = "neuro-film.u5.r2ao5.combined-velvia-operator-report.v1"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AO5 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO5 config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"] != "u5.r2ao5-combined-velvia-operator-v1"
        or int(config["inputs"]["combined_rows"]) != 71
        or config["fit"]["model"] != "one_matrix"
        or config["training_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AO5 frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    inputs = config["inputs"]
    datasets = load_combined_velvia_pairs(
        ROOT / inputs["chart_pairs"],
        ROOT / inputs["palette_pairs"],
        config,
    )
    evaluation = evaluate_combined_velvia_operator(datasets, config)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "input_rows": {
            name: len(source) for name, (source, _) in datasets.items()
        },
        **evaluation,
        "claim_ceiling": config["claim_ceiling"],
    }
    raw = _canonical_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ao5_combined_velvia_operator_v1.json",
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
                "operator_sha256": report["fit"]["operator_sha256"],
                "scores": {
                    name: {
                        "one_matrix_rmse": score["metrics"]["one_matrix"][
                            "rgb_rmse"
                        ],
                        "gain_over_identity": score[
                            "one_matrix_gain_over_identity"
                        ],
                        "gain_over_full_affine": score[
                            "one_matrix_gain_over_full_affine"
                        ],
                    }
                    for name, score in report["scores"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
