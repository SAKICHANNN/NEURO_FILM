#!/usr/bin/env python
"""Run frozen AO4C chart-to-pigment cross-domain validation."""

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

from src.real_film.velvia_cross_domain import (  # noqa: E402
    evaluate_cross_domain,
    load_cross_domain_pairs,
)


CONFIG_SHA256 = "0258b12294fd98cedfb2fa987f1fb0aefb9a8dd4aac9d5d9cc0b11613a4adc14"
REPORT_SCHEMA = "neuro-film.u5.r2ao4c.velvia-cross-domain-report.v1"


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
            raise RuntimeError("AO4C requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO4C config hash mismatch")
    config = json.loads(raw)
    if (
        config["experiment_id"]
        != "u5.r2ao4c-velvia-chart-palette-cross-domain-v1"
        or config["fit"]["model"] != "one_matrix"
        or len(config["positive_cross_domain_directions"]) != 2
        or config["training_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AO4C frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    inputs = config["inputs"]
    datasets = load_cross_domain_pairs(
        ROOT / inputs["chart_pairs"],
        ROOT / inputs["palette_pairs"],
        config,
    )
    evaluation = evaluate_cross_domain(datasets, config)
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
        default=ROOT
        / "configs/u5_r2ao4c_velvia_chart_palette_cross_domain_v1.json",
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
                "directions": [
                    {
                        "fit": row["fit_domain"],
                        "confirm": row["confirmation_domain"],
                        "one_matrix_rmse": row["confirmation_metrics"][
                            "one_matrix"
                        ]["rgb_rmse"],
                        "gain_over_identity": row[
                            "one_matrix_gain_over_identity"
                        ],
                        "gain_over_full_affine": row[
                            "one_matrix_gain_over_full_affine"
                        ],
                        "pass": row["automatic_pass"],
                    }
                    for row in report["directions"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
