#!/usr/bin/env python
"""Run the frozen U5.R2AO1 real Velvia chart explainability pilot."""

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

from src.real_film.velvia_chart_explainability import (  # noqa: E402
    evaluate_chart_proxy,
    load_exact_pairs,
)


CONFIG_SHA256 = "abd20f5d20372a87c024fa48cebe7e4e79cc48d22cae5a247c60a33d7312bbea"
REPORT_SCHEMA = "neuro-film.u5.r2ao1.velvia-chart-explainability-report.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AO1 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO1 config hash mismatch")
    config = json.loads(raw)
    source = config["source"]
    if (
        config["experiment_id"]
        != "u5.r2ao1-velvia-chart-operator-explainability-v1"
        or source["paired_patch_channel_order"]
        != ["film_rgb", "reference_rgb"]
        or source["encoded_space"] != "sRGB"
        or source["working_space"] != "linear_srgb_d65"
        or config["split"]["formal_status"]
        != "development_explainability_not_external_confirmation"
        or config["production_integration_allowed"]
        or config["stock_response_claim_allowed"]
        or config["calibrated_reference_claim_allowed"]
    ):
        raise ValueError("AO1 frozen contract mismatch")
    return config


def run(
    config: dict[str, Any],
    *,
    paired_patches_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    source, target = load_exact_pairs(paired_patches_path, config)
    result = evaluate_chart_proxy(source, target, config)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": _git_commit(),
        "config_sha256": CONFIG_SHA256,
        "source": config["source"],
        **result,
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_write(output_path, _canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ao1_velvia_chart_operator_explainability_v1.json"
        ),
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--paired-patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(
        args.config, expected_sha256=args.expected_config_sha256
    )
    report = run(
        config,
        paired_patches_path=args.paired_patches,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "selected_model": report["selected_model"],
                "decision": report["decision"],
                "one_matrix_mean_confirmation_rgb_rmse": report["aggregate"][
                    "one_matrix"
                ]["mean_confirmation_rgb_rmse"],
                "two_matrix_mean_confirmation_rgb_rmse": report["aggregate"][
                    "two_matrix"
                ]["mean_confirmation_rgb_rmse"],
                "report_sha256": _sha256(_canonical_json(report)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
