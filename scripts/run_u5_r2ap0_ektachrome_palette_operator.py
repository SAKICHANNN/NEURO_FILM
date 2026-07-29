#!/usr/bin/env python
"""Run frozen AP0 Ektachrome complete-row operator audit."""

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

from src.eval.ektachrome_palette_operator import (  # noqa: E402
    evaluate_ektachrome_palette_operator,
    load_ektachrome_pairs,
    validate_contract,
)


CONFIG_SHA256 = "d1c36a560081256a268b30768d8f994fd76badb84705cff6df24230219a895f8"
REPORT_SCHEMA = "neuro-film.u5.r2ap0.ektachrome-palette-operator-report.v1"


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
            raise RuntimeError("AP0 requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AP0 config hash mismatch")
    config = json.loads(raw)
    validate_contract(ROOT, config)
    return config


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    source_config = config["source"]
    source, target = load_ektachrome_pairs(
        ROOT / source_config["paired_palettes"], config
    )
    direction_config = config["velvia_direction_control"]
    operator_config = json.loads(
        (ROOT / direction_config["operator_config"]).read_bytes()
    )
    evaluation = evaluate_ektachrome_palette_operator(
        source,
        target,
        config,
        velvia_operator_payload=operator_config["witnesses"][
            direction_config["operator_witness_id"]
        ],
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "input_pair_count": len(source),
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
        / "configs/u5_r2ap0_ektachrome_palette_operator_v1.json",
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
                "stable_evidence_id": report["stable_evidence_id"],
                "one_matrix_rgb_rmse": report["cross_validated_metrics"][
                    "one_matrix"
                ]["rgb_rmse"],
                "gain_over_identity": report[
                    "one_matrix_gain_over_identity"
                ],
                "gain_over_full_affine": report[
                    "one_matrix_gain_over_full_affine"
                ],
                "direction": report["velvia_direction_control"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
