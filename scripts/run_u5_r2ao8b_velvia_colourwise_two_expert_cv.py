#!/usr/bin/env python
"""Run frozen AO8B nested cross-validation."""

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

from src.eval.velvia_colourwise_two_expert_cv import (  # noqa: E402
    evaluate_colourwise_two_expert_cv,
)


CONFIG_SHA256 = "52b5836403d93be9063fd131758faf961bbf0558e0541bbd9e552d021e14bea0"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if (
        expected_sha256 != CONFIG_SHA256
        or hashlib.sha256(raw).hexdigest() != CONFIG_SHA256
    ):
        raise ValueError("AO8B config hash mismatch")
    return json.loads(raw)


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    report = {
        **evaluate_colourwise_two_expert_cv(ROOT, config),
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
    }
    encoded = _canonical_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2ao8b_velvia_colourwise_two_expert_cv_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        load_config(
            args.config, expected_sha256=args.expected_config_sha256
        ),
        args.output,
    )
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "selected_candidate_id": report["selected_candidate_id"],
                "gains": report["gains"],
                "checks": report["checks"],
                "report_sha256": hashlib.sha256(
                    _canonical_json(report)
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
