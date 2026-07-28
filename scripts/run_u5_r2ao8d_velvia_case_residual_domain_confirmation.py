#!/usr/bin/env python
"""Run frozen AO8D residual-domain confirmation."""

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

from src.eval.velvia_case_residual_domain_confirmation import (  # noqa: E402
    evaluate_case_residual_domain_confirmation,
)


CONFIG_SHA256 = "ded2a198c15c07e6ddcff959c64c3b2787856d8e9760a781e44e7a09b06f4e19"


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
        raise ValueError("AO8D config hash mismatch")
    return json.loads(raw)


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    report = {
        **evaluate_case_residual_domain_confirmation(ROOT, config),
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
        / "configs/u5_r2ao8d_velvia_case_residual_domain_confirmation_v1.json",
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
                "aggregate": report["aggregate"],
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
