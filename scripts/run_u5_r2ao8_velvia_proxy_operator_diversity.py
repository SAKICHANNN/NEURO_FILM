#!/usr/bin/env python
"""Run the frozen AO8 Velvia display-proxy operator diversity audit."""

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

from src.eval.velvia_proxy_operator_diversity import (  # noqa: E402
    evaluate_operator_diversity,
)


CONFIG_SHA256 = "7aae0d70c2fb5ca30a6a115bc89caf00dba299649b2f73efb4fea104b5df12b8"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 != CONFIG_SHA256 or actual != CONFIG_SHA256:
        raise ValueError("AO8 config hash mismatch")
    return json.loads(raw)


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    evaluation = evaluate_operator_diversity(ROOT, config)
    report = {
        **evaluation,
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
        default=ROOT / "configs/u5_r2ao8_velvia_proxy_operator_diversity_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(
        args.config, expected_sha256=args.expected_config_sha256
    )
    report = run(config, args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": hashlib.sha256(
                    _canonical_json(report)
                ).hexdigest(),
                "chart_palette": report["chart_palette"],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
