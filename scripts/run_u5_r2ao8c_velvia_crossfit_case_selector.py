#!/usr/bin/env python
"""Run frozen AO8C nested cross-fitted case retrieval."""

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

from src.eval.velvia_crossfit_case_selector import (  # noqa: E402
    evaluate_crossfit_case_selector,
)


CONFIG_SHA256 = "f414c263e255338f75fcc791ab54e744d0cd80886e452bd42371897c94e8fa8a"


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
        raise ValueError("AO8C config hash mismatch")
    return json.loads(raw)


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    report = {
        **evaluate_crossfit_case_selector(ROOT, config),
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
        / "configs/u5_r2ao8c_velvia_crossfit_case_selector_v1.json",
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
                "shuffle_p": report["shuffled_label_control"][
                    "empirical_p_value"
                ],
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
