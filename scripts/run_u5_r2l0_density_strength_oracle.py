#!/usr/bin/env python
"""Run the frozen U5.R2L0 density-strength Oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.density_strength_oracle import evaluate_oracle  # noqa: E402


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encode(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, value: object) -> str:
    encoded = _encode(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return _sha256(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2l0_density_strength_oracle_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2l0_density_strength_oracle_v1/report.json",
    )
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=(
            ROOT
            / "outputs/u5_r2l0_density_strength_oracle_v1/selected_manifest.json"
        ),
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    selected = (
        args.selected_manifest
        if args.selected_manifest.is_absolute()
        else ROOT / args.selected_manifest
    )
    config_bytes = config_path.read_bytes()
    software_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report, selected_manifest = evaluate_oracle(
        root=ROOT,
        config=json.loads(config_bytes),
        config_sha256=_sha256(config_bytes),
        software_commit=software_commit,
    )
    report_sha256 = _write_atomic(output, report)
    manifest_sha256 = _write_atomic(selected, selected_manifest)
    print(
        json.dumps(
            {
                "automatic_checks_passed": report["automatic_checks_passed"],
                "output": str(output),
                "report_sha256": report_sha256,
                "selected_manifest": str(selected),
                "selected_manifest_sha256": manifest_sha256,
                "aggregates": report["aggregates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automatic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
