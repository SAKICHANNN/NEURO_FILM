#!/usr/bin/env python
"""Run the frozen U5.R2BR2 photographic tone stress."""

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

from src.eval.kci_velvia_tone_photographic_stress import (  # noqa: E402
    evaluate_photographic_stress,
    sha256_file,
)


CONFIG_SHA256 = "9b57487a105603b91ec53aa3279c8468a2b69d44e6003a3f3613bc526ebaa9e2"
REPORT_SCHEMA = "neuro-film.u5.r2br2.kci-velvia-tone-photographic-stress-report.v1"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _require_clean_tracked_worktree() -> None:
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("BR2 requires a clean tracked worktree")


def _load_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} identity drift")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2br2_kci_velvia_tone_photographic_stress_v1.json",
    )
    parser.add_argument(
        "--parent-report",
        type=Path,
        default=ROOT / "outputs/eval/u5_r2br1_kci_velvia_tone_residual_v1/run_a.json",
    )
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_clean_tracked_worktree()
    config = _load_json(args.config, CONFIG_SHA256, "BR2 config")
    parent_report = _load_json(
        args.parent_report,
        config["parent"]["formal_report_sha256"],
        "BR1 parent report",
    )
    result = evaluate_photographic_stress(
        root=ROOT,
        config=config,
        parent_report=parent_report,
        visual_root=args.visual_root,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "parent": config["parent"],
        "population": config["population"],
        **result,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = _canonical_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "metrics": report["metrics"],
                "report_sha256": hashlib.sha256(encoded).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
