#!/usr/bin/env python
"""Run the frozen U5.R2AJ0C1 primary Hald frontier twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.haldclut_primary_frontier import (
    EXPECTED_CONFIG_RAW_SHA256,
    HaldPrimaryError,
    finalize_repeat,
    load_config_snapshot,
    run_single_process,
)
from src.eval.haldclut_structural import write_canonical_json


def _confined(path: Path, *, base: Path, label: str) -> Path:
    base_absolute = Path(os.path.abspath(base))
    candidate = path if path.is_absolute() else ROOT / path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(base_absolute)
    except ValueError as exc:
        raise HaldPrimaryError(f"{label} must remain under {base}") from exc
    if not relative.parts:
        raise HaldPrimaryError(f"{label} may not equal confinement root")
    current = base_absolute
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise HaldPrimaryError(f"{label} may not traverse symlink")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aj0c1_hald_primary_structural_frontier_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/u5_r2aj0c1_hald_primary_structural_frontier_v1",
    )
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--software-commit")
    args = parser.parse_args()
    config_path = _confined(
        args.config, base=ROOT / "configs", label="config path"
    )
    output_dir = _confined(
        args.output, base=ROOT / "outputs", label="output path"
    )
    config, raw_sha, _canonical_sha = load_config_snapshot(
        config_path,
        expected_raw_sha256=args.expected_config_sha256
        or EXPECTED_CONFIG_RAW_SHA256,
    )
    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.software_commit is not None and args.software_commit != head_commit:
        raise HaldPrimaryError("software commit must equal repository HEAD")
    commit = args.software_commit or head_commit
    if args.single_run:
        if not args.expected_config_sha256 or not args.software_commit:
            raise HaldPrimaryError(
                "single-run requires parent-bound config SHA and commit"
            )
        hashes = run_single_process(
            root=ROOT,
            config_path=config_path,
            output_dir=output_dir,
            expected_config_raw_sha256=raw_sha,
            software_commit=commit,
        )
        print(
            json.dumps(
                {
                    **hashes,
                    "state": "primary-structural-run-non-photographic",
                    "photographic_contract_design_allowed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run_dirs = [output_dir / "run_a", output_dir / "run_b"]
    script = Path(__file__).resolve()
    for run_dir in run_dirs:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--config",
                str(config_path),
                "--output",
                str(run_dir),
                "--single-run",
                "--expected-config-sha256",
                raw_sha,
                "--software-commit",
                commit,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise HaldPrimaryError(
                "independent C1 child failed:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
    load_config_snapshot(
        config_path, expected_raw_sha256=raw_sha
    )
    decision = finalize_repeat(
        root=ROOT,
        first_dir=run_dirs[0],
        second_dir=run_dirs[1],
        config=config,
        config_raw_sha256=raw_sha,
        software_commit=commit,
    )
    decision_path = output_dir / "repeat_decision.json"
    decision_sha = write_canonical_json(decision_path, decision)
    print(
        json.dumps(
            {
                "automatic_pass": decision["automatic_pass"],
                "decision": decision["decision"],
                "selected_count": decision["selected_count"],
                "photographic_contract_design_allowed": decision[
                    "photographic_contract_design_allowed"
                ],
                "photograph_rendered": False,
                "repeat_decision_sha256": decision_sha,
                "repeat_decision_file_sha256": hashlib.sha256(
                    decision_path.read_bytes()
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if decision["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
