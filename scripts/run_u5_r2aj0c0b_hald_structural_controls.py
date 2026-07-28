#!/usr/bin/env python
"""Run the strict controls-only U5.R2AJ0C0B calibration twice."""

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

from src.eval.haldclut_structural import (
    HaldStructuralError,
    finalize_repeat_evidence,
    load_config_snapshot,
    run_single_control_process,
    write_canonical_json,
)


def _confined(path: Path, *, base: Path, label: str) -> Path:
    base_absolute = Path(os.path.abspath(base))
    candidate = path if path.is_absolute() else ROOT / path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(base_absolute)
    except ValueError as exc:
        raise HaldStructuralError(f"{label} must remain under {base}") from exc
    if not relative.parts:
        raise HaldStructuralError(f"{label} may not equal confinement root")
    current = base_absolute
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise HaldStructuralError(f"{label} may not traverse symlink")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aj0c0b_hald_structural_controls_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2aj0c0b_hald_structural_controls_v2",
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
    config, config_raw_sha, _canonical_sha = load_config_snapshot(
        config_path, expected_raw_sha256=args.expected_config_sha256
    )
    head_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.software_commit is not None and args.software_commit != head_commit:
        raise HaldStructuralError(
            "software commit must equal the current repository HEAD"
        )
    commit = args.software_commit or head_commit

    if args.single_run:
        if not args.expected_config_sha256 or not args.software_commit:
            raise HaldStructuralError(
                "single-run requires parent-bound config SHA and commit"
            )
        hashes = run_single_control_process(
            root=ROOT,
            config_path=config_path,
            output_dir=output_dir,
            expected_config_raw_sha256=config_raw_sha,
            software_commit=commit,
        )
        print(
            json.dumps(
                {
                    **hashes,
                    "state": "control-run-only-non-promoting",
                    "c1_contract_design_allowed": False,
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
                config_raw_sha,
                "--software-commit",
                commit,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise HaldStructuralError(
                "independent C0B child failed:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
    load_config_snapshot(config_path, expected_raw_sha256=config_raw_sha)
    decision = finalize_repeat_evidence(
        first_dir=run_dirs[0],
        second_dir=run_dirs[1],
        config=config,
        config_raw_sha256=config_raw_sha,
        software_commit=commit,
        root=ROOT,
    )
    decision_path = output_dir / "repeat_decision.json"
    repeat_sha = write_canonical_json(decision_path, decision)
    print(
        json.dumps(
            {
                "automatic_pass": decision["automatic_pass"],
                "c1_contract_design_allowed": decision[
                    "c1_contract_design_allowed"
                ],
                "primary_evaluation_allowed": False,
                "repeat_decision_sha256": repeat_sha,
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
