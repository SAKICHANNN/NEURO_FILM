#!/usr/bin/env python
"""Acquire and audit the frozen U5.R2AJ0B RawTherapee HaldCLUT archive."""

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

from src.eval.haldclut_archive import (
    HaldArchiveError,
    finalize_repeat_evidence,
    load_config_snapshot,
    run_acquisition,
    write_canonical_json,
)


def _confined_cli_path(path: Path, *, base: Path, label: str) -> Path:
    """Confine CLI-controlled reads/writes and reject symlink traversal."""

    base_absolute = Path(os.path.abspath(base))
    if base_absolute.exists() and base_absolute.is_symlink():
        raise HaldArchiveError(f"{label} confinement root may not be a symlink")
    candidate = path if path.is_absolute() else ROOT / path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(base_absolute)
    except ValueError as exc:
        raise HaldArchiveError(f"{label} must remain below {base}") from exc
    if not relative.parts:
        raise HaldArchiveError(f"{label} may not equal its confinement root")
    current = base_absolute
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise HaldArchiveError(f"{label} may not traverse a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(base_absolute.resolve())
    except ValueError as exc:
        raise HaldArchiveError(f"resolved {label} escapes confinement") from exc
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aj0b_haldclut_acquisition_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/u5_r2aj0b_haldclut_acquisition_v1",
    )
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--software-commit")
    args = parser.parse_args()
    config_path = _confined_cli_path(
        args.config, base=ROOT / "configs", label="config path"
    )
    output_dir = _confined_cli_path(
        args.output, base=ROOT / "outputs", label="output path"
    )
    config, config_sha256 = load_config_snapshot(
        config_path, expected_sha256=args.expected_config_sha256
    )
    commit = args.software_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    if args.single_run:
        if not args.expected_config_sha256 or not args.software_commit:
            raise HaldArchiveError(
                "single-run mode requires parent-bound config and commit"
            )
        result = run_acquisition(
            root=ROOT,
            config=config,
            config_sha256=config_sha256,
            output_dir=output_dir,
            software_commit=commit,
        )
        report = result["report"]
        print(
            json.dumps(
                {
                    "archive_sha256": report["archive"]["sha256"],
                    "manifest_sha256": result["manifest_sha256"],
                    "report_sha256": result["report_sha256"],
                    "single_run_pass": report["single_run_pass"],
                    "structural_audit_ready": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["single_run_pass"] else 2

    script = Path(__file__).resolve()
    run_dirs = [output_dir / "run_a", output_dir / "run_b"]
    for run_dir in run_dirs:
        command = [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--output",
            str(run_dir),
            "--single-run",
            "--expected-config-sha256",
            config_sha256,
            "--software-commit",
            commit,
        ]
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise HaldArchiveError(
                "independent acquisition audit failed: "
                f"{completed.stdout}\n{completed.stderr}"
            )

    # Re-read only to prove the on-disk contract did not drift while children
    # were running. The executed dict remains the original byte snapshot.
    load_config_snapshot(config_path, expected_sha256=config_sha256)
    decision = finalize_repeat_evidence(
        first_dir=run_dirs[0],
        second_dir=run_dirs[1],
        config=config,
        config_sha256=config_sha256,
        software_commit=commit,
    )
    decision_path = output_dir / "repeat_decision.json"
    write_canonical_json(decision_path, decision)
    print(
        json.dumps(
            {
                "archive_sha256": decision["archive"]["sha256"]
                if decision["archive"]
                else None,
                "automatic_pass": decision["automatic_pass"],
                "repeat_decision_sha256": hashlib.sha256(
                    decision_path.read_bytes()
                ).hexdigest(),
                "structural_audit_ready": decision[
                    "structural_audit_ready"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if decision["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
