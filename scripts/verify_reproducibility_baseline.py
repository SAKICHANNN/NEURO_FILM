#!/usr/bin/env python3
"""Verify tracked reproducibility inputs and optionally capture local environment facts.

This verifier intentionally checks only committed configuration/test boundaries.
It does not render images, read private evaluation assets, download dependencies,
or claim that the historical regression fixtures are an artifact gold set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "configs" / "reproducibility_baseline.json"


class BaselineVerificationError(ValueError):
    """Raised for an invalid or drifted reproducibility contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineVerificationError(f"cannot load baseline {path}: {exc}") from exc
    if payload.get("schema_version") != 1:
        raise BaselineVerificationError("baseline schema_version must be 1")
    if not isinstance(payload.get("tracked_inputs"), dict) or not payload["tracked_inputs"]:
        raise BaselineVerificationError("baseline requires non-empty tracked_inputs")
    if not isinstance(payload.get("required_test_files"), list) or not payload["required_test_files"]:
        raise BaselineVerificationError("baseline requires non-empty required_test_files")
    return payload


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify(baseline: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    input_hashes: dict[str, str] = {}
    for relative, expected in baseline["tracked_inputs"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing tracked input: {relative}")
            continue
        actual = sha256_file(path)
        input_hashes[relative] = actual
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"invalid expected sha256 for {relative}")
        elif actual != expected:
            errors.append(f"checksum drift: {relative}")
    for relative in baseline["required_test_files"]:
        if not (root / relative).is_file():
            errors.append(f"missing required test: {relative}")
    for field in ("benchmark_registry", "regression_fixture_registry"):
        relative = baseline.get(field)
        if not isinstance(relative, str) or not (root / relative).is_file():
            errors.append(f"missing {field}: {relative!r}")
    return {"ok": not errors, "errors": errors, "input_hashes": input_hashes}


def environment_report(verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("Pillow", "omegaconf", "pytest", "torch")},
        "verification": verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the tracked local reproducibility baseline.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-environment", type=Path, help="Write an ignored local environment report after verification.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable verification report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_baseline(args.baseline)
    verification = verify(baseline)
    report = environment_report(verification)
    if args.write_environment:
        args.write_environment.parent.mkdir(parents=True, exist_ok=True)
        args.write_environment.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("BASELINE_OK" if verification["ok"] else "BASELINE_FAILED")
        for error in verification["errors"]:
            print(f"ERROR: {error}")
    return 0 if verification["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
