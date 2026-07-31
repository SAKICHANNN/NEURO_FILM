#!/usr/bin/env python
"""Run the frozen AO4R mature root-polynomial baseline."""

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

from src.eval.balica_root_polynomial_baseline import (  # noqa: E402
    evaluate_balica_root_polynomial,
    load_pairs,
)


CONFIG_SHA256 = "83454d5e575521c9d2c6d089623a6a3206c2156e6171e3ecd038da303c1465b3"
REPORT_SCHEMA = "neuro-film.u5.r2ao4r.balica-root-polynomial-report.v1"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _require_clean_tracked_worktree() -> None:
    for command in (("git", "diff", "--quiet"), ("git", "diff", "--cached", "--quiet")):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AO4R requires a clean tracked worktree")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected_sha256 != CONFIG_SHA256 or _sha256(raw) != CONFIG_SHA256:
        raise ValueError("AO4R config hash mismatch")
    config = json.loads(raw)
    for parent in config["parents"].values():
        parent_raw = (ROOT / parent["path"]).read_bytes()
        payload = json.loads(parent_raw)
        observed = payload.get("decision")
        if isinstance(observed, dict):
            observed = observed.get("code")
        if _sha256(parent_raw) != parent["sha256"] or observed != parent["required_decision"]:
            raise ValueError("AO4R parent identity or decision drift")
    if (
        config["external_method"]["dataset_publicly_downloadable"]
        or config["models"]["candidate"] != "root_polynomial_degree3_gamma_prophoto"
        or config["models"]["hard_output_clipping"]
        or config["models"]["include_intercept"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AO4R frozen contract mismatch")
    return config


def run(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    _require_clean_tracked_worktree()
    result = evaluate_balica_root_polynomial(load_pairs(ROOT, config), config)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "result": result,
        "automatic_pass": result["automatic_pass"],
        "decision": result["decision"],
        "claim_ceiling": config["claim_ceiling"],
    }
    _atomic_write(output_path, _canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ao4r_balica_root_polynomial_baseline_v1.json",
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, expected_sha256=args.expected_config_sha256)
    report = run(config, args.output)
    print(
        json.dumps(
            {
                "automatic_pass": report["automatic_pass"],
                "decision": report["decision"],
                "report_sha256": _sha256(_canonical_json(report)),
                "candidate_summary": report["result"]["candidate_summary"],
                "failed_checks": [
                    row["name"]
                    for row in report["result"]["automatic_checks"]
                    if not row["passed"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
