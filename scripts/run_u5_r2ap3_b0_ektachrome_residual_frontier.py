#!/usr/bin/env python
"""Render and evaluate the frozen AP3 B0 Ektachrome residual frontier."""

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

from src.eval.b0_real_film_residual_frontier import (  # noqa: E402
    evaluate_bank,
    render_bank,
)
from src.eval.global_frontier import sha256_file  # noqa: E402


CONFIG_SHA256 = "45f81100778a00896473f377d3041f6f9306f5a997ab86304910e286b1840d48"


def load_config(path: Path, *, expected_sha256: str) -> dict:
    if (
        not CONFIG_SHA256
        or expected_sha256 != CONFIG_SHA256
        or sha256_file(path) != CONFIG_SHA256
    ):
        raise ValueError("AP3 config hash mismatch")
    config = json.loads(path.read_bytes())
    parent_path = ROOT / str(config["parent_decision"])
    bank_path = ROOT / str(config["bank_source_config"])
    if sha256_file(parent_path) != str(config["parent_decision_sha256"]):
        raise ValueError("AP3 parent decision hash mismatch")
    if sha256_file(bank_path) != str(config["bank_source_config_sha256"]):
        raise ValueError("AP3 bank source hash mismatch")
    parent = json.loads(parent_path.read_bytes())
    bank_source = json.loads(bank_path.read_bytes())
    expected_strengths = [
        (0.1, 0.25),
        (0.15, 0.35),
        (0.2, 0.5),
        (0.25, 0.65),
    ]
    actual_strengths = [
        (float(row["tone_strength"]), float(row["chroma_strength"]))
        for row in config["residual_bank"]
    ]
    source_strengths = [
        (float(row["tone_strength"]), float(row["chroma_strength"]))
        for row in bank_source["residual_bank"]
    ]
    if (
        parent.get("decision") != config["required_parent_decision"]
        or config.get("experiment_id")
        != "u5.r2ap3-b0-ektachrome-residual-frontier-v1"
        or actual_strengths != expected_strengths
        or source_strengths != expected_strengths
        or config.get("candidate_count") != 4
        or config["factorization"].get("hard_clip_allowed")
        or config["factorization"].get(
            "per_image_fit_or_adjustment_allowed"
        )
        or config.get("training_allowed")
        or config.get("operator_refit_allowed")
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
    ):
        raise ValueError("AP3 frozen contract mismatch")
    return config


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AP3 requires a clean tracked worktree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs/u5_r2ap3_b0_ektachrome_residual_frontier_v1.json"
        ),
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, expected_sha256=args.expected_config_sha256)
    _require_clean_tracked_worktree()
    output_root = ROOT / str(config["output_root"])
    if args.render_dir is not None:
        render_dir = (
            args.render_dir
            if args.render_dir.is_absolute()
            else ROOT / args.render_dir
        )
        result = render_bank(root=ROOT, config=config, output_dir=render_dir)
        print(
            json.dumps(
                {
                    "manifest": str(result["manifest_path"]),
                    "sha256": result["manifest_sha256"],
                    "records": len(result["manifest"]["records"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = args.manifest or output_root / "render_pass1" / "manifest.json"
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    output = args.output or output_root / "automatic_report.json"
    if not output.is_absolute():
        output = ROOT / output
    result = evaluate_bank(root=ROOT, config=config, manifest_path=manifest)
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "manifest_sha256": sha256_file(manifest),
        **result,
        "visual": {
            "status": (
                "required_not_yet_run"
                if result["shortlist"]
                else "not_run_automatic_gate_failed"
            )
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "report": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "automatic_survivors": result["automatic_survivors"],
                "shortlist": result["shortlist"],
                "automatic_decision": result["automatic_decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
