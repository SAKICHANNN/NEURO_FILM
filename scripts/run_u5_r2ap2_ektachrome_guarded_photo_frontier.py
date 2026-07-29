#!/usr/bin/env python
"""Render and evaluate the frozen AP2 guarded Ektachrome frontier."""

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

from src.eval.global_frontier import sha256_file  # noqa: E402
from src.eval.positive_film_frontier import (  # noqa: E402
    build_blind_sheets,
    evaluate_bank,
    render_bank,
)


CONFIG_SHA256 = "8defb46249fd74115410c3f27732313fe7226fdede7e110457d6208eaea1b8c5"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def load_config(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if (
        not CONFIG_SHA256
        or expected_sha256 != CONFIG_SHA256
        or sha256_file(path) != CONFIG_SHA256
    ):
        raise ValueError("AP2 config hash mismatch")
    config = json.loads(path.read_bytes())
    parent_path = ROOT / str(config["parent_decision"])
    if sha256_file(parent_path) != str(config["parent_decision_sha256"]):
        raise ValueError("AP2 parent decision hash mismatch")
    parent = json.loads(parent_path.read_bytes())
    rendering = config["rendering"]
    guard = rendering["residual_guard"]
    if (
        parent.get("decision") != config["required_parent_decision"]
        or config.get("experiment_id")
        != "u5.r2ap2-ektachrome-guarded-photo-frontier-v1"
        or config.get("strengths") != [0.75, 1.0]
        or config.get("candidate_count") != 2
        or rendering.get("execution_policy")
        != "source_inclusive_residual_guard_v1"
        or rendering.get("hard_clipping_allowed") is not False
        or guard.get("hard_clip_allowed") is not False
        or guard.get("per_image_fit_allowed") is not False
        or config.get("training_allowed")
        or config.get("operator_fitting_allowed")
        or config.get("production_integration_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
    ):
        raise ValueError("AP2 frozen contract mismatch")
    return config


def _require_clean_tracked_worktree() -> None:
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise RuntimeError("AP2 requires a clean tracked worktree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/u5_r2ap2_ektachrome_guarded_photo_frontier_v1.json"
        ),
    )
    parser.add_argument("--expected-config-sha256", default=CONFIG_SHA256)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-visual", action="store_true")
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
    visual: dict[str, Any] = {"status": "not_requested"}
    if args.build_visual and result["shortlist"]:
        visual = {
            "status": "pending_blind_review",
            **build_blind_sheets(
                root=ROOT,
                config=config,
                manifest_path=manifest,
                shortlist=result["shortlist"],
                output_dir=output.parent / "visual",
            ),
        }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": CONFIG_SHA256,
        "manifest_sha256": sha256_file(manifest),
        **result,
        "visual": visual,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = _canonical_json(report)
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
