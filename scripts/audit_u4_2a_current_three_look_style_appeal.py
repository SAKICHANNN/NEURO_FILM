#!/usr/bin/env python3
"""Build or formally score the committed U4.2A anonymous review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.current_three_look_style_appeal import (
    canonical_bytes,
    evaluate_observations,
    materialize_blind_package,
    sha256_bytes,
)
from src.filmcase.vision_audit import build_blind_audit

CONFIG_PATH = "configs/u4_2a_current_three_look_style_appeal_v1.json"
OBSERVATIONS_PATH = "configs/u4_2a_current_three_look_style_appeal_observations_v1.json"
BASE_BOUND_PATHS = (
    CONFIG_PATH,
    "docs/planning/U4_2A_CURRENT_THREE_LOOK_STYLE_APPEAL_CONTRACT.md",
    "src/eval/current_three_look_style_appeal.py",
    "scripts/audit_u4_2a_current_three_look_style_appeal.py",
    "tests/test_u4_2a_current_three_look_style_appeal.py",
)


def _run(*args: str) -> bytes:
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True).stdout


def _tracked_clean() -> bool:
    return (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def _git_blob(path: str) -> bytes:
    return _run("git", "show", f"HEAD:{path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "formal"), required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not _tracked_clean():
        raise RuntimeError("U4.2A execution requires a clean tracked tree")
    config: dict[str, Any] = json.loads(_git_blob(CONFIG_PATH))
    build = materialize_blind_package(
        config, ROOT, args.output_dir, reverse=args.order == "reverse"
    )
    if args.mode == "build":
        report = {
            "schema": "neuro_film.u4_2a_blind_build_report.v1",
            **build,
        }
    else:
        if args.report is None:
            raise ValueError("--report is required for formal mode")
        observations_blob = _git_blob(OBSERVATIONS_PATH)
        observations = json.loads(observations_blob)
        if (
            observations.get("public_manifest_sha256")
            != build["scientific_payload"]["public_manifest_sha256"]
        ):
            raise RuntimeError("frozen observations bind a different blind package")
        if (
            observations.get("build_scientific_identity")
            != build["scientific_identity"]
        ):
            raise RuntimeError("frozen observations bind a different build identity")
        expected_sheet_sha256s = [
            row["sha256"] for row in build["scientific_payload"]["sheet_rows"]
        ]
        if observations.get("review_sheet_sha256s") != expected_sheet_sha256s:
            raise RuntimeError("frozen observations bind different review sheets")
        plan = build_blind_audit(
            config["population"]["sample_ids"],
            config["arms"],
            seed=int(config["blind_protocol"]["seed"]),
            rounds=int(config["blind_protocol"]["rounds"]),
        )
        result = evaluate_observations(config, plan, observations)
        bound_paths = (*BASE_BOUND_PATHS, OBSERVATIONS_PATH)
        execution = {
            "source_commit": _run("git", "rev-parse", "HEAD").decode().strip(),
            "git_blob_bindings": {
                path: sha256_bytes(_git_blob(path)) for path in bound_paths
            },
            "tracked_diff_clean": True,
            "order_is_scientifically_invariant": True,
            "builder_mapping_used_in_memory": True,
            "mapping_persisted": False,
            "reviewer_mapping_read_or_reconstructed_before_observation_freeze": observations[
                "mapping_read_or_reconstructed_before_freeze"
            ],
            "observation_freeze_status": observations["status"],
            "reviewer_class": observations["reviewer_class"],
            "review_media_scope": observations["review_media_scope"],
            "review_sheet_read_count": observations["review_sheet_read_count"],
            "review_sheet_sha256s": observations["review_sheet_sha256s"],
            "non_review_media_reads_before_freeze": observations[
                "non_review_media_reads_before_freeze"
            ],
            "score_rubric_sha256": observations["score_rubric_sha256"],
            "new_data_downloads": 0,
            "network_reads": 0,
            "training_runs": 0,
            "operator_fit_runs": 0,
        }
        scientific = {
            **build["scientific_payload"],
            "build_scientific_identity": build["scientific_identity"],
            "observation_sha256": sha256_bytes(observations_blob),
            "result": result,
            "execution": execution,
            "status": result["status"],
        }
        identity = sha256_bytes(canonical_bytes(scientific))
        report = {
            "schema": "neuro_film.u4_2a_current_three_look_style_appeal_formal_report.v1",
            "scientific_payload": scientific,
            "scientific_identity": identity,
            "status": result["status"],
        }
    encoded = canonical_bytes(report)
    destination = args.report or (args.output_dir / "build_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    print(
        json.dumps(
            {
                "report_bytes": len(encoded),
                "report_sha256": hashlib.sha256(encoded).hexdigest(),
                "scientific_identity": report["scientific_identity"],
                "status": report["status"],
                "public_manifest": str(args.output_dir / "public_manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
