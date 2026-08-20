#!/usr/bin/env python3
"""Run the frozen CanonCGT fixed-atlas shared-LUT D0."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.canoncgt_fixed_atlas_shared_lut import (
    apply_lut_bank,
    build_lut_bank,
    evaluate,
)
from src.eval.global_frontier import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "apply", "evaluate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2repid6_canoncgt_fixed_atlas_shared_lut_v1.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    build_dir = output_root / "build"
    apply_dir = output_root / "apply"
    if args.phase == "build":
        result = build_lut_bank(
            root=ROOT,
            config=config,
            output_dir=build_dir,
            device=args.device,
            reverse=args.reverse,
        )
        print(json.dumps({"path": str(result["path"]), "sha256": result["sha256"]}, indent=2, sort_keys=True))
        return 0
    build_manifest = build_dir / "build_manifest.json"
    if args.phase == "apply":
        result = apply_lut_bank(
            root=ROOT,
            config=config,
            build_manifest_path=build_manifest,
            output_dir=apply_dir,
            device=args.device,
            reverse=args.reverse,
        )
        print(json.dumps({"path": str(result["path"]), "sha256": result["sha256"]}, indent=2, sort_keys=True))
        return 0
    result = evaluate(
        root=ROOT,
        config=config,
        build_manifest_path=build_manifest,
        apply_manifest_path=apply_dir / "apply_manifest.json",
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": commit,
        "config_sha256": sha256_file(config_path),
        "build_manifest_sha256": sha256_file(build_manifest),
        "apply_manifest_sha256": sha256_file(apply_dir / "apply_manifest.json"),
        **result,
    }
    scientific = dict(report)
    scientific.pop("software_commit", None)
    scientific_id = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report["scientific_id"] = scientific_id
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path = output_root / "formal_report.json"
    path.write_bytes(encoded)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "decision": result["decision"], "scientific_id": scientific_id}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
