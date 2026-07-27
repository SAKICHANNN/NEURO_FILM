from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.eval.spektrafilm_spatial_dir import (
    evaluate_manifests,
    load_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ad1_spektrafilm_spatial_dir_ablation_v1.json",
    )
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--manifest-b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/eval/u5_r2ad1/spektrafilm_spatial_dir_ablation_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_decision"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["decision"] != config["parent_decision"]["required_decision"]:
        raise ValueError("AD0 parent decision mismatch")
    result = {
        "schema_version": "u5-r2ad1-spektrafilm-spatial-dir-result-v1",
        "node": "U5.R2AD1",
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256_file(args.config),
        "parent_decision_sha256": sha256_file(parent_path),
        "manifest_a_sha256": sha256_file(args.manifest_a),
        "manifest_b_sha256": sha256_file(args.manifest_b),
        "evaluation": evaluate_manifests(
            load_manifest(args.manifest_a),
            load_manifest(args.manifest_b),
            config,
            root=ROOT,
        ),
        "claim_ceiling": config["claim_ceiling"],
        "forbidden": config["forbidden"],
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
