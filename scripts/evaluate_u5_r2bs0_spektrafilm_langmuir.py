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

from src.eval.spektrafilm_langmuir import (  # noqa: E402
    evaluate_manifests,
    load_manifest,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2bs0_spektrafilm_langmuir_ablation_v1.json",
    )
    parser.add_argument("--manifest-a", type=Path, required=True)
    parser.add_argument("--manifest-b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/u5_r2bs0/spektrafilm_langmuir_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(args.config)
    manifest_a = load_manifest(args.manifest_a)
    manifest_b = load_manifest(args.manifest_b)
    for manifest in (manifest_a, manifest_b):
        if manifest["config_sha256"] != config_sha256:
            raise ValueError("external render config hash mismatch")
    result = {
        "schema_version": "u5-r2bs0-spektrafilm-langmuir-result-v1",
        "node": "U5.R2BS0",
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": config_sha256,
        "manifest_a_sha256": sha256_file(args.manifest_a),
        "manifest_b_sha256": sha256_file(args.manifest_b),
        "evaluation": evaluate_manifests(
            manifest_a,
            manifest_b,
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
