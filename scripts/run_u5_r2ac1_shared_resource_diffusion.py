from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

from src.eval.dorf_film_response import sha256_file
from src.eval.shared_resource_diffusion import evaluate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ac1_shared_resource_diffusion_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/u5_r2ac1/shared_resource_diffusion_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_path = ROOT / config["parent_decision"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["decision"] != config["parent_decision"]["required_decision"]:
        raise ValueError("AC0 parent decision mismatch")
    result = {
        "schema_version": "u5-r2ac1-shared-resource-diffusion-result-v1",
        "node": "U5.R2AC1",
        "software_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256_file(args.config),
        "parent_decision_sha256": sha256_file(parent_path),
        "evaluation": evaluate(config),
        "claim_ceiling": config["claim_ceiling"],
        "forbidden": config["forbidden"],
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["payload_sha256"] = sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
