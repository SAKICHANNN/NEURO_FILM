from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

from src.eval.dorf_film_response import (
    audit_archive,
    parse_curves,
    read_archive,
    sha256_file,
    strict_rgb_triplets,
)
from src.eval.dorf_response_diversity import evaluate_bank


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ab1_dorf_response_diversity_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/u5_r2ab1/dorf_response_diversity_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    parent_path = ROOT / config["parent_decision"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent["decision"] != config["parent_decision"]["required_decision"]:
        raise ValueError("AB0 parent decision mismatch")
    archive_config_path = ROOT / config["source"]["archive_config"]
    archive_config = json.loads(archive_config_path.read_text(encoding="utf-8"))
    archive_path = ROOT / archive_config["source"]["path"]
    audit_archive(archive_path, archive_config)
    payload, _ = read_archive(archive_path)
    curves = parse_curves(payload)
    triplets = strict_rgb_triplets(
        curves, eligible_scales=set(config["source"]["eligible_scale_labels"])
    )
    evaluation = evaluate_bank(triplets, config)
    result = {
        "schema_version": "u5-r2ab1-dorf-response-diversity-result-v1",
        "node": "U5.R2AB1",
        "software_commit": _git_commit(),
        "config_path": str(args.config.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(args.config),
        "parent_decision_sha256": sha256_file(parent_path),
        "archive_config_sha256": sha256_file(archive_config_path),
        "archive_sha256": sha256_file(archive_path),
        "evaluation": evaluation,
        "claim_ceiling": config["claim_ceiling"],
        "forbidden": config["forbidden"],
    }
    result["payload_sha256"] = _canonical_sha(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
