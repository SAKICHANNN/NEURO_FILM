from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from src.eval.dorf_film_response import audit_archive, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2ab0_dorf_source_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/eval/u5_r2ab0/dorf_source_audit_v1.json",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    archive_path = ROOT / config["source"]["path"]
    audit = audit_archive(archive_path, config)
    result = {
        "schema_version": "u5-r2ab0-dorf-source-audit-v1",
        "node": "U5.R2AB0",
        "config_path": str(args.config.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(args.config),
        "source": {
            "page_url": config["source"]["page_url"],
            "archive_url": config["source"]["archive_url"],
            "archive_path": config["source"]["path"],
            "zip_bytes": archive_path.stat().st_size,
            "zip_sha256": sha256_file(archive_path),
            **audit["archive"],
        },
        "inventory": audit["inventory"],
        "rights": config["rights"],
        "claim_ceiling": config["claim_ceiling"],
        "source_contradiction": {
            "official_page_claimed_points_per_curve": 1000,
            "downloaded_member_actual_samples_per_curve": 1024,
        },
        "decision": "limited_internal_research_source_pass",
        "allowed_next": [
            "preregister a synthetic explicit per-channel response diversity and safety pilot",
            "use exact normalized I-to-B curves as historical response priors",
        ],
        "forbidden": config["forbidden"],
    }
    result["payload_sha256"] = canonical_sha(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
