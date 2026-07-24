#!/usr/bin/env python3
"""Decode frozen U5.R2H2 scores and apply the preregistered gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.global_frontier import sha256_file  # noqa: E402
from src.eval.hard_routing_visual_audit import adjudicate_scores  # noqa: E402


def main() -> int:
    config_path = (
        ROOT / "configs" / "u5_r2h2_hard_routing_visual_audit_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / str(config["output_dir"])
    scoring_path = output_dir / "blind_scoring_frozen.json"
    key_path = output_dir / "blind_key.json"
    severe_path = output_dir / "severe_review_frozen.json"
    result = adjudicate_scores(
        json.loads(scoring_path.read_text(encoding="utf-8")),
        json.loads(key_path.read_text(encoding="utf-8")),
        json.loads(severe_path.read_text(encoding="utf-8")),
        config["gates"],
    )
    result.update(
        {
            "experiment_id": config["experiment_id"],
            "config_sha256": sha256_file(config_path),
            "blind_scoring_sha256": sha256_file(scoring_path),
            "blind_key_sha256": sha256_file(key_path),
            "severe_review_sha256": sha256_file(severe_path),
        }
    )
    output_path = output_dir / "adjudication.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": sha256_file(output_path),
                "decision": result["decision"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
