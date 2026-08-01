from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.spektrafilm_langmuir import (  # noqa: E402
    adjudicate_blind_observations,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--automatic-report", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    automatic = json.loads(args.automatic_report.read_text(encoding="utf-8"))
    if not automatic["evaluation"]["visual_review_allowed"]:
        raise ValueError("automatic report does not allow visual adjudication")
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    result = {
        "schema_version": "u5-r2bs0-spektrafilm-langmuir-adjudication-v1",
        "node": "U5.R2BS0",
        "automatic_report_sha256": sha256_file(args.automatic_report),
        "observations_sha256": sha256_file(args.observations),
        "mapping_sha256": sha256_file(args.mapping),
        "automatic_summary": automatic["evaluation"]["summary"],
        "visual_adjudication": adjudicate_blind_observations(observations, mapping),
        "claim_ceiling": automatic["claim_ceiling"],
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
