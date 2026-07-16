"""Run the frozen SF1.2 page-only shared-author live-rights preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.yfcc_shared_author_rights import (  # noqa: E402
    run_shared_author_rights_preflight,
    select_shared_author_candidates,
)
from src.real_film.yfcc_stock_source import atomic_json  # noqa: E402


def _load_hashed(path: Path, expected: str) -> dict:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise SystemExit(f"input hash drift: {path}")
    return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_yfcc_shared_author_rights_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = _load_hashed(ROOT / config["input_report"], str(config["input_report_sha256"]))
    decision = _load_hashed(ROOT / config["input_decision"], str(config["input_decision_sha256"]))
    matrix = select_shared_author_candidates(report, decision, config)
    result = run_shared_author_rights_preflight(matrix, config)
    digest = atomic_json(ROOT / config["report"], result)
    print(
        json.dumps(
            {
                "page_requests": result["page_requests"],
                "usable_shared_authors": result["usable_shared_author_count"],
                "decision": result["decision"],
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
