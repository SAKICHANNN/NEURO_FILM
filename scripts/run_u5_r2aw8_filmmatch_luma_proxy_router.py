"""Run the held-frame median-code hard-router pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_luma_proxy_router import (  # noqa: E402
    evaluate_luma_proxy_router,
)


def _load_bound(root: Path, row: dict) -> dict:
    raw = (root / row["path"]).read_bytes()
    payload = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != row["sha256"]
        or payload["stable_evidence_id"] != row["stable_evidence_id"]
    ):
        raise ValueError("bound parent report identity drift")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2aw8_filmmatch_luma_proxy_router_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/"
        "luma_proxy_router_report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample_config = json.loads(
        (ROOT / config["parent"]["sample_config"]).read_text(encoding="utf-8")
    )
    _load_bound(ROOT, config["parent"]["selective_oracle"])
    oracle = _load_bound(ROOT, config["parent"]["regime_oracle"])
    report = evaluate_luma_proxy_router(
        extract_pair_datasets(ROOT, sample_config), oracle, config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "router_gate_passed": report["router_gate_passed"],
                "validation_router_candidate_opened": report[
                    "validation_router_candidate_opened"
                ],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
