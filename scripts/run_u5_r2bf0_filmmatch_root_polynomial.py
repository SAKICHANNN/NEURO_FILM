"""Run the frozen FilmMatch root-polynomial directionality baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_chart_pairs import extract_pair_datasets  # noqa: E402
from src.eval.filmmatch_root_polynomial import (  # noqa: E402
    evaluate_root_polynomial,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2bf0_filmmatch_root_polynomial_baseline_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1"
        / "root_polynomial_baseline_report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parent_config = json.loads(
        (ROOT / config["parent"]["config"]).read_text(encoding="utf-8")
    )
    datasets = extract_pair_datasets(ROOT, parent_config)
    report = evaluate_root_polynomial(datasets, config, root=ROOT)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stable_evidence_id": report["stable_evidence_id"],
                "forward_selected": report["forward_selection"]["selected"],
                "forward_capacity_passed": report[
                    "forward_capacity_decision"
                ]["passed"],
                "rendering_opened": report["rendering_opened"],
                "inverse_selected": report["inverse_selection"]["selected"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
