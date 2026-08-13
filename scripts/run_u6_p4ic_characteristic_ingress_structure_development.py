"""Run one P4IC development process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.opponent_diffusion_photographic_confirmation import (
    CharacteristicIngressAnalyticalRuntime,
    evaluate_with_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(
        (
            ROOT
            / "configs/u6_p4ic_characteristic_ingress_structure_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    result = evaluate_with_runtime(
        contract,
        root=ROOT,
        output_dir=args.output_dir,
        runtime_class=CharacteristicIngressAnalyticalRuntime,
        result_schema="neuro-film.u6-p4ic-characteristic-ingress-structure-development-result.v1",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "automatic_pass": result["automatic_pass"],
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
