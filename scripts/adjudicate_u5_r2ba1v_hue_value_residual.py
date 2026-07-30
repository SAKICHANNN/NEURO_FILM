from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.factorization_adjudication import adjudicate


def main() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2ba1v_hue_value_residual_adjudication_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = adjudicate(root=ROOT, config=config)
    path = (
        ROOT
        / "outputs/u5_r2ba1v_hue_value_residual_visual_v1/adjudication.json"
    )
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
