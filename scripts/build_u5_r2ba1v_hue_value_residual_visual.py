from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.factorization_visual import build_visual_evidence


def main() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2ba1v_hue_value_residual_visual_v1.json"
        ).read_text(encoding="utf-8")
    )
    report = build_visual_evidence(
        root=ROOT,
        config=config,
        output_dir=(
            ROOT / "outputs/u5_r2ba1v_hue_value_residual_visual_v1"
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
