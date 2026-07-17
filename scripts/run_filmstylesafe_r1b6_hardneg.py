"""Run the R1B6 bounded bloom/halation legitimate-local hard-negative prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.eval.filmstylesafe_synthetic import (  # noqa: E402
    SyntheticFailureError,
    canonical_parameter_hash,
    run_bounded_bloom_halation_hardneg_v0,
)
from src.real_film.fsa_owi_pilot import dhash64  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "filmstylesafe_r1b6_hardneg_operator_v0.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "filmstylesafe" / "r1b6" / "report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    params = {
        "operator_id": config["operator_id"],
        "hard_negative_label": config["hard_negative_label"],
        "strength": config["strength"],
        "threshold": config["threshold"],
        "edge_threshold": config["edge_threshold"],
        "min_radius": config["min_radius"],
        "max_radius": config["max_radius"],
        "radius_gamma": config["radius_gamma"],
        "scale_count": config["scale_count"],
    }
    result = run_bounded_bloom_halation_hardneg_v0(
        ROOT / config["input_path"],
        ROOT / config["output_path"],
        params,
    )
    with Image.open(ROOT / config["input_path"]) as image:
        parent_dhash = dhash64(image.convert("RGB"))
    perceptual = parent_dhash + ("0" * (64 - len(parent_dhash)))
    report = {
        **result,
        "parent_scene_id": "u41-01",
        "perceptual_hash": perceptual,
        "hard_negative_label": config["hard_negative_label"],
        "config_parameter_hash": canonical_parameter_hash(params),
        "visual_adjudication_status": "pending Codex or Cursor vision adjudication",
        "claim_ceiling": config["claim_ceiling"],
    }
    if report["parameter_hash"] != report["config_parameter_hash"]:
        raise SyntheticFailureError("parameter hash drift")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "operator_id",
                    "input_hash",
                    "output_hash",
                    "parameter_hash",
                    "output_path",
                    "hard_negative_label",
                    "visual_adjudication_status",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyntheticFailureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
