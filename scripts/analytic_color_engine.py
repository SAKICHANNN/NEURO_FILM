#!/usr/bin/env python3
"""Discover or verify the opt-in analytic colour engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.analytic_render_recipe import (
    SCHEMA_ID as RECIPE_SCHEMA_ID,
)
from src.inference.analytic_render_recipe import (
    verify_analytic_render_recipe_files,
)
from src.inference.analytic_y_chromaticity_profile import (
    ENGINE_ID,
    load_analytic_y_chromaticity_profile,
)

DEFAULT_PROFILE = ROOT / "configs/render_profiles/analytic_y_chromaticity_cb56_v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover or verify the opt-in analytic colour engine."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capabilities", action="store_true")
    action.add_argument("--verify-recipe", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    return parser


def _capabilities(profile_path: Path) -> dict[str, object]:
    runtime = load_analytic_y_chromaticity_profile(profile_path, root=ROOT)
    profile = runtime.profile
    return {
        "schema_id": "kmcfm.analytic-color-engine-capabilities.v1",
        "engine": {
            "cli_value": "analytic-y-chromaticity",
            "engine_id": ENGINE_ID,
            "profile_id": profile["profile_id"],
            "profile_version": profile["profile_version"],
            "profile_sha256": runtime.profile_sha256,
        },
        "input": {
            "working_space": profile["execution"]["input_working_space"],
            "transfer_state": profile["execution"]["input_color_state"],
        },
        "output": {
            "working_space": profile["output"]["working_space"],
            "transfer": profile["output"]["transfer"],
            "bit_depths": profile["output"]["supported_bit_depths"],
            "icc_profile": profile["output"]["icc_profile"],
        },
        "recipe_schema_id": RECIPE_SCHEMA_ID,
        "effects_after_colour_allowed": profile["execution"][
            "effects_after_colour_allowed"
        ],
        "product_default": profile["identity"]["product_default"],
        "research_champion": profile["identity"]["research_champion"],
        "claim_ceiling": profile["evidence"]["claim_ceiling"],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.capabilities:
        payload = _capabilities(args.profile)
    else:
        recipe = json.loads(args.verify_recipe.read_text(encoding="utf-8"))
        verify_analytic_render_recipe_files(
            recipe, profile_path=args.profile, root=ROOT
        )
        payload = {
            "schema_id": "kmcfm.analytic-render-recipe-verification.v1",
            "verified": True,
            "recipe_schema_id": recipe["schema_id"],
            "profile_sha256": recipe["profile"]["sha256"],
            "input_sha256": recipe["input"]["sha256"],
            "output_sha256": recipe["output"]["sha256"],
            "claim_ceiling": recipe["claim"]["claim_ceiling"],
        }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
