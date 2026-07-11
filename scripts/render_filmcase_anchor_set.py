#!/usr/bin/env python3
"""Replay normalized color-only owner-anchor variants on a frozen input set.

This is a thin, isolated caller of the legacy deterministic CLI. It does not
modify that CLI, source images, or historical artifacts. These variants are
anchor-inspired normalized replays, not pixel-identical recreations of the
historical mixed-source/grain-enabled contact-sheet outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "pipeline_color_baseline.py"
DEFAULT_SET = ROOT / "outputs" / "filmcase" / "u41_provisional_eval_set.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "filmcase" / "u42_normalized_anchors"

# Derived from the union-40 run IDs, with grain disabled for the color-only
# pass. The control remains intentionally conservative rather than an anchor.
RECIPES: dict[str, tuple[str, ...]] = {
    "anchor01_color_only": ("--strength", "0.55", "--luma-strength", "0.35"),
    "anchor09_color_only": ("--strength", "0.50", "--luma-strength", "0.25", "--gamut-safe", "--gamut-mode", "source", "--output-margin", "4"),
    "anchor53_color_only": ("--strength", "0.72", "--luma-strength", "0.35", "--gamut-safe", "--gamut-mode", "source"),
    "anchor55_color_only": ("--strength", "0.50", "--luma-strength", "0.35", "--gamut-safe", "--gamut-mode", "source"),
    "anchor56_color_only": ("--strength", "0.58", "--luma-strength", "0.35", "--gamut-safe", "--gamut-mode", "source"),
    "bland_safe_rich_control": ("--preset", "safe-rich"),
}

# EXP-VIS-00 challenger: entered only after its single red-highlight check
# passed with the explicit output margin. It is not an owner-preference anchor.
CHALLENGER_RECIPES: dict[str, tuple[str, ...]] = {
    "anchor56_chroma_margin4_challenger": ("--strength", "0.58", "--luma-strength", "0.35", "--gamut-safe", "--gamut-mode", "chroma", "--output-margin", "4"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(source: Path, destination: Path, recipe: tuple[str, ...]) -> list[str]:
    return [sys.executable, str(PIPELINE), str(source), "--style", "velvia_50", "--grain", "0", "--format", "png", "--seed", "20260711", *recipe, "--output", str(destination)]


def load_frozen_set(path: Path, *, include_stress: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_set", payload)
    samples = frozen.get("samples") if isinstance(frozen, dict) else None
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a frozen sample list")
    allowed_splits = {"gold", "stress"} if include_stress else {"gold"}
    available = [row for row in samples if row.get("availability") == "available" and row.get("split") in allowed_splits]
    if not available:
        raise ValueError("frozen set has no available samples")
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description="Render normalized color-only FilmCase anchor replays.")
    parser.add_argument("--frozen-set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-stress", action="store_true", help="Also render stress samples after the gold replay is established.")
    parser.add_argument("--include-exp-vis-challenger", action="store_true", help="Include the unpromoted margin-bounded chroma challenger in an EXP-VIS-00 comparison.")
    parser.add_argument("--write", action="store_true", help="Run rendering; otherwise print the replay plan only.")
    args = parser.parse_args()
    samples = load_frozen_set(args.frozen_set, include_stress=args.include_stress)
    recipes = {**RECIPES, **CHALLENGER_RECIPES} if args.include_exp_vis_challenger else RECIPES
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    plan = [{"candidate_id": candidate, "sample_id": str(sample["id"]), "source": sample["source_path"]} for candidate in recipes for sample in samples]
    if not args.write:
        print(json.dumps({"render_count": len(plan), "plan": plan}, indent=2))
        return 0
    records: list[dict[str, Any]] = []
    for candidate_id, recipe in recipes.items():
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            source = Path(str(sample["source_path"]))
            if not source.is_absolute():
                source = ROOT / source
            destination = candidate_dir / f"{sample['id']}.png"
            command = build_command(source, destination, recipe)
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            records.append({"candidate_id": candidate_id, "sample_id": sample["id"], "source_sha256": sample["source_sha256"], "output": str(destination.relative_to(ROOT)).replace("\\", "/"), "output_sha256": sha256_file(destination), "command": command})
    manifest = {"schema_version": 1, "claim_boundary": "normalized color-only anchor-inspired replay; not historical pixel-identical output", "frozen_set_sha256": sha256_file(args.frozen_set), "recipes": {key: list(value) for key, value in recipes.items()}, "records": records}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"render_count": len(records), "manifest": str((output_dir / 'manifest.json'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
