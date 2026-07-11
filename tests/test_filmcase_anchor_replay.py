from __future__ import annotations

from pathlib import Path

import json

from scripts.render_filmcase_anchor_set import CHALLENGER_RECIPES, RECIPES, build_command, load_frozen_set


def test_anchor_replay_commands_are_color_only_and_bound_to_legacy_cli(tmp_path: Path) -> None:
    command = build_command(tmp_path / "input.jpg", tmp_path / "output.png", RECIPES["anchor56_color_only"])
    assert command[1].endswith("scripts\\pipeline_color_baseline.py") or command[1].endswith("scripts/pipeline_color_baseline.py")
    assert command[command.index("--grain") + 1] == "0"
    assert "--gamut-safe" in command
    assert command[command.index("--output") + 1].endswith("output.png")


def test_anchor_replay_defaults_to_gold_only(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"frozen_set": {"samples": [{"id": "gold", "availability": "available", "split": "gold"}, {"id": "stress", "availability": "available", "split": "stress"}]}}), encoding="utf-8")
    assert [row["id"] for row in load_frozen_set(frozen)] == ["gold"]
    assert [row["id"] for row in load_frozen_set(frozen, include_stress=True)] == ["gold", "stress"]


def test_challenger_is_bounded_chroma_with_explicit_margin() -> None:
    recipe = CHALLENGER_RECIPES["anchor56_chroma_margin4_challenger"]
    assert recipe[recipe.index("--gamut-mode") + 1] == "chroma"
    assert recipe[recipe.index("--output-margin") + 1] == "4"
    assert "anchor56_chroma_margin4_challenger" not in RECIPES
