from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import render_film

FROZEN_COMMIT = "1" * 40
TRANSIENT_COMMIT = "2" * 40


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:31, :47]
    rgb = np.stack(
        (
            (xx * 5 + yy * 3) % 256,
            (xx * 7 + yy * 11) % 256,
            (xx * 13 + yy * 2) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: tuple[str, ...],
) -> dict[str, object]:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _source(source)
    monkeypatch.setattr(render_film, "_source_commit", lambda _root: FROZEN_COMMIT)
    monkeypatch.setattr(
        render_film,
        "_validate_runtime_source_scope",
        lambda _root, _commit, _scope: None,
    )
    monkeypatch.setattr(
        render_film.subprocess,
        "check_output",
        lambda *_args, **_kwargs: TRANSIENT_COMMIT + "\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_film.py",
            str(source),
            *selection,
            "--output",
            str(output),
            "--write-recipe",
        ],
    )
    assert render_film.main() == 0
    return json.loads(output.with_suffix(".recipe.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "selection",
    [
        ("--product-look", "ektar_100"),
        (
            "--use-render-profile",
            "--render-profile",
            str(render_film.ROOT / "configs/render_profiles/safe_rich_product_v1.json"),
            "--style",
            "ektar_100",
        ),
    ],
)
def test_product_recipe_reuses_invocation_commit_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: tuple[str, ...],
) -> None:
    recipe = _run(tmp_path, monkeypatch, selection)
    assert recipe["software"]["commit"] == FROZEN_COMMIT  # type: ignore[index]


def test_legacy_recipe_retains_live_commit_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _run(tmp_path, monkeypatch, ())
    assert recipe["software"]["commit"] == TRANSIENT_COMMIT  # type: ignore[index]
