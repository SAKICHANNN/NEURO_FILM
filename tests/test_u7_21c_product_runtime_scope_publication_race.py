from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import render_film
from src.inference.product_desktop import ProductDesktopError


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:43, :61]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3) % 256,
            (xx * 2 + yy * 11) % 256,
            (xx * 13 + yy * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def test_mid_render_scope_drift_rejects_before_bundle_publication(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _source(source)
    calls = 0

    def reject_second(root, commit, scope):  # type: ignore[no-untyped-def]
        nonlocal calls
        del root, commit, scope
        calls += 1
        if calls == 2:
            raise ProductDesktopError("runtime source scope changed")

    monkeypatch.setattr(render_film, "_validate_runtime_source_scope", reject_second)
    monkeypatch.setattr(render_film, "_source_commit", lambda _root: "1" * 40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_film.py",
            str(source),
            "--product-look",
            "ektar_100",
            "--look-amount",
            "0.65",
            "--output",
            str(output),
            "--write-recipe",
        ],
    )

    assert render_film._run_cli() == 1
    assert calls == 2
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert not tuple(tmp_path.glob(".*.stage*"))
