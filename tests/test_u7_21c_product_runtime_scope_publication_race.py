from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import render_film
from src.inference.product_desktop import ProductDesktopError

CANONICAL_EKTAR_SHA256 = (
    "fc51547d1e00a0a4a36dce96847afe09d27b3b531b65172d3f32fa2a46d2087d"
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


@pytest.mark.parametrize("selector", ["shortcut", "explicit-profile"])
def test_successful_product_render_validates_twice_and_remains_exact(
    tmp_path: Path, monkeypatch, selector: str
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _source(source)
    original = render_film._validate_runtime_source_scope
    calls = 0

    def recording_validator(root, commit, scope):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        original(root, commit, scope)

    monkeypatch.setattr(
        render_film, "_validate_runtime_source_scope", recording_validator
    )
    selection = (
        ("--product-look", "ektar_100")
        if selector == "shortcut"
        else (
            "--use-render-profile",
            "--render-profile",
            str(render_film.ROOT / "configs/render_profiles/safe_rich_product_v1.json"),
            "--style",
            "ektar_100",
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_film.py",
            str(source),
            *selection,
            "--look-amount",
            "0.65",
            "--output",
            str(output),
            "--write-recipe",
        ],
    )

    assert render_film.main() == 0
    assert calls == 2
    assert _sha256(output) == CANONICAL_EKTAR_SHA256
    assert output.with_suffix(".recipe.json").is_file()


def test_legacy_render_does_not_acquire_product_runtime_scope_policy(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.png"
    output = tmp_path / "legacy.png"
    _source(source)

    def forbidden(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("legacy render crossed product runtime-scope policy")

    monkeypatch.setattr(render_film, "_validate_runtime_source_scope", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_film.py", str(source), "--output", str(output)],
    )

    assert render_film.main() == 0
    assert output.is_file()
