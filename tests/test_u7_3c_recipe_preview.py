from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference.recipe_preview import (
    RecipePreviewError,
    build_recipe_output_preview,
    build_recipe_output_previews,
    render_recipe_preview_html,
)


def _png16(path: Path, *, width: int = 1200, height: int = 800) -> str:
    x = np.linspace(0, 65535, width, dtype=np.uint16)
    y = np.linspace(65535, 0, height, dtype=np.uint16)[:, None]
    rgb = np.stack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            np.full((height, width), 32768, dtype=np.uint16),
        ),
        axis=-1,
    )
    assert cv2.imwrite(str(path), rgb[..., ::-1])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, digest: str) -> dict[str, object]:
    return {
        "status": "valid",
        "recipe_path": "safe.recipe.json",
        "style": "safe_look",
        "output_path": str(path),
        "output_sha256": digest,
        "output_format": "PNG",
        "output_bit_depth": 16,
        "output_label": "film-inspired",
        "evidence_grade": "look-approximation",
    }


def test_preview_is_hash_bound_bounded_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "render.png"
    digest = _png16(path)
    first = build_recipe_output_preview(_entry(path, digest))
    second = build_recipe_output_preview(_entry(path, digest))
    assert first == second
    assert (first.source_width, first.source_height) == (1200, 800)
    assert (first.preview_width, first.preview_height) == (960, 640)
    assert hashlib.sha256(first.png_bytes).hexdigest() == first.png_sha256
    decoded = cv2.imdecode(np.frombuffer(first.png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.dtype == np.uint8
    assert decoded.shape == (640, 960, 3)


def test_preview_rejects_mismatch_before_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "render.png"
    digest = _png16(path, width=64, height=48)
    called = False

    def forbidden_decode(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("decode must not run")

    monkeypatch.setattr(cv2, "imdecode", forbidden_decode)
    with pytest.raises(RecipePreviewError, match="SHA-256 mismatch"):
        build_recipe_output_preview(_entry(path, "0" * 64))
    assert digest != "0" * 64
    assert called is False


def test_preview_rejects_missing_wrong_depth_and_claim(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(RecipePreviewError, match="missing or not regular"):
        build_recipe_output_preview(_entry(missing, "0" * 64))

    path = tmp_path / "render.png"
    rgb8 = np.zeros((16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), rgb8)
    row = _entry(path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RecipePreviewError, match="true RGB16 PNG"):
        build_recipe_output_preview(row)
    row["evidence_grade"] = "calibrated-reference"
    with pytest.raises(RecipePreviewError, match="look-approximation"):
        build_recipe_output_preview(row)


def test_catalog_and_html_are_self_contained_and_escape_labels(tmp_path: Path) -> None:
    path = tmp_path / "render.png"
    digest = _png16(path, width=64, height=48)
    row = _entry(path, digest)
    row["style"] = '<script>alert("x")</script>'
    previews = build_recipe_output_previews({"entries": [row]})
    first = render_recipe_preview_html(previews)
    second = render_recipe_preview_html(previews)
    assert first == second
    text = first.decode("utf-8")
    assert "data:image/png;base64," in text
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text
    assert "default-src 'none'" in text
    assert "img-src data:" in text
    assert "http://" not in text and "https://" not in text
