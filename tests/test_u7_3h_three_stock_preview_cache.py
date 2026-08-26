from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview_cache import (
    CACHE_INDEX_NAME,
    ThreeStockPreviewCacheError,
    inspect_three_stock_preview_cache,
    publish_three_stock_preview_cache_index,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.bin"
    profile = tmp_path / "profile.json"
    preview = tmp_path / "preview"
    preview.mkdir()
    source.write_bytes(b"source pixels")
    profile.write_text('{"profile":1}\n', encoding="utf-8")
    rows = []
    for style in ("velvia_50", "portra_400", "ektar_100"):
        output = preview / f"{style}.preview.png"
        output.write_bytes(style.encode("ascii"))
        rows.append(
            {
                "film_stock_id": f"stock:{style}",
                "style_id": style,
                "output_path": str(output),
                "output_sha256": sha256_file(output),
            }
        )
    manifest = {
        "schema_version": "neuro-film.three-stock-direct-preview.v1",
        "input_sha256": sha256_file(source),
        "preview_width": 40,
        "preview_height": 30,
        "preview_pixels": 1200,
        "max_preview_pixels": 2000,
        "look_amount": 1.0,
        "rows": rows,
    }
    (preview / "preview.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return preview, source, profile


def test_publish_and_inspect_cache_uses_only_relative_output_names(tmp_path: Path) -> None:
    preview, source, profile = _fixture(tmp_path)
    published = publish_three_stock_preview_cache_index(
        preview,
        input_path=source,
        profile_path=profile,
        parent_contract_sha256="a" * 64,
    )
    inspected = inspect_three_stock_preview_cache(
        preview, input_path=source, profile_path=profile
    )
    assert inspected == published
    assert [row["filename"] for row in inspected["rows"]] == [
        "velvia_50.preview.png",
        "portra_400.preview.png",
        "ektar_100.preview.png",
    ]
    assert str(tmp_path) not in (preview / CACHE_INDEX_NAME).read_text("utf-8")


def test_cache_rejects_input_profile_output_and_index_drift(tmp_path: Path) -> None:
    preview, source, profile = _fixture(tmp_path)
    publish_three_stock_preview_cache_index(
        preview,
        input_path=source,
        profile_path=profile,
        parent_contract_sha256="b" * 64,
    )
    source.write_bytes(b"changed source")
    with pytest.raises(ThreeStockPreviewCacheError, match="input drift"):
        inspect_three_stock_preview_cache(preview, input_path=source, profile_path=profile)
    source.write_bytes(b"source pixels")
    profile.write_text('{"profile":2}\n', encoding="utf-8")
    with pytest.raises(ThreeStockPreviewCacheError, match="profile drift"):
        inspect_three_stock_preview_cache(preview, input_path=source, profile_path=profile)
    profile.write_text('{"profile":1}\n', encoding="utf-8")
    (preview / "portra_400.preview.png").write_bytes(b"foreign")
    with pytest.raises(ThreeStockPreviewCacheError, match="output drift"):
        inspect_three_stock_preview_cache(preview, input_path=source, profile_path=profile)


def test_publish_rejects_existing_index_and_preview_output_drift(tmp_path: Path) -> None:
    preview, source, profile = _fixture(tmp_path)
    (preview / "velvia_50.preview.png").write_bytes(b"drift")
    with pytest.raises(ThreeStockPreviewCacheError, match="output identity drift"):
        publish_three_stock_preview_cache_index(
            preview,
            input_path=source,
            profile_path=profile,
            parent_contract_sha256="c" * 64,
        )
    (preview / "velvia_50.preview.png").write_bytes(b"velvia_50")
    publish_three_stock_preview_cache_index(
        preview,
        input_path=source,
        profile_path=profile,
        parent_contract_sha256="c" * 64,
    )
    with pytest.raises(ThreeStockPreviewCacheError, match="already exists"):
        publish_three_stock_preview_cache_index(
            preview,
            input_path=source,
            profile_path=profile,
            parent_contract_sha256="c" * 64,
        )
