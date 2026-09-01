from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.inference.three_stock_preview_cache as cache_module
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


def _publish(preview: Path, source: Path, profile: Path) -> dict[str, object]:
    return publish_three_stock_preview_cache_index(
        preview,
        input_path=source,
        profile_path=profile,
        parent_contract_sha256="a" * 64,
    )


def _stage_entries(preview: Path) -> list[Path]:
    return list(preview.glob(f".{CACHE_INDEX_NAME}.*.stage"))


def test_success_preserves_canonical_bytes_and_parent_inspection(
    tmp_path: Path,
) -> None:
    preview, source, profile = _fixture(tmp_path)

    published = _publish(preview, source, profile)

    expected = (
        json.dumps(published, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert (preview / CACHE_INDEX_NAME).read_bytes() == expected
    assert inspect_three_stock_preview_cache(
        preview, input_path=source, profile_path=profile
    ) == published
    assert _stage_entries(preview) == []


def test_existing_destination_rejects_and_remains_exact(tmp_path: Path) -> None:
    preview, source, profile = _fixture(tmp_path)
    destination = preview / CACHE_INDEX_NAME
    destination.write_bytes(b"foreign-before")

    with pytest.raises(ThreeStockPreviewCacheError, match="already exists"):
        _publish(preview, source, profile)

    assert destination.read_bytes() == b"foreign-before"
    assert _stage_entries(preview) == []


def test_late_foreign_destination_wins_and_remains_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, source, profile = _fixture(tmp_path)
    destination = preview / CACHE_INDEX_NAME
    real_publish = cache_module.publish_create_only

    def inject(stage: Path, final: Path):  # type: ignore[no-untyped-def]
        assert final == destination
        final.write_bytes(b"late-foreign")
        return real_publish(stage, final)

    monkeypatch.setattr(cache_module, "publish_create_only", inject)
    with pytest.raises(ThreeStockPreviewCacheError, match="already exists"):
        _publish(preview, source, profile)

    assert destination.read_bytes() == b"late-foreign"
    assert _stage_entries(preview) == []


def test_publication_failure_removes_only_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, source, profile = _fixture(tmp_path)

    def fail(_stage: Path, _final: Path) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(cache_module, "publish_create_only", fail)
    with pytest.raises(OSError, match="injected publication failure"):
        _publish(preview, source, profile)

    assert not (preview / CACHE_INDEX_NAME).exists()
    assert _stage_entries(preview) == []


def test_foreign_stage_replacement_is_not_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, source, profile = _fixture(tmp_path)

    def replace_then_fail(stage: Path, _final: Path) -> None:
        stage.unlink()
        stage.write_bytes(b"foreign-stage-replacement")
        raise OSError("injected after stage replacement")

    monkeypatch.setattr(cache_module, "publish_create_only", replace_then_fail)
    with pytest.raises(OSError, match="after stage replacement"):
        _publish(preview, source, profile)

    stages = _stage_entries(preview)
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"foreign-stage-replacement"
