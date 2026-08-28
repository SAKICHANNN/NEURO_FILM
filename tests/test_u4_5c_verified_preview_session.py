from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview_cache import (
    ThreeStockPreviewCacheError,
    publish_three_stock_preview_cache_index,
)
from src.inference.three_stock_preview_session import (
    admit_verified_three_stock_preview_session,
    lookup_verified_three_stock_preview_session,
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
        output.write_bytes(f"immutable-{style}".encode("ascii"))
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
    (preview / "preview.json").write_text(json.dumps(manifest), encoding="utf-8")
    publish_three_stock_preview_cache_index(
        preview,
        input_path=source,
        profile_path=profile,
        parent_contract_sha256="a" * 64,
    )
    return preview, source, profile


def test_admission_returns_ordered_immutable_owned_payloads(tmp_path: Path) -> None:
    preview, source, profile = _fixture(tmp_path)
    session = admit_verified_three_stock_preview_session(
        preview, input_path=source, profile_path=profile
    )
    snapshot = lookup_verified_three_stock_preview_session(session)
    assert [row.style_id for row in snapshot.rows] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    assert all(isinstance(row.payload, bytes) for row in snapshot.rows)
    with pytest.raises(FrozenInstanceError):
        snapshot.preview_width = 41  # type: ignore[misc]


def test_warm_lookup_never_reads_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preview, source, profile = _fixture(tmp_path)
    session = admit_verified_three_stock_preview_session(
        preview, input_path=source, profile_path=profile
    )

    def _forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("warm lookup attempted filesystem access")

    monkeypatch.setattr(Path, "read_bytes", _forbidden_read)
    assert lookup_verified_three_stock_preview_session(session) is session.snapshot


def test_admitted_snapshot_survives_disk_tamper_but_new_admission_rejects(
    tmp_path: Path,
) -> None:
    preview, source, profile = _fixture(tmp_path)
    session = admit_verified_three_stock_preview_session(
        preview, input_path=source, profile_path=profile
    )
    original = lookup_verified_three_stock_preview_session(session)
    (preview / "portra_400.preview.png").write_bytes(b"foreign replacement")

    current = lookup_verified_three_stock_preview_session(session)
    assert current is original
    assert current.rows[1].payload == b"immutable-portra_400"
    with pytest.raises(ThreeStockPreviewCacheError, match="output drift"):
        admit_verified_three_stock_preview_session(
            preview, input_path=source, profile_path=profile
        )


def test_admission_rejects_input_and_profile_drift(tmp_path: Path) -> None:
    preview, source, profile = _fixture(tmp_path)
    source.write_bytes(b"changed source")
    with pytest.raises(ThreeStockPreviewCacheError, match="input drift"):
        admit_verified_three_stock_preview_session(
            preview, input_path=source, profile_path=profile
        )
    source.write_bytes(b"source pixels")
    profile.write_text('{"profile":2}\n', encoding="utf-8")
    with pytest.raises(ThreeStockPreviewCacheError, match="profile drift"):
        admit_verified_three_stock_preview_session(
            preview, input_path=source, profile_path=profile
        )
