from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import src.inference.three_stock_preview_cache as cache_module
from src.inference.render_contract import sha256_file
from src.inference.three_stock_preview_cache import (
    CACHE_INDEX_NAME,
    ThreeStockPreviewCacheError,
    inspect_receipt_bound_three_stock_preview_cache,
    publish_three_stock_preview_cache_index,
)
from src.inference.three_stock_preview_session import (
    admit_receipt_bound_three_stock_preview_session,
    lookup_receipt_bound_three_stock_preview_session,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
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
    return preview, source, profile, sha256_file(preview / CACHE_INDEX_NAME)


def test_bound_inspection_reads_and_parses_the_exact_index_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, source, profile, receipt = _fixture(tmp_path)
    original_read_bytes = Path.read_bytes
    index_reads = 0

    def _counted_read_bytes(path: Path) -> bytes:
        nonlocal index_reads
        if path == preview / CACHE_INDEX_NAME:
            index_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _counted_read_bytes)
    index = inspect_receipt_bound_three_stock_preview_cache(
        preview,
        input_path=source,
        profile_path=profile,
        cache_index_sha256=receipt,
    )
    assert index_reads == 1
    assert index["parent_contract_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "receipt",
    ["A" * 64, "a" * 63, "g" * 64, 7],
)
def test_invalid_receipt_rejects_before_any_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: object,
) -> None:
    preview, source, profile, _ = _fixture(tmp_path)

    def _forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("invalid receipt reached filesystem")

    monkeypatch.setattr(Path, "read_bytes", _forbidden_read)
    with pytest.raises(ThreeStockPreviewCacheError, match="lowercase 64-hex"):
        inspect_receipt_bound_three_stock_preview_cache(
            preview,
            input_path=source,
            profile_path=profile,
            cache_index_sha256=receipt,  # type: ignore[arg-type]
        )


def test_wrong_well_formed_receipt_rejects_before_media_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, source, profile, _ = _fixture(tmp_path)

    def _forbidden_media_hash(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("wrong receipt reached media hashing")

    monkeypatch.setattr(cache_module, "sha256_file", _forbidden_media_hash)
    with pytest.raises(ThreeStockPreviewCacheError, match="receipt drift"):
        inspect_receipt_bound_three_stock_preview_cache(
            preview,
            input_path=source,
            profile_path=profile,
            cache_index_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_contract_sha256", "b" * 64),
        ("preview_width", 41),
        ("preview_pixels", 1201),
        ("look_amount", 0.5),
        ("claim_ceiling", "foreign authority"),
        ("rows", "mutate-row-hash"),
    ],
)
def test_index_metadata_or_expected_hash_mutation_rejects_against_receipt(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    preview, source, profile, receipt = _fixture(tmp_path)
    index_path = preview / CACHE_INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if field == "rows":
        index["rows"][1]["output_sha256"] = "b" * 64
    else:
        index[field] = value
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ThreeStockPreviewCacheError, match="receipt drift"):
        admit_receipt_bound_three_stock_preview_session(
            preview,
            input_path=source,
            profile_path=profile,
            cache_index_sha256=receipt,
        )


def test_bound_session_is_immutable_and_warm_lookup_has_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview, source, profile, receipt = _fixture(tmp_path)
    session = admit_receipt_bound_three_stock_preview_session(
        preview,
        input_path=source,
        profile_path=profile,
        cache_index_sha256=receipt,
    )
    snapshot = lookup_receipt_bound_three_stock_preview_session(session)
    assert snapshot.cache_index_sha256 == receipt
    assert [row.style_id for row in snapshot.preview.rows] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    with pytest.raises(FrozenInstanceError):
        snapshot.cache_index_sha256 = "b" * 64  # type: ignore[misc]

    def _forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("warm lookup attempted filesystem access")

    monkeypatch.setattr(Path, "read_bytes", _forbidden_read)
    monkeypatch.setattr(Path, "read_text", _forbidden_read)
    assert lookup_receipt_bound_three_stock_preview_session(session) is snapshot


def test_admitted_snapshot_survives_index_replacement_but_readmission_rejects(
    tmp_path: Path,
) -> None:
    preview, source, profile, receipt = _fixture(tmp_path)
    session = admit_receipt_bound_three_stock_preview_session(
        preview,
        input_path=source,
        profile_path=profile,
        cache_index_sha256=receipt,
    )
    original = lookup_receipt_bound_three_stock_preview_session(session)
    index_path = preview / CACHE_INDEX_NAME
    index_path.write_bytes(index_path.read_bytes() + b"\n")

    assert lookup_receipt_bound_three_stock_preview_session(session) is original
    assert original.preview.rows[1].payload == b"immutable-portra_400"
    with pytest.raises(ThreeStockPreviewCacheError, match="receipt drift"):
        admit_receipt_bound_three_stock_preview_session(
            preview,
            input_path=source,
            profile_path=profile,
            cache_index_sha256=receipt,
        )


@pytest.mark.parametrize("target", ["source", "profile", "preview"])
def test_bound_admission_retains_media_drift_rejection(
    tmp_path: Path, target: str
) -> None:
    preview, source, profile, receipt = _fixture(tmp_path)
    expected = target
    if target == "source":
        source.write_bytes(b"changed source")
        expected = "input drift"
    elif target == "profile":
        profile.write_text('{"profile":2}\n', encoding="utf-8")
        expected = "profile drift"
    else:
        (preview / "portra_400.preview.png").write_bytes(b"changed preview")
        expected = "output drift"

    with pytest.raises(ThreeStockPreviewCacheError, match=expected):
        admit_receipt_bound_three_stock_preview_session(
            preview,
            input_path=source,
            profile_path=profile,
            cache_index_sha256=receipt,
        )
