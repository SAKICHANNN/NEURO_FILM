from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.render_contract import sha256_file
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    render_three_stock_input_batch_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"

EXPECTED_OUTPUTS = {
    "ektar_100.png": "e7ca6e1b2c0fa0b0d75b7db4e0a63612d595ca4ee233034c023f36f580da41ca",
    "portra_400.png": "bfe057fd2cc83ea4732d71b79f8afc93c40c790d883ba2f2031fb32207d8574f",
    "velvia_50.png": "d34b4749515c0c1ab5f60fa2d196820fd96c4553ad29df93aab19fc99ef4d627",
}


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:24, :32]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3 + 11) % 256,
            (xx * 2 + yy * 13 + 17) % 256,
            (xx * 19 + yy * 5 + 23) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _manifest(path: Path, source: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": INPUT_MANIFEST_SCHEMA,
                "jobs": [
                    {
                        "job_id": "source",
                        "input_path": str(source),
                        "input_sha256": sha256_file(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _render(manifest: Path, destination: Path):
    return render_three_stock_input_batch_to_directory(
        manifest,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        tile_workers=1,
        png_compression=0,
    )


def test_replaced_outer_stage_and_foreign_payload_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    manifest = tmp_path / "jobs.json"
    destination = tmp_path / "published"
    _source(source)
    _manifest(manifest, source)

    from src.inference import three_stock_input_batch as module

    witness: dict[str, Path] = {}

    def replace_stage_then_fail(_input_path: Path, child_stage: Path, **_kwargs):
        stage = Path(child_stage).parent
        owned_backup = stage.with_name(f"{stage.name}.owned-backup")
        stage.rename(owned_backup)
        stage.mkdir()
        foreign = stage / "foreign.bin"
        foreign.write_bytes(b"foreign-owner")
        witness.update(stage=stage, owned_backup=owned_backup, foreign=foreign)
        raise RuntimeError("injected stage identity replacement")

    monkeypatch.setattr(
        module, "render_three_stock_batch_to_directory", replace_stage_then_fail
    )
    with pytest.raises(RuntimeError, match="stage identity replacement"):
        _render(manifest, destination)

    assert witness["stage"].is_dir()
    assert witness["foreign"].read_bytes() == b"foreign-owner"
    assert witness["owned_backup"].is_dir()
    assert not destination.exists()


def test_ordinary_failure_removes_still_owned_outer_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    manifest = tmp_path / "jobs.json"
    destination = tmp_path / "published"
    _source(source)
    _manifest(manifest, source)

    def fail_child(*_args, **_kwargs):
        raise RuntimeError("ordinary child failure")

    monkeypatch.setattr(
        "src.inference.three_stock_input_batch.render_three_stock_batch_to_directory",
        fail_child,
    )
    with pytest.raises(RuntimeError, match="ordinary child failure"):
        _render(manifest, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*.stage"))


def test_successful_product_pixels_remain_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    manifest = tmp_path / "jobs.json"
    destination = tmp_path / "published"
    _source(source)
    _manifest(manifest, source)

    receipt = _render(manifest, destination)
    assert receipt["schema_version"] == "neuro-film.three-stock-input-batch.v1"
    assert [row["style_id"] for row in receipt["jobs"][0]["rows"]] == [
        "velvia_50",
        "portra_400",
        "ektar_100",
    ]
    assert {
        path.name: sha256_file(path)
        for path in sorted((destination / "source").glob("*.png"))
    } == EXPECTED_OUTPUTS
