from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.inference.render_contract import sha256_file
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    ThreeStockInputBatchError,
    render_three_stock_input_batch_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:24, :32]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3 + offset * 11) % 256,
            (xx * 2 + yy * 13 + offset * 17) % 256,
            (xx * 19 + yy * 5 + offset * 23) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _manifest(path: Path, sources: list[tuple[str, Path]]) -> None:
    payload = {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "jobs": [
            {
                "job_id": job_id,
                "input_path": str(source),
                "input_sha256": sha256_file(source),
            }
            for job_id, source in sources
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _render(manifest: Path, destination: Path) -> dict:
    return render_three_stock_input_batch_to_directory(
        manifest,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


def test_forward_reverse_jobs_publish_exact_portable_receipt(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _source(first, 1)
    _source(second, 2)
    manifest = tmp_path / "jobs.json"
    destination = tmp_path / "published"

    _manifest(manifest, [("b-second", second), ("a-first", first)])
    forward = _render(manifest, destination)
    forward_bytes = (destination / "batch.json").read_bytes()
    assert str(tmp_path) not in forward_bytes.decode("utf-8")
    assert [row["job_id"] for row in forward["jobs"]] == ["a-first", "b-second"]
    assert len(list(destination.glob("*/*.png"))) == 6
    assert len(list(destination.glob("*/*.recipe.json"))) == 6
    for job in forward["jobs"]:
        assert not Path(job["child_manifest_path"]).is_absolute()
        for row in job["rows"]:
            assert not Path(row["output_path"]).is_absolute()
            assert not Path(row["recipe_path"]).is_absolute()
            recipe = json.loads((destination / row["recipe_path"]).read_text("utf-8"))
            assert (
                Path(recipe["output"]["path"])
                == (destination / row["output_path"]).resolve()
            )
            assert recipe["claim"]["output_label"] == "film-inspired"
            assert recipe["claim"]["evidence_grade"] == "look-approximation"
            decoded = cv2.imread(
                str(destination / row["output_path"]), cv2.IMREAD_UNCHANGED
            )
            assert decoded.dtype == np.uint16
            assert decoded.shape == (24, 32, 3)

    shutil.rmtree(destination)
    _manifest(manifest, [("a-first", first), ("b-second", second)])
    reverse = _render(manifest, destination)
    assert reverse == forward
    assert (destination / "batch.json").read_bytes() == forward_bytes


def test_relative_input_paths_resolve_from_manifest_parent(tmp_path: Path) -> None:
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    source = source_root / "source.png"
    _source(source, 3)
    manifest = tmp_path / "jobs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": INPUT_MANIFEST_SCHEMA,
                "jobs": [
                    {
                        "job_id": "relative",
                        "input_path": "inputs/source.png",
                        "input_sha256": sha256_file(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt = _render(manifest, tmp_path / "published")
    assert receipt["jobs"][0]["input_sha256"] == sha256_file(source)


def test_all_input_hashes_preflight_before_render_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _source(first, 1)
    _source(second, 2)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, [("a", first), ("b", second)])
    second.write_bytes(b"drift")

    def forbidden(*args, **kwargs):
        raise AssertionError("render must not run before complete preflight")

    monkeypatch.setattr(
        "src.inference.three_stock_input_batch.render_three_stock_batch_to_directory",
        forbidden,
    )
    destination = tmp_path / "published"
    with pytest.raises(ThreeStockInputBatchError, match="input hash drifted"):
        _render(manifest, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*.stage"))


def test_injected_child_failure_removes_owned_stage_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _source(first, 1)
    _source(second, 2)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, [("a", first), ("b", second)])

    from src.inference import three_stock_input_batch as module

    original = module.render_three_stock_batch_to_directory
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected child failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", fail_second)
    destination = tmp_path / "published"
    with pytest.raises(RuntimeError, match="injected child failure"):
        _render(manifest, destination)
    assert calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*.stage"))


def test_child_recipe_input_identity_drift_rejects_entire_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source, 1)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, [("source", source)])

    from src.inference import three_stock_input_batch as module

    original = module.render_three_stock_batch_to_directory

    def drift_recipe(*args, **kwargs):
        child = original(*args, **kwargs)
        child_root = Path(args[1])
        recipe_path = child_root / "velvia_50.recipe.json"
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe["input"]["sha256"] = "0" * 64
        recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
        return child

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", drift_recipe)
    destination = tmp_path / "published"
    with pytest.raises(ThreeStockInputBatchError, match="recipe input identity"):
        _render(manifest, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*.stage"))


def test_late_foreign_destination_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source, 1)
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, [("source", source)])
    destination = tmp_path / "published"

    def foreign_claim(stage: Path, target: Path) -> None:
        target.mkdir()
        (target / "foreign.bin").write_bytes(b"foreign")
        raise FileExistsError("foreign destination")

    monkeypatch.setattr(
        "src.inference.three_stock_input_batch._publish_no_replace", foreign_claim
    )
    with pytest.raises(FileExistsError, match="foreign destination"):
        _render(manifest, destination)
    assert (destination / "foreign.bin").read_bytes() == b"foreign"
    assert list(destination.iterdir()) == [destination / "foreign.bin"]
    assert not list(tmp_path.glob(".published.*.stage"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": INPUT_MANIFEST_SCHEMA, "jobs": []}, "1..100"),
        (
            {
                "schema_version": INPUT_MANIFEST_SCHEMA,
                "jobs": [
                    {"job_id": "../escape", "input_path": "x", "input_sha256": "0" * 64}
                ],
            },
            "id is invalid",
        ),
    ],
)
def test_invalid_manifests_fail_before_publication(
    tmp_path: Path, payload: dict, message: str
) -> None:
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ThreeStockInputBatchError, match=message):
        _render(manifest, tmp_path / "published")
    assert not list(tmp_path.glob(".published.*.stage"))
