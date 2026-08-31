from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.render_contract import sha256_file
from src.inference.resumable_three_stock_input_batch import (
    ResumableThreeStockBatchError,
    render_resumable_three_stock_input_batch_to_directory,
)
from src.inference.three_stock_batch import (
    ThreeStockBatchError,
    render_three_stock_batch_to_directory,
)
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    ThreeStockInputBatchError,
    render_three_stock_input_batch_to_directory,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
START_COMMIT = "1" * 40
DRIFT_COMMIT = "2" * 40


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
    Image.fromarray(rgb, mode="RGB").save(path, compress_level=0)


def _manifest(path: Path, count: int = 2) -> list[Path]:
    sources: list[Path] = []
    jobs: list[dict[str, str]] = []
    for index in range(count):
        source = path.parent / f"source-{index}.png"
        _source(source, index)
        sources.append(source)
        jobs.append(
            {
                "job_id": f"job-{index}",
                "input_path": source.name,
                "input_sha256": sha256_file(source),
            }
        )
    path.write_text(
        json.dumps({"schema_version": INPUT_MANIFEST_SCHEMA, "jobs": jobs}),
        encoding="utf-8",
    )
    return sources


def _input_batch(manifest: Path, destination: Path) -> dict:
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


def _resumable(manifest: Path, workspace: Path, destination: Path) -> dict:
    return render_resumable_three_stock_input_batch_to_directory(
        manifest,
        workspace,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )


def _recipe_commits(directory: Path) -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["software"]["commit"]
        for path in directory.rglob("*.recipe.json")
    }


def test_child_explicit_snapshot_is_used_without_live_head_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit commit must not resolve live HEAD")

    monkeypatch.setattr(
        "src.inference.three_stock_batch.subprocess.check_output", forbidden
    )
    destination = tmp_path / "published"
    render_three_stock_batch_to_directory(
        source,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
        software_commit=START_COMMIT,
    )
    assert _recipe_commits(destination) == {START_COMMIT}


@pytest.mark.parametrize("invalid", [True, 1, "A" * 40, "1" * 39, "g" * 40, "1" * 41])
def test_invalid_child_snapshot_rejects_before_decode_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    source = tmp_path / "source.png"
    _source(source, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid commit must reject before decode")

    monkeypatch.setattr("src.inference.three_stock_batch.load_working_image", forbidden)
    destination = tmp_path / "published"
    with pytest.raises(ThreeStockBatchError, match="lowercase full 40-hex"):
        render_three_stock_batch_to_directory(
            source,
            destination,
            root=ROOT,
            profile_path=PROFILE,
            statistics_path=STATISTICS,
            guardrails_path=GUARDRAILS,
            software_commit=invalid,  # type: ignore[arg-type]
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*.stage"))


def test_direct_child_default_resolves_live_head_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _source(source, 1)
    calls = 0

    def head(*args, **kwargs) -> str:
        nonlocal calls
        calls += 1
        return START_COMMIT + "\n"

    monkeypatch.setattr("src.inference.three_stock_batch.subprocess.check_output", head)
    destination = tmp_path / "published"
    render_three_stock_batch_to_directory(
        source,
        destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        tile_size=16,
        png_compression=0,
    )
    assert calls == 1
    assert _recipe_commits(destination) == {START_COMMIT}


def test_input_transaction_passes_one_snapshot_to_every_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=3)
    from src.inference import three_stock_input_batch as module

    resolved: list[str] = []

    def fixed(root: Path) -> str:
        resolved.append(str(root))
        return START_COMMIT

    observed: list[str | None] = []
    original = module.render_three_stock_batch_to_directory

    def record(*args, **kwargs):
        observed.append(kwargs.get("software_commit"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_software_commit", fixed)
    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", record)
    destination = tmp_path / "published"
    _input_batch(manifest, destination)
    assert len(resolved) == 2
    assert observed == [START_COMMIT] * 3
    assert _recipe_commits(destination) == {START_COMMIT}


def test_input_transaction_head_drift_rejects_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=3)
    from src.inference import three_stock_input_batch as module

    commits = iter((START_COMMIT, DRIFT_COMMIT))
    monkeypatch.setattr(module, "_software_commit", lambda root: next(commits))
    destination = tmp_path / "published"
    with pytest.raises(ThreeStockInputBatchError, match="drifted before publication"):
        _input_batch(manifest, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".published.*.stage"))


def test_resumable_passes_state_snapshot_but_final_head_drift_still_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=1)
    from src.inference import resumable_three_stock_input_batch as module

    commits = iter((START_COMMIT, DRIFT_COMMIT))
    monkeypatch.setattr(module, "_software_commit", lambda root: next(commits))
    observed: list[str | None] = []
    original = module.render_three_stock_batch_to_directory

    def record(*args, **kwargs):
        observed.append(kwargs.get("software_commit"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", record)
    workspace = tmp_path / "workspace"
    destination = tmp_path / "published"
    with pytest.raises(
        ResumableThreeStockBatchError,
        match="input, configuration or core identity drifted",
    ):
        _resumable(manifest, workspace, destination)
    assert observed == [START_COMMIT]
    assert _recipe_commits(workspace) == {START_COMMIT}
    assert workspace.is_dir()
    assert not destination.exists()
