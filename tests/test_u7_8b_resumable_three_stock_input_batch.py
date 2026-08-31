from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference.render_contract import sha256_file
from src.inference.resumable_three_stock_input_batch import (
    ResumableThreeStockBatchError,
    render_resumable_three_stock_input_batch_to_directory,
)
from src.inference.three_stock_input_batch import INPUT_MANIFEST_SCHEMA

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


def _manifest(path: Path, count: int = 3) -> list[Path]:
    sources: list[Path] = []
    jobs = []
    for index in range(count):
        source = path.parent / f"source-{index:03d}.png"
        _source(source, index + 1)
        sources.append(source)
        jobs.append(
            {
                "job_id": f"job-{index:03d}",
                "input_path": str(source),
                "input_sha256": sha256_file(source),
            }
        )
    path.write_text(
        json.dumps({"schema_version": INPUT_MANIFEST_SCHEMA, "jobs": jobs}),
        encoding="utf-8",
    )
    return sources


def _render(
    manifest: Path,
    workspace: Path,
    destination: Path,
    *,
    maximum_new_jobs: int | None = None,
) -> dict:
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
        maximum_new_jobs=maximum_new_jobs,
    )


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_pause_resume_skips_completed_job_and_matches_clean_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"

    progress = _render(manifest, workspace, destination, maximum_new_jobs=1)
    assert progress["completed_job_count"] == 1
    assert progress["remaining_job_count"] == 2
    assert progress["destination_published"] is False
    assert workspace.is_dir()
    assert not destination.exists()
    first_hashes = _tree_hashes(workspace / "job-000")

    from src.inference import resumable_three_stock_input_batch as module

    original = module.render_three_stock_batch_to_directory
    calls: list[str] = []

    def record(input_path: Path, *args, **kwargs):
        calls.append(Path(input_path).name)
        return original(input_path, *args, **kwargs)

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", record)
    resumed = _render(manifest, workspace, destination)
    assert calls == ["source-001.png", "source-002.png"]
    assert not workspace.exists()
    assert destination.is_dir()
    assert _tree_hashes(destination / "job-000") == first_hashes
    resumed_hashes = _tree_hashes(destination)
    assert len(list(destination.glob("job-*/*.png"))) == 9
    assert len(list(destination.glob("job-*/*.recipe.json"))) == 9

    shutil.rmtree(destination)
    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", original)
    clean = _render(manifest, workspace, destination)
    assert clean == resumed
    assert _tree_hashes(destination) == resumed_hashes


def test_complete_child_with_stale_checkpoint_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    from src.inference import resumable_three_stock_input_batch as module

    original_write = module._write_checkpoint
    monkeypatch.setattr(module, "_write_checkpoint", lambda *_args, **_kwargs: None)
    progress = _render(manifest, workspace, destination, maximum_new_jobs=1)
    assert progress["completed_job_count"] == 1
    checkpoint = json.loads((workspace / "checkpoint.json").read_text("utf-8"))
    assert checkpoint["completed_jobs"] == []

    monkeypatch.setattr(module, "_write_checkpoint", original_write)
    original_render = module.render_three_stock_batch_to_directory
    calls = 0

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_render(*args, **kwargs)

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", count)
    result = _render(manifest, workspace, destination)
    assert result["job_count"] == 2
    assert calls == 1


def test_checkpoint_claiming_missing_child_rejects_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    _render(manifest, workspace, destination, maximum_new_jobs=1)
    shutil.rmtree(workspace / "job-000")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing claimed child must reject before render")

    monkeypatch.setattr(
        "src.inference.resumable_three_stock_input_batch.render_three_stock_batch_to_directory",
        forbidden,
    )
    with pytest.raises(ResumableThreeStockBatchError, match="missing child"):
        _render(manifest, workspace, destination)
    assert not destination.exists()


def test_tampered_completed_child_rejects_without_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    _render(manifest, workspace, destination, maximum_new_jobs=1)
    (workspace / "job-000" / "velvia_50.png").write_bytes(b"tampered")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered checkpoint must reject before render")

    monkeypatch.setattr(
        "src.inference.resumable_three_stock_input_batch.render_three_stock_batch_to_directory",
        forbidden,
    )
    with pytest.raises(ResumableThreeStockBatchError, match="artifact identity"):
        _render(manifest, workspace, destination)
    assert not destination.exists()


def test_crash_after_last_child_before_receipt_resumes_without_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    from src.inference import resumable_three_stock_input_batch as module

    original_final = module._final_receipt

    def interrupt(**_kwargs):
        raise RuntimeError("injected final-receipt interruption")

    monkeypatch.setattr(module, "_final_receipt", interrupt)
    with pytest.raises(RuntimeError, match="final-receipt interruption"):
        _render(manifest, workspace, destination)
    assert sorted(path.name for path in workspace.glob("job-*")) == [
        "job-000",
        "job-001",
    ]
    assert not (workspace / "batch.json").exists()
    assert not destination.exists()

    monkeypatch.setattr(module, "_final_receipt", original_final)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete checkpoint must not rerender")

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", forbidden)
    result = _render(manifest, workspace, destination)
    assert result["job_count"] == 2


def test_existing_complete_receipt_retries_late_publication_without_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=1)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    from src.inference import resumable_three_stock_input_batch as module

    original_rename = module.os.rename

    def late_foreign(source: str | os.PathLike, target: str | os.PathLike) -> None:
        if Path(source) == workspace and Path(target) == destination:
            destination.mkdir()
            (destination / "foreign.bin").write_bytes(b"foreign")
            raise FileExistsError("late foreign destination")
        original_rename(source, target)

    monkeypatch.setattr(module.os, "rename", late_foreign)
    with pytest.raises(FileExistsError, match="late foreign"):
        _render(manifest, workspace, destination)
    assert (destination / "foreign.bin").read_bytes() == b"foreign"
    assert (workspace / "batch.json").is_file()

    shutil.rmtree(destination)
    monkeypatch.setattr(module.os, "rename", original_rename)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete retry must not rerender")

    monkeypatch.setattr(module, "render_three_stock_batch_to_directory", forbidden)
    result = _render(manifest, workspace, destination)
    assert result["job_count"] == 1


def test_reserved_transient_is_removed_only_after_valid_resume(
    tmp_path: Path
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    _render(manifest, workspace, destination, maximum_new_jobs=1)
    transient = workspace / ".u7-8b-job-001-stale.candidate"
    transient.mkdir()
    (transient / "partial.bin").write_bytes(b"partial")
    result = _render(manifest, workspace, destination)
    assert result["job_count"] == 2
    assert not transient.exists()


def test_state_bound_stale_initialization_sibling_is_reconciled(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=1)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    from src.inference import resumable_three_stock_input_batch as module

    jobs = module._load_jobs(manifest)
    state = module._expected_state(
        jobs=jobs,
        manifest_path=manifest,
        output_directory=destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        look_amount=1.0,
        seed=31,
        tile_size=16,
        tile_workers=1,
        png_compression=0,
    )
    stale = tmp_path / (
        f".{workspace.name}.u7-8b-init-{state['workspace_id']}-"
        "0123456789abcdef0123456789abcdef"
    )
    stale.mkdir()
    module.atomic_write_json(stale / "resume.json", state)
    module.atomic_write_json(
        stale / "checkpoint.json",
        module._checkpoint_payload(state["workspace_id"], []),
    )

    result = _render(manifest, workspace, destination)
    assert result["job_count"] == 1
    assert not stale.exists()


@pytest.mark.parametrize("phase", ["empty", "resume", "atomic-temp"])
def test_early_state_bound_initialization_crash_is_reconciled(
    tmp_path: Path, phase: str
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=1)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    from src.inference import resumable_three_stock_input_batch as module

    jobs = module._load_jobs(manifest)
    state = module._expected_state(
        jobs=jobs,
        manifest_path=manifest,
        output_directory=destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        look_amount=1.0,
        seed=31,
        tile_size=16,
        tile_workers=1,
        png_compression=0,
    )
    stale = tmp_path / (
        f".{workspace.name}.u7-8b-init-{state['workspace_id']}-"
        "fedcba9876543210fedcba9876543210"
    )
    stale.mkdir()
    if phase == "resume":
        module.atomic_write_json(stale / "resume.json", state)
    elif phase == "atomic-temp":
        (stale / ".resume.json.12345.tmp").write_bytes(b"partial")

    result = _render(manifest, workspace, destination)
    assert result["job_count"] == 1
    assert not stale.exists()


def test_core_drift_rejects_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    _render(manifest, workspace, destination, maximum_new_jobs=1)
    from src.inference import resumable_three_stock_input_batch as module

    original_hashes = module._core_hashes

    def drift(root: Path) -> dict[str, str]:
        values = original_hashes(root)
        values[next(iter(values))] = "0" * 64
        return values

    monkeypatch.setattr(module, "_core_hashes", drift)
    monkeypatch.setattr(
        module,
        "render_three_stock_batch_to_directory",
        lambda *_args, **_kwargs: pytest.fail("core drift must reject before render"),
    )
    with pytest.raises(ResumableThreeStockBatchError, match="identity mismatch"):
        _render(manifest, workspace, destination)


def test_hardlinked_child_member_rejects(tmp_path: Path) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    _render(manifest, workspace, destination, maximum_new_jobs=1)
    recipe = workspace / "job-000" / "velvia_50.recipe.json"
    original = tmp_path / "original-recipe.json"
    original.write_bytes(recipe.read_bytes())
    recipe.unlink()
    os.link(original, recipe)
    try:
        with pytest.raises(ResumableThreeStockBatchError, match="regular single-link"):
            _render(manifest, workspace, destination)
    finally:
        recipe.unlink()


def test_child_manifest_extra_semantics_reject_before_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    _render(manifest, workspace, destination, maximum_new_jobs=1)
    child_manifest = workspace / "job-000" / "batch.json"
    payload = json.loads(child_manifest.read_text("utf-8"))
    payload["unbound_semantics"] = "forbidden"
    child_manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "src.inference.resumable_three_stock_input_batch.render_three_stock_batch_to_directory",
        lambda *_args, **_kwargs: pytest.fail("semantic drift must reject before render"),
    )
    with pytest.raises(ResumableThreeStockBatchError, match="manifest identity"):
        _render(manifest, workspace, destination)


def test_destination_aliasing_lease_rejects_without_creation(tmp_path: Path) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=1)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / f".{workspace.name}.u7-8b.lease"

    with pytest.raises(ResumableThreeStockBatchError, match="must be distinct"):
        _render(manifest, workspace, destination)
    assert not destination.exists()
    assert not workspace.exists()


def test_concurrent_lease_rejects_without_workspace_mutation(
    tmp_path: Path
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published"
    from src.inference import resumable_three_stock_input_batch as module

    jobs = module._load_jobs(manifest)
    state = module._expected_state(
        jobs=jobs,
        manifest_path=manifest,
        output_directory=destination,
        root=ROOT,
        profile_path=PROFILE,
        statistics_path=STATISTICS,
        guardrails_path=GUARDRAILS,
        look_amount=1.0,
        seed=31,
        tile_size=16,
        tile_workers=1,
        png_compression=0,
    )
    lease = workspace.with_name(f".{workspace.name}.u7-8b.lease")
    before = sorted(path.name for path in tmp_path.iterdir())
    with (
        module._exclusive_lease(lease, module._lease_payload(state)),
        pytest.raises(ResumableThreeStockBatchError, match="another writer"),
    ):
        _render(manifest, workspace, destination)
    after = sorted(path.name for path in tmp_path.iterdir())
    assert not workspace.exists()
    assert not destination.exists()
    assert set(after) - set(before) == {lease.name}


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("maximum_new_jobs", False, "maximum_new_jobs"),
        ("tile_size", False, "tile_size"),
        ("look_amount", True, "look_amount"),
    ],
)
def test_boolean_numeric_arguments_fail_closed(
    tmp_path: Path, keyword: str, value: object, message: str
) -> None:
    manifest = tmp_path / "jobs.json"
    _manifest(manifest, count=1)
    arguments = {
        "manifest_path": manifest,
        "workspace_directory": tmp_path / "resume-workspace",
        "output_directory": tmp_path / "published",
        "root": ROOT,
        "profile_path": PROFILE,
        "statistics_path": STATISTICS,
        "guardrails_path": GUARDRAILS,
        keyword: value,
    }
    with pytest.raises(ResumableThreeStockBatchError, match=message):
        render_resumable_three_stock_input_batch_to_directory(**arguments)
    assert not (tmp_path / "resume-workspace").exists()
    assert not (tmp_path / "published").exists()
