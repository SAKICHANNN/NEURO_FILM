from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.inference.desktop_single_look_batch_recovery as recovery_module
import src.inference.product_desktop as desktop_module
from src.inference.desktop_single_look_batch_recovery import (
    DesktopBatchProgressReceipt,
    DesktopSingleLookBatchRecoveryError,
    export_resumable_desktop_single_look_batch,
)
from src.inference.product_desktop import ProductDesktopWorkflow
from src.inference.render_contract import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:32, :48]
    rgb = np.stack(
        (
            (xx * 13 + yy * 7 + offset * 29) % 251,
            (xx * 3 + yy * 17 + offset * 47 + 19) % 251,
            (xx * 11 + yy * 5 + offset * 61 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _sources(tmp_path: Path, count: int = 3) -> tuple[Path, ...]:
    rows = []
    for index in range(count):
        source = tmp_path / f"u7-12d-{index:02d}.png"
        _source(source, index)
        rows.append(source)
    return tuple(rows)


def _workflow(tmp_path: Path, **kwargs: object) -> ProductDesktopWorkflow:
    scratch = tmp_path / f"scratch-{len(list(tmp_path.glob('scratch-*'))):02d}"
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=3_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
        **kwargs,  # type: ignore[arg-type]
    )


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _pause(
    tmp_path: Path,
    sources: tuple[Path, ...],
    workspace: Path,
    destination: Path,
    *,
    maximum_new_jobs: int = 1,
) -> DesktopBatchProgressReceipt:
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    result = export_resumable_desktop_single_look_batch(
        workflow,
        bound,
        "portra_400",
        workspace,
        destination,
        maximum_new_jobs=maximum_new_jobs,
    )
    assert isinstance(result, DesktopBatchProgressReceipt)
    assert workflow.close()
    return result


def test_contract_freezes_restart_and_flat_publication_boundaries() -> None:
    contract = json.loads(
        (ROOT / "configs/u7_12d_desktop_single_look_batch_recovery_v1.json").read_text(
            "utf-8"
        )
    )
    assert contract["checkpoint"]["completed_children_fully_revalidated_before_reuse"]
    assert contract["publication"]["aggregate_receipt_matches_unchanged_u7_11a"]
    assert contract["publication"]["operation"].startswith("same-parent Windows")
    assert "not hard-power-loss durability" in contract["claim_ceiling"]
    assert "calibrated stock response" in contract["claim_ceiling"]
    assert "physical-film reproduction" in contract["claim_ceiling"]


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_fresh_process_resume_skips_completed_and_matches_u7_11a(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    paused = _pause(tmp_path, sources, workspace, destination)
    assert paused.completed_job_count == 1
    assert paused.remaining_job_count == 2
    assert workspace.is_dir()
    assert not destination.exists()
    completed_bytes = {
        name: payload
        for name, payload in _tree_bytes(workspace / "0001").items()
        if name != "child.json"
    }

    calls: list[str] = []

    def record(command, cwd, environment):  # type: ignore[no-untyped-def]
        calls.append(Path(command[3]).name)
        return desktop_module._run_command(command, cwd, environment)

    resumed_workflow = _workflow(tmp_path, command_runner=record)
    _, rebound = resumed_workflow.render_batch_previews(sources, 0.625)
    resumed = export_resumable_desktop_single_look_batch(
        resumed_workflow,
        rebound,
        "portra_400",
        workspace,
        destination,
    )
    assert calls == [rebound[1].basename, rebound[2].basename]
    assert resumed.job_count == 3
    assert not workspace.exists()
    assert not (tmp_path / ".resume-workspace.u7-12d.lease").exists()
    assert completed_bytes == {
        name: payload
        for name, payload in _tree_bytes(destination).items()
        if name in completed_bytes
    }
    recovered_bytes = _tree_bytes(destination)
    recovered_receipt = resumed.receipt
    assert resumed_workflow.close()

    shutil.rmtree(destination)
    legacy_workflow = _workflow(tmp_path)
    _, legacy_bound = legacy_workflow.render_batch_previews(sources, 0.625)
    legacy = legacy_workflow.export_batch(legacy_bound, "portra_400", destination)
    assert legacy.receipt == recovered_receipt
    assert _tree_bytes(destination) == recovered_bytes
    assert legacy_workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_cancel_preserves_verified_children_and_resume_renders_remaining(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    cancel = threading.Event()

    def progress(done: int, _total: int, _name: str) -> None:
        if done == 1:
            cancel.set()

    paused = export_resumable_desktop_single_look_batch(
        workflow,
        bound,
        "portra_400",
        workspace,
        destination,
        cancel_event=cancel,
        progress=progress,
    )
    assert isinstance(paused, DesktopBatchProgressReceipt)
    assert paused.completed_job_count == 1
    assert not destination.exists()
    assert workflow.close()

    calls = 0

    def record(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return desktop_module._run_command(command, cwd, environment)

    resumed_workflow = _workflow(tmp_path, command_runner=record)
    _, rebound = resumed_workflow.render_batch_previews(sources, 0.625)
    result = export_resumable_desktop_single_look_batch(
        resumed_workflow,
        rebound,
        "portra_400",
        workspace,
        destination,
    )
    assert result.job_count == 3
    assert calls == 2
    assert resumed_workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_stale_checkpoint_reconciles_without_rerendering_complete_child(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    _pause(tmp_path, sources, workspace, destination)
    checkpoint_path = workspace / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
    checkpoint["completed_jobs"] = []
    checkpoint_path.write_text(
        json.dumps(
            checkpoint, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    calls = 0

    def record(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return desktop_module._run_command(command, cwd, environment)

    workflow = _workflow(tmp_path, command_runner=record)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    result = export_resumable_desktop_single_look_batch(
        workflow, bound, "portra_400", workspace, destination
    )
    assert result.job_count == 3
    assert calls == 2
    assert workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_tampered_or_missing_claimed_child_rejects_before_render(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    _pause(tmp_path, sources, workspace, destination)
    image = next((workspace / "0001").glob("*.png"))
    image.write_bytes(image.read_bytes() + b"tampered")

    def forbidden(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("invalid completed child must reject before render")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    with pytest.raises(DesktopSingleLookBatchRecoveryError, match="semantics drifted"):
        export_resumable_desktop_single_look_batch(
            workflow, bound, "portra_400", workspace, destination
        )
    assert not destination.exists()
    assert workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_missing_claimed_child_rejects_before_render(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    _pause(tmp_path, sources, workspace, destination)
    shutil.rmtree(workspace / "0001")

    def forbidden(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("missing claimed child must reject before render")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    with pytest.raises(DesktopSingleLookBatchRecoveryError, match="missing child"):
        export_resumable_desktop_single_look_batch(
            workflow, bound, "portra_400", workspace, destination
        )
    assert not destination.exists()
    assert workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_child_failure_preserves_completed_child_and_resume_skips_it(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    calls = 0

    def fail_second(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            return subprocess.CompletedProcess(command, 23, "", "injected failure")
        return desktop_module._run_command(command, cwd, environment)

    workflow = _workflow(tmp_path, command_runner=fail_second)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    with pytest.raises(DesktopSingleLookBatchRecoveryError, match="injected failure"):
        export_resumable_desktop_single_look_batch(
            workflow, bound, "portra_400", workspace, destination
        )
    assert (workspace / "0001").is_dir()
    assert not (workspace / "0002").exists()
    assert not list(workspace.glob("*.candidate"))
    assert workflow.close()

    resume_calls = 0

    def record(command, cwd, environment):  # type: ignore[no-untyped-def]
        nonlocal resume_calls
        resume_calls += 1
        return desktop_module._run_command(command, cwd, environment)

    resumed_workflow = _workflow(tmp_path, command_runner=record)
    _, rebound = resumed_workflow.render_batch_previews(sources, 0.625)
    result = export_resumable_desktop_single_look_batch(
        resumed_workflow,
        rebound,
        "portra_400",
        workspace,
        destination,
    )
    assert result.job_count == 3
    assert resume_calls == 2
    assert resumed_workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_exact_reserved_transient_is_reconciled_but_foreign_member_is_not(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    _pause(tmp_path, sources, workspace, destination)
    state = json.loads((workspace / "resume.json").read_text("utf-8"))
    transient = workspace / (
        f".u7-12d-{state['workspace_id']}-0002-"
        "0123456789abcdef0123456789abcdef.candidate"
    )
    transient.mkdir()

    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    result = export_resumable_desktop_single_look_batch(
        workflow, bound, "portra_400", workspace, destination
    )
    assert result.job_count == 3
    assert not transient.exists()
    assert workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_unexpected_workspace_member_is_preserved_and_rejected(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    _pause(tmp_path, sources, workspace, destination)
    foreign = workspace / "foreign.bin"
    foreign.write_bytes(b"foreign")

    def forbidden(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("foreign workspace member must reject before render")

    workflow = _workflow(tmp_path, command_runner=forbidden)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    with pytest.raises(DesktopSingleLookBatchRecoveryError, match="unexpected member"):
        export_resumable_desktop_single_look_batch(
            workflow, bound, "portra_400", workspace, destination
        )
    assert foreign.read_bytes() == b"foreign"
    assert not destination.exists()
    assert workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_second_writer_rejects_while_exact_lease_is_locked(tmp_path: Path) -> None:
    lease = tmp_path / "recovery.lease"
    expected = b'{"schema_version":"test","workspace_id":"bound"}\n'
    with (
        recovery_module._exclusive_lease(lease, expected),
        pytest.raises(
            DesktopSingleLookBatchRecoveryError, match="held by another writer"
        ),
        recovery_module._exclusive_lease(lease, expected),
    ):
        pytest.fail("second writer unexpectedly acquired lease")


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_late_foreign_destination_is_preserved_and_retry_does_not_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    original_rename = recovery_module.os.rename

    def late_foreign(source: str | os.PathLike, target: str | os.PathLike) -> None:
        if ".u7-12d-publish-" in Path(source).name and Path(target) == destination:
            destination.mkdir()
            (destination / "foreign.bin").write_bytes(b"foreign")
            raise FileExistsError("late foreign destination")
        original_rename(source, target)

    monkeypatch.setattr(recovery_module.os, "rename", late_foreign)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    with pytest.raises(FileExistsError, match="late foreign"):
        export_resumable_desktop_single_look_batch(
            workflow, bound, "portra_400", workspace, destination
        )
    assert (destination / "foreign.bin").read_bytes() == b"foreign"
    assert sorted(path.name for path in workspace.glob("[0-9][0-9][0-9][0-9]")) == [
        "0001",
        "0002",
    ]
    assert workflow.close()

    shutil.rmtree(destination)
    monkeypatch.setattr(recovery_module.os, "rename", original_rename)

    def forbidden(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("complete retry must not rerender")

    retry_workflow = _workflow(tmp_path, command_runner=forbidden)
    _, rebound = retry_workflow.render_batch_previews(sources, 0.625)
    result = export_resumable_desktop_single_look_batch(
        retry_workflow,
        rebound,
        "portra_400",
        workspace,
        destination,
    )
    assert result.job_count == 2
    assert retry_workflow.close()


@pytest.mark.skipif(
    os.name != "nt", reason="formal recovery publication is Windows-only"
)
def test_post_publication_cleanup_failure_is_monotonic_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path, count=2)
    workspace = tmp_path / "resume-workspace"
    destination = tmp_path / "published-batch"
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    monkeypatch.setattr(
        recovery_module, "_cleanup_verified_workspace", lambda *_args, **_kwargs: False
    )
    result = export_resumable_desktop_single_look_batch(
        workflow, bound, "portra_400", workspace, destination
    )
    assert result.receipt_path.is_file()
    assert sha256_file(result.receipt_path) == result.receipt_sha256
    assert workspace.is_dir()
    assert not (tmp_path / ".resume-workspace.u7-12d.lease").exists()
    assert workflow.close()


def test_core_identity_binds_implementation_and_parent_receipt_sources(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, count=2)
    workflow = _workflow(tmp_path)
    _, bound = workflow.render_batch_previews(sources, 0.625)
    state = recovery_module._expected_state(
        workflow=workflow,
        rows=bound,
        style_id="portra_400",
        destination=tmp_path / "published-batch",
    )
    assert state["core_sha256"] == {
        relative: sha256_file(ROOT / relative)
        for relative in recovery_module._CORE_PATHS
    }
    assert {
        "src/inference/desktop_single_look_batch_recovery.py",
        "src/inference/product_desktop.py",
        "src/inference/render_contract.py",
    } <= set(state["core_sha256"])
    assert workflow.close()
