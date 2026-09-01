"""Restart-safe transaction for one selected desktop Look Approximation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

try:  # pragma: no cover - keeps import diagnostics useful off Windows.
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]

from src.film_physics.create_only_file import (
    PublishedFileIdentity,
    remove_if_published,
)

from .product_desktop import (
    PRODUCT_LOOKS,
    DesktopBatchInput,
    DesktopBatchReceipt,
    ProductDesktopError,
    ProductDesktopWorkflow,
    _batch_directory_rename_supported,
    _batch_input_matches,
    _bind_file_after_directory_rename,
    _canonical_sha256,
    _cleanup_bound_stage,
    _directory_matches,
    _DirectorySeal,
    _file_matches,
    _FileSeal,
    _normalized_path,
    _replace_owned_json,
    _safe_output_stem,
    _seal_directory,
    _seal_file,
    _source_commit,
    _valid_windows_component,
    _write_bound_json,
)
from .render_contract import (
    atomic_write_json,
    sha256_file,
    validate_render_recipe,
    verify_render_recipe_files,
    verify_render_recipe_inputs,
)

WORKSPACE_SCHEMA = "kmcfm.desktop-single-look-recovery-workspace.v1"
CHECKPOINT_SCHEMA = "kmcfm.desktop-single-look-recovery-checkpoint.v1"
CHILD_SCHEMA = "kmcfm.desktop-single-look-recovery-child.v1"
PROGRESS_SCHEMA = "kmcfm.desktop-single-look-recovery-progress.v1"
LEASE_SCHEMA = "kmcfm.desktop-single-look-recovery-lease.v1"

_LOOK_IDS = tuple(row["style_id"] for row in PRODUCT_LOOKS)
_RESERVED_PREFIX = ".u7-12d-"
_CORE_PATHS = (
    "src/inference/desktop_single_look_batch_recovery.py",
    "src/inference/product_desktop.py",
    "src/inference/render_contract.py",
    "scripts/render_film.py",
    "configs/render_profiles/safe_rich_product_v1.json",
    "configs/film_color_stats.json",
    "configs/color_guardrails.json",
)
ProgressCallback = Callable[[int, int, str], None]


class DesktopSingleLookBatchRecoveryError(ProductDesktopError):
    """Reject an unsafe or inconsistent desktop recovery workspace."""


@dataclass(frozen=True)
class DesktopBatchProgressReceipt:
    """Verified resumable progress with no final destination publication."""

    workspace_id: str
    style_id: str
    look_amount: float
    workspace_directory: Path
    output_directory: Path
    completed_job_count: int
    remaining_job_count: int
    reused_job_count: int
    newly_completed_job_count: int
    receipt: dict[str, Any]


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery JSON contains duplicate keys"
            )
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_single_link(path, label)
    try:
        value = json.loads(path.read_text("utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DesktopSingleLookBatchRecoveryError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise DesktopSingleLookBatchRecoveryError(f"{label} must be an object")
    return value


def _is_reparse(path: Path, details: os.stat_result | None = None) -> bool:
    try:
        current = path.lstat() if details is None else details
    except FileNotFoundError:
        return False
    flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return path.is_symlink() or bool(
        int(getattr(current, "st_file_attributes", 0)) & flag
    )


def _require_directory(path: Path, label: str) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise DesktopSingleLookBatchRecoveryError(f"{label} is missing") from exc
    if (
        not stat.S_ISDIR(details.st_mode)
        or _is_reparse(path, details)
        or int(getattr(details, "st_nlink", 1)) != 1
    ):
        raise DesktopSingleLookBatchRecoveryError(
            f"{label} must be a normal single-link directory"
        )


def _require_regular_single_link(path: Path, label: str) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise DesktopSingleLookBatchRecoveryError(f"{label} is missing") from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or _is_reparse(path, details)
        or int(getattr(details, "st_nlink", 1)) != 1
    ):
        raise DesktopSingleLookBatchRecoveryError(
            f"{label} must be a normal single-link file"
        )


def _path_binding(path: Path) -> str:
    return hashlib.sha256(_normalized_path(path).encode("utf-8")).hexdigest()


def _core_hashes(root: Path) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in _CORE_PATHS}


def _preview_authority(workflow: ProductDesktopWorkflow) -> dict[str, Any]:
    state = workflow.preview_state
    if state is None:
        raise DesktopSingleLookBatchRecoveryError(
            "render previews before resumable export"
        )
    rows = state.manifest.get("rows")
    if not isinstance(rows, list):
        raise DesktopSingleLookBatchRecoveryError("preview manifest rows are invalid")
    preview_rows: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise DesktopSingleLookBatchRecoveryError("preview manifest row is invalid")
        style_id = row.get("style_id")
        output_sha256 = row.get("output_sha256")
        if not isinstance(style_id, str) or not isinstance(output_sha256, str):
            raise DesktopSingleLookBatchRecoveryError(
                "preview manifest identity is invalid"
            )
        preview_rows.append({"style_id": style_id, "output_sha256": output_sha256})
    session = []
    for seal in state.session_bindings:
        try:
            relative = seal.identity.path.relative_to(workflow.root).as_posix()
        except ValueError as exc:
            raise DesktopSingleLookBatchRecoveryError(
                "preview session binding is outside the project root"
            ) from exc
        session.append({"path": relative, "sha256": seal.sha256})
    identity = {
        "representative_input_sha256": state.input_sha256,
        "look_amount": state.look_amount,
        "source_commit": state.source_commit,
        "preview_width": state.manifest.get("preview_width"),
        "preview_height": state.manifest.get("preview_height"),
        "preview_rows": sorted(preview_rows, key=lambda row: row["style_id"]),
        "session_bindings": sorted(session, key=lambda row: row["path"]),
    }
    return {"preview_authority_sha256": _canonical_sha256(identity), **identity}


def _jobs(rows: Sequence[DesktopBatchInput]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": f"{index:04d}",
            "input_basename": row.basename,
            "input_path_binding_sha256": _path_binding(row.path),
            "input_sha256": row.sha256,
            "input_size": row.size,
            "input_device": row.device,
            "input_inode": row.inode,
        }
        for index, row in enumerate(rows, 1)
    ]


def _expected_state(
    *,
    workflow: ProductDesktopWorkflow,
    rows: Sequence[DesktopBatchInput],
    style_id: str,
    destination: Path,
) -> dict[str, Any]:
    preview = workflow.preview_state
    if preview is None:
        raise DesktopSingleLookBatchRecoveryError(
            "render previews before resumable export"
        )
    identity = {
        "schema_version": WORKSPACE_SCHEMA,
        "destination_binding_sha256": _path_binding(destination),
        "style_id": style_id,
        "look_amount": preview.look_amount,
        "source_commit": preview.source_commit,
        "output_format": "PNG",
        "output_bit_depth": 16,
        "jobs": _jobs(rows),
        "preview_authority": _preview_authority(workflow),
        "core_sha256": _core_hashes(workflow.root),
        "claim": {
            "output_label": "film-inspired / Look Approximation",
            "evidence_grade": "look-approximation",
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
        },
    }
    return {"workspace_id": _canonical_sha256(identity), **identity}


def _checkpoint_payload(workspace_id: str, rows: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "workspace_id": workspace_id,
        "completed_jobs": [
            {"job_id": job_id, "child_receipt_sha256": rows[job_id]}
            for job_id in sorted(rows)
        ],
    }


def _lease_payload(state: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(
        {"schema_version": LEASE_SCHEMA, "workspace_id": state["workspace_id"]}
    )


@contextmanager
def _exclusive_lease(path: Path, expected: bytes) -> Iterator[BinaryIO]:
    if os.name != "nt" or msvcrt is None:
        raise DesktopSingleLookBatchRecoveryError(
            "desktop recovery publication requires Windows"
        )
    if not _lexists(path):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as handle:
                if handle.write(expected) != len(expected):
                    raise OSError("short recovery lease write")
                handle.flush()
                os.fsync(handle.fileno())
    _require_regular_single_link(path, "recovery lease")
    handle = path.open("r+b", buffering=0)
    try:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery workspace is held by another writer"
            ) from exc
        try:
            handle.seek(0)
            if handle.read() != expected:
                raise DesktopSingleLookBatchRecoveryError(
                    "recovery lease identity mismatch"
                )
            yield handle
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _create_workspace(workspace: Path, state: Mapping[str, Any]) -> None:
    stage = workspace.with_name(
        f".{workspace.name}.u7-12d-init-{state['workspace_id']}-{uuid.uuid4().hex}"
    )
    stage.mkdir()
    seal = _seal_directory(stage)
    files: list[_FileSeal] = []
    try:
        files.append(_write_bound_json(stage / "resume.json", state))
        files.append(
            _write_bound_json(
                stage / "checkpoint.json",
                _checkpoint_payload(str(state["workspace_id"]), {}),
            )
        )
        os.rename(stage, workspace)
    except BaseException:
        _cleanup_bound_stage(seal, files)
        raise


def _validate_workspace_state(workspace: Path, state: Mapping[str, Any]) -> None:
    _require_directory(workspace, "recovery workspace")
    if _load_json(workspace / "resume.json", "recovery state") != state:
        raise DesktopSingleLookBatchRecoveryError(
            "recovery workspace identity mismatch"
        )


def _cleanup_reserved_transients(
    workspace: Path,
    *,
    workspace_id: str,
    jobs: Sequence[Mapping[str, Any]],
    rows: Sequence[DesktopBatchInput],
    style_id: str,
) -> None:
    """Remove only incomplete child stages attributable to this workspace."""

    job_map = {
        str(job["job_id"]): (job, source)
        for job, source in zip(jobs, rows, strict=True)
    }
    pattern = re.compile(
        rf"^{re.escape(_RESERVED_PREFIX + workspace_id)}-"
        r"(?P<job_id>[0-9]{4})-[0-9a-f]{32}\.candidate$"
    )
    for entry in tuple(workspace.iterdir()):
        match = pattern.fullmatch(entry.name)
        if match is None:
            continue
        job_id = match.group("job_id")
        if job_id not in job_map:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery transient names an unknown job"
            )
        job, source = job_map[job_id]
        _require_directory(entry, "recovery transient")
        image_name, recipe_name = _output_names(job, source, style_id)
        allowed = {image_name, recipe_name, "child.json"}
        if not {member.name for member in entry.iterdir()} <= allowed:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery transient contains an unexpected member"
            )
        directory = _seal_directory(entry)
        files: list[_FileSeal] = []
        for member in entry.iterdir():
            _require_regular_single_link(member, "recovery transient member")
            files.append(_seal_file(member))
        _cleanup_bound_stage(directory, files)
        if _lexists(entry):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery transient could not be removed safely"
            )


def _output_names(
    job: Mapping[str, Any], source: DesktopBatchInput, style_id: str
) -> tuple[str, str]:
    stem = f"{job['job_id']}-{_safe_output_stem(source.path)}-{style_id}"
    return f"{stem}.png", f"{stem}.recipe.json"


def _validate_child(
    child: Path,
    *,
    workflow: ProductDesktopWorkflow,
    source: DesktopBatchInput,
    job: Mapping[str, Any],
    style_id: str,
    destination: Path,
) -> tuple[dict[str, Any], str]:
    _require_directory(child, f"recovery child {job['job_id']}")
    image_name, recipe_name = _output_names(job, source, style_id)
    expected = {image_name, recipe_name, "child.json"}
    if {entry.name for entry in child.iterdir()} != expected:
        raise DesktopSingleLookBatchRecoveryError(
            f"recovery child {job['job_id']} member set drifted"
        )
    image_path = child / image_name
    recipe_path = child / recipe_name
    child_path = child / "child.json"
    for path, label in (
        (image_path, "child image"),
        (recipe_path, "child recipe"),
        (child_path, "child receipt"),
    ):
        _require_regular_single_link(path, label)
    image_sha = sha256_file(image_path)
    recipe_sha = sha256_file(recipe_path)
    recipe = _load_json(recipe_path, "child recipe")
    try:
        validate_render_recipe(recipe)
        verify_render_recipe_inputs(
            recipe,
            profile_path=workflow.root
            / "configs/render_profiles/safe_rich_product_v1.json",
            root=workflow.root,
        )
    except Exception as exc:
        raise DesktopSingleLookBatchRecoveryError(
            f"recovery child {job['job_id']} recipe is invalid"
        ) from exc
    final_image = (destination / image_name).resolve(strict=False)
    preview = workflow.preview_state
    if preview is None:
        raise DesktopSingleLookBatchRecoveryError("preview authority disappeared")
    if (
        recipe["input"]["path"] != str(source.path)
        or recipe["input"]["sha256"] != source.sha256
        or recipe["render"]["style"] != style_id
        or float(recipe["render"]["look_amount"]) != preview.look_amount
        or recipe["output"]["format"] != "PNG"
        or recipe["output"]["bit_depth"] != 16
        or recipe["output"]["path"] != str(final_image)
        or recipe["output"]["sha256"] != image_sha
        or str(recipe["software"]["commit"]).lower() != preview.source_commit
        or recipe["claim"]["evidence_grade"] != "look-approximation"
        or recipe["claim"].get("calibrated_reference_allowed") is not False
    ):
        raise DesktopSingleLookBatchRecoveryError(
            f"recovery child {job['job_id']} semantics drifted"
        )
    expected_receipt = {
        "schema_version": CHILD_SCHEMA,
        "job_id": job["job_id"],
        "input_basename": source.basename,
        "input_sha256": source.sha256,
        "output_path": PurePosixPath(image_name).as_posix(),
        "output_sha256": image_sha,
        "recipe_path": PurePosixPath(recipe_name).as_posix(),
        "recipe_sha256": recipe_sha,
    }
    receipt = _load_json(child_path, "child receipt")
    if receipt != expected_receipt:
        raise DesktopSingleLookBatchRecoveryError(
            f"recovery child {job['job_id']} receipt drifted"
        )
    return expected_receipt, sha256_file(child_path)


def _checkpoint_rows(workspace: Path, workspace_id: str) -> dict[str, str]:
    payload = _load_json(workspace / "checkpoint.json", "recovery checkpoint")
    if set(payload) != {"schema_version", "workspace_id", "completed_jobs"}:
        raise DesktopSingleLookBatchRecoveryError("recovery checkpoint keys mismatch")
    if (
        payload["schema_version"] != CHECKPOINT_SCHEMA
        or payload["workspace_id"] != workspace_id
        or not isinstance(payload["completed_jobs"], list)
    ):
        raise DesktopSingleLookBatchRecoveryError(
            "recovery checkpoint identity mismatch"
        )
    rows: dict[str, str] = {}
    for row in payload["completed_jobs"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"job_id", "child_receipt_sha256"}
            or not isinstance(row["job_id"], str)
            or not isinstance(row["child_receipt_sha256"], str)
            or row["job_id"] in rows
        ):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery checkpoint row is invalid"
            )
        rows[row["job_id"]] = row["child_receipt_sha256"]
    if list(rows) != sorted(rows):
        raise DesktopSingleLookBatchRecoveryError(
            "recovery checkpoint order is invalid"
        )
    return rows


def _inspect_workspace(
    workspace: Path,
    *,
    workflow: ProductDesktopWorkflow,
    rows: Sequence[DesktopBatchInput],
    jobs: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    style_id: str,
    destination: Path,
) -> tuple[dict[str, tuple[dict[str, Any], str]], bool]:
    claimed = _checkpoint_rows(workspace, str(state["workspace_id"]))
    job_map = {
        str(job["job_id"]): (job, source)
        for job, source in zip(jobs, rows, strict=True)
    }
    members = {entry.name for entry in workspace.iterdir()}
    allowed = {"resume.json", "checkpoint.json", *job_map}
    unexpected = members - allowed
    if unexpected:
        raise DesktopSingleLookBatchRecoveryError(
            "recovery workspace contains an unexpected member"
        )
    if set(claimed) - set(job_map):
        raise DesktopSingleLookBatchRecoveryError(
            "recovery checkpoint names an unknown job"
        )
    validated: dict[str, tuple[dict[str, Any], str]] = {}
    for job_id in sorted(set(members) & set(job_map)):
        job, source = job_map[job_id]
        validated[job_id] = _validate_child(
            workspace / job_id,
            workflow=workflow,
            source=source,
            job=job,
            style_id=style_id,
            destination=destination,
        )
    if set(claimed) - set(validated):
        raise DesktopSingleLookBatchRecoveryError(
            "recovery checkpoint names a missing child"
        )
    for job_id, digest in claimed.items():
        if validated[job_id][1] != digest:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery checkpoint child identity drifted"
            )
    reconciled = set(validated) != set(claimed)
    if reconciled:
        atomic_write_json(
            workspace / "checkpoint.json",
            _checkpoint_payload(
                str(state["workspace_id"]),
                {job_id: value[1] for job_id, value in validated.items()},
            ),
        )
    return validated, reconciled


def _write_checkpoint(
    workspace: Path,
    workspace_id: str,
    validated: Mapping[str, tuple[dict[str, Any], str]],
) -> None:
    atomic_write_json(
        workspace / "checkpoint.json",
        _checkpoint_payload(
            workspace_id,
            {job_id: value[1] for job_id, value in validated.items()},
        ),
    )


def _progress_receipt(
    *,
    state: Mapping[str, Any],
    workspace: Path,
    destination: Path,
    completed: int,
    total: int,
    reused: int,
    newly_completed: int,
    reconciled: bool,
) -> DesktopBatchProgressReceipt:
    payload = {
        "schema_version": PROGRESS_SCHEMA,
        "workspace_id": state["workspace_id"],
        "style_id": state["style_id"],
        "look_amount": state["look_amount"],
        "job_count": total,
        "completed_job_count": completed,
        "remaining_job_count": total - completed,
        "reused_job_count": reused,
        "newly_completed_job_count": newly_completed,
        "stale_checkpoint_reconciled": reconciled,
        "complete": False,
        "destination_published": False,
        "claim": state["claim"],
    }
    return DesktopBatchProgressReceipt(
        workspace_id=str(state["workspace_id"]),
        style_id=str(state["style_id"]),
        look_amount=float(state["look_amount"]),
        workspace_directory=workspace,
        output_directory=destination,
        completed_job_count=completed,
        remaining_job_count=total - completed,
        reused_job_count=reused,
        newly_completed_job_count=newly_completed,
        receipt=payload,
    )


def _final_receipt(
    state: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    validated: Mapping[str, tuple[dict[str, Any], str]],
) -> dict[str, Any]:
    receipt_jobs = [validated[str(job["job_id"])][0] for job in jobs]
    identity = {
        "schema_version": "kmcfm.desktop-single-look-batch.v1",
        "style_id": state["style_id"],
        "look_amount": state["look_amount"],
        "source_commit": state["source_commit"],
        "job_count": len(jobs),
        "output_format": "PNG",
        "output_bit_depth": 16,
        "jobs": [
            {
                key: row[key]
                for key in (
                    "job_id",
                    "input_basename",
                    "input_sha256",
                    "output_path",
                    "output_sha256",
                    "recipe_path",
                    "recipe_sha256",
                )
            }
            for row in receipt_jobs
        ],
        "claim": state["claim"],
    }
    return {"batch_id": _canonical_sha256(identity), **identity}


def _copy_create_only(
    source: Path, destination: Path, expected_source_sha256: str
) -> _FileSeal:
    _require_regular_single_link(source, "recovery assembly source")
    source_details = os.lstat(source)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o666,
    )
    details = os.fstat(descriptor)
    identity = PublishedFileIdentity(
        path=destination, device=details.st_dev, inode=details.st_ino
    )
    try:
        digest = hashlib.sha256()
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            descriptor = -1
            opened = os.fstat(reader.fileno())
            if (
                opened.st_dev != source_details.st_dev
                or opened.st_ino != source_details.st_ino
                or opened.st_nlink != 1
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise DesktopSingleLookBatchRecoveryError(
                    "recovery assembly source changed before copy"
                )
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        seal = _seal_file(destination)
        copied_source_sha256 = digest.hexdigest()
        if (
            copied_source_sha256 != expected_source_sha256
            or seal.identity != identity
            or seal.sha256 != copied_source_sha256
        ):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery assembly copy identity drifted"
            )
        return seal
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        remove_if_published(identity)
        raise


def _assemble_and_publish(
    *,
    workflow: ProductDesktopWorkflow,
    workspace: Path,
    destination: Path,
    state: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    validated: Mapping[str, tuple[dict[str, Any], str]],
) -> tuple[dict[str, Any], _DirectorySeal, list[_FileSeal]]:
    stage = destination.with_name(
        f".{destination.name}.u7-12d-publish-{state['workspace_id']}-{uuid.uuid4().hex}"
    )
    if _lexists(stage):
        raise DesktopSingleLookBatchRecoveryError(
            "recovery publication stage unexpectedly exists"
        )
    stage.mkdir()
    stage_seal = _seal_directory(stage)
    files: list[_FileSeal] = []
    final_root: _DirectorySeal | None = None
    final_files: list[_FileSeal] = []
    expected: set[str] = set()
    try:
        for job in jobs:
            row = validated[str(job["job_id"])][0]
            for key, sha_key in (
                ("output_path", "output_sha256"),
                ("recipe_path", "recipe_sha256"),
            ):
                name = str(row[key])
                expected.add(name)
                files.append(
                    _copy_create_only(
                        workspace / str(job["job_id"]) / name,
                        stage / name,
                        str(row[sha_key]),
                    )
                )
        receipt = _final_receipt(state, jobs, validated)
        files.append(_write_bound_json(stage / "batch.json", receipt))
        expected.add("batch.json")
        if {entry.name for entry in stage.iterdir()} != expected:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery publication stage member set drifted"
            )
        if _lexists(destination):
            raise DesktopSingleLookBatchRecoveryError("recovery destination appeared")
        os.rename(stage, destination)
        final_root = _seal_directory(destination)
        final_files = [
            _bind_file_after_directory_rename(seal, destination) for seal in files
        ]
        if {entry.name for entry in destination.iterdir()} != expected or not all(
            _file_matches(seal) for seal in final_files
        ):
            raise DesktopSingleLookBatchRecoveryError(
                "published recovery batch identity drifted"
            )
        for row in receipt["jobs"]:
            recipe_path = destination / row["recipe_path"]
            recipe = _load_json(recipe_path, "published recovery recipe")
            try:
                verify_render_recipe_files(
                    recipe,
                    profile_path=workflow.root
                    / "configs/render_profiles/safe_rich_product_v1.json",
                    root=workflow.root,
                )
            except Exception as exc:
                raise DesktopSingleLookBatchRecoveryError(
                    "published recovery recipe is invalid"
                ) from exc
            if (
                sha256_file(destination / row["output_path"]) != row["output_sha256"]
                or sha256_file(recipe_path) != row["recipe_sha256"]
            ):
                raise DesktopSingleLookBatchRecoveryError(
                    "published recovery member hash drifted"
                )
        if _load_json(destination / "batch.json", "published batch receipt") != receipt:
            raise DesktopSingleLookBatchRecoveryError(
                "published recovery receipt drifted"
            )
        return receipt, final_root, final_files
    except BaseException:
        if _directory_matches(stage_seal):
            _cleanup_bound_stage(stage_seal, files)
        elif final_root is not None:
            _cleanup_bound_stage(final_root, final_files)
        raise


def _cleanup_verified_workspace(
    workspace: Path,
    *,
    jobs: Sequence[Mapping[str, Any]],
    validated: Mapping[str, tuple[dict[str, Any], str]],
) -> bool:
    try:
        root = _seal_directory(workspace)
        root_files = [
            _seal_file(workspace / "resume.json"),
            _seal_file(workspace / "checkpoint.json"),
        ]
        child_seals: list[tuple[_DirectorySeal, list[_FileSeal]]] = []
        for job in jobs:
            job_id = str(job["job_id"])
            if job_id not in validated:
                return False
            child = workspace / job_id
            directory = _seal_directory(child)
            child_seals.append(
                (directory, [_seal_file(entry) for entry in child.iterdir()])
            )
    except (OSError, ProductDesktopError):
        return False
    for directory, files in reversed(child_seals):
        for seal in reversed(files):
            remove_if_published(seal.identity)
        if not _directory_matches(directory):
            return False
        try:
            directory.path.rmdir()
        except OSError:
            return False
    for seal in reversed(root_files):
        remove_if_published(seal.identity)
    if not _directory_matches(root):
        return False
    try:
        root.path.rmdir()
    except OSError:
        return False
    return True


def export_resumable_desktop_single_look_batch(
    workflow: ProductDesktopWorkflow,
    inputs: Sequence[DesktopBatchInput],
    style_id: str,
    workspace_directory: Path,
    output_directory: Path,
    *,
    cancel_event: threading.Event | None = None,
    progress: ProgressCallback | None = None,
    maximum_new_jobs: int | None = None,
) -> DesktopBatchReceipt | DesktopBatchProgressReceipt:
    """Resume verified single-look children and publish one exact flat batch."""

    if maximum_new_jobs is not None and (
        isinstance(maximum_new_jobs, bool)
        or not isinstance(maximum_new_jobs, int)
        or not 1 <= maximum_new_jobs <= 100
    ):
        raise DesktopSingleLookBatchRecoveryError(
            "maximum_new_jobs must be an integer in [1,100]"
        )
    with workflow._lock:
        preview = workflow.preview_state
        if preview is None:
            raise DesktopSingleLookBatchRecoveryError(
                "render previews before resumable export"
            )
        if style_id not in _LOOK_IDS:
            raise DesktopSingleLookBatchRecoveryError(
                "unknown or unavailable product look"
            )
        rows = tuple(inputs)
        if not 2 <= len(rows) <= 100:
            raise DesktopSingleLookBatchRecoveryError(
                "desktop recovery batch requires two to 100 photos"
            )
        if tuple(sorted((row.path for row in rows), key=_normalized_path)) != tuple(
            row.path for row in rows
        ):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery batch input order is not canonical"
            )
        if len({_normalized_path(row.path) for row in rows}) != len(rows) or len(
            {(row.device, row.inode) for row in rows}
        ) != len(rows):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery inputs must identify unique files"
            )
        if preview.input_path != rows[0].path or preview.input_sha256 != rows[0].sha256:
            raise DesktopSingleLookBatchRecoveryError(
                "representative preview does not bind this recovery batch"
            )
        workflow._validate_state(preview)
        workflow._validate_session(preview)
        if not all(_batch_input_matches(row) for row in rows):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery input changed after preview"
            )
        if not _batch_directory_rename_supported():
            raise DesktopSingleLookBatchRecoveryError(
                "desktop recovery publication requires Windows"
            )
        workspace = Path(workspace_directory).resolve(strict=False)
        destination = Path(output_directory).resolve(strict=False)
        if workspace == destination or workspace.parent != destination.parent:
            raise DesktopSingleLookBatchRecoveryError(
                "recovery workspace and destination must be distinct siblings"
            )
        if not _valid_windows_component(workspace.name) or not _valid_windows_component(
            destination.name
        ):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery workspace or destination name is unsafe on Windows"
            )
        _require_directory(workspace.parent, "recovery parent")
        if _lexists(destination):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery destination must be absent"
            )
        state = _expected_state(
            workflow=workflow,
            rows=rows,
            style_id=style_id,
            destination=destination,
        )
        jobs = list(state["jobs"])
        lease_path = workspace.with_name(f".{workspace.name}.u7-12d.lease")
        if (
            len(
                {
                    _normalized_path(workspace),
                    _normalized_path(destination),
                    _normalized_path(lease_path),
                }
            )
            != 3
        ):
            raise DesktopSingleLookBatchRecoveryError(
                "recovery workspace, destination and lease must be distinct"
            )
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("PYTHON")
        }
        published = False
        with _exclusive_lease(lease_path, _lease_payload(state)):
            if _lexists(destination):
                raise DesktopSingleLookBatchRecoveryError(
                    "recovery destination appeared"
                )
            if not _lexists(workspace):
                _create_workspace(workspace, state)
            _validate_workspace_state(workspace, state)
            _cleanup_reserved_transients(
                workspace,
                workspace_id=str(state["workspace_id"]),
                jobs=jobs,
                rows=rows,
                style_id=style_id,
            )
            validated, reconciled = _inspect_workspace(
                workspace,
                workflow=workflow,
                rows=rows,
                jobs=jobs,
                state=state,
                style_id=style_id,
                destination=destination,
            )
            reused = len(validated)
            newly_completed = 0
            for job, source in zip(jobs, rows, strict=True):
                job_id = str(job["job_id"])
                if job_id in validated:
                    continue
                if cancel_event is not None and cancel_event.is_set():
                    return _progress_receipt(
                        state=state,
                        workspace=workspace,
                        destination=destination,
                        completed=len(validated),
                        total=len(jobs),
                        reused=reused,
                        newly_completed=newly_completed,
                        reconciled=reconciled,
                    )
                if maximum_new_jobs is not None and newly_completed >= maximum_new_jobs:
                    return _progress_receipt(
                        state=state,
                        workspace=workspace,
                        destination=destination,
                        completed=len(validated),
                        total=len(jobs),
                        reused=reused,
                        newly_completed=newly_completed,
                        reconciled=reconciled,
                    )
                candidate = workspace / (
                    f"{_RESERVED_PREFIX}{state['workspace_id']}-{job_id}-"
                    f"{uuid.uuid4().hex}.candidate"
                )
                candidate.mkdir()
                candidate_seal = _seal_directory(candidate)
                candidate_files: list[_FileSeal] = []
                image_name, recipe_name = _output_names(job, source, style_id)
                image_path = candidate / image_name
                recipe_path = candidate / recipe_name
                try:
                    completed = workflow.command_runner(
                        workflow._build_export_command(
                            source.path,
                            style_id,
                            image_path,
                            preview.look_amount,
                        ),
                        workflow.root,
                        environment,
                    )
                    if completed.returncode != 0:
                        detail = (
                            completed.stderr or completed.stdout or "render failed"
                        )[-4000:]
                        raise DesktopSingleLookBatchRecoveryError(
                            f"recovery child {job_id} failed: {detail}"
                        )
                    if not image_path.is_file() or not recipe_path.is_file():
                        raise DesktopSingleLookBatchRecoveryError(
                            f"recovery child {job_id} returned without a complete pair"
                        )
                    candidate_files.extend(
                        (_seal_file(image_path), _seal_file(recipe_path))
                    )
                    recipe = _load_json(recipe_path, "new recovery recipe")
                    try:
                        verify_render_recipe_files(
                            recipe,
                            profile_path=workflow.root
                            / "configs/render_profiles/safe_rich_product_v1.json",
                            root=workflow.root,
                        )
                    except Exception as exc:
                        raise DesktopSingleLookBatchRecoveryError(
                            f"recovery child {job_id} recipe is invalid"
                        ) from exc
                    recipe["output"]["path"] = str(
                        (destination / image_name).resolve(strict=False)
                    )
                    validate_render_recipe(recipe)
                    verify_render_recipe_inputs(
                        recipe,
                        profile_path=workflow.root
                        / "configs/render_profiles/safe_rich_product_v1.json",
                        root=workflow.root,
                    )
                    candidate_files[1] = _replace_owned_json(recipe_path, recipe)
                    child_receipt = {
                        "schema_version": CHILD_SCHEMA,
                        "job_id": job_id,
                        "input_basename": source.basename,
                        "input_sha256": source.sha256,
                        "output_path": PurePosixPath(image_name).as_posix(),
                        "output_sha256": candidate_files[0].sha256,
                        "recipe_path": PurePosixPath(recipe_name).as_posix(),
                        "recipe_sha256": candidate_files[1].sha256,
                    }
                    candidate_files.append(
                        _write_bound_json(candidate / "child.json", child_receipt)
                    )
                    candidate_result = _validate_child(
                        candidate,
                        workflow=workflow,
                        source=source,
                        job=job,
                        style_id=style_id,
                        destination=destination,
                    )
                    canonical = workspace / job_id
                    if _lexists(canonical):
                        raise DesktopSingleLookBatchRecoveryError(
                            f"recovery child destination appeared for {job_id}"
                        )
                    os.rename(candidate, canonical)
                    validated[job_id] = _validate_child(
                        canonical,
                        workflow=workflow,
                        source=source,
                        job=job,
                        style_id=style_id,
                        destination=destination,
                    )
                    if validated[job_id] != candidate_result:
                        raise DesktopSingleLookBatchRecoveryError(
                            f"recovery child {job_id} changed during publication"
                        )
                    _write_checkpoint(workspace, str(state["workspace_id"]), validated)
                    newly_completed += 1
                    if progress is not None:
                        progress(len(validated), len(jobs), source.basename)
                except BaseException:
                    if _directory_matches(candidate_seal):
                        for path in (image_path, recipe_path, candidate / "child.json"):
                            if path.is_file():
                                try:
                                    seal = _seal_file(path)
                                except ProductDesktopError:
                                    continue
                                if seal not in candidate_files:
                                    candidate_files.append(seal)
                        _cleanup_bound_stage(candidate_seal, candidate_files)
                    raise

            if cancel_event is not None and cancel_event.is_set():
                return _progress_receipt(
                    state=state,
                    workspace=workspace,
                    destination=destination,
                    completed=len(validated),
                    total=len(jobs),
                    reused=reused,
                    newly_completed=newly_completed,
                    reconciled=reconciled,
                )
            workflow._validate_state(preview)
            workflow._validate_session(preview)
            if _source_commit(workflow.root) != preview.source_commit or not all(
                _batch_input_matches(row) for row in rows
            ):
                raise DesktopSingleLookBatchRecoveryError(
                    "recovery authority changed before publication"
                )
            current = _expected_state(
                workflow=workflow,
                rows=rows,
                style_id=style_id,
                destination=destination,
            )
            if current != state:
                raise DesktopSingleLookBatchRecoveryError(
                    "recovery identity changed before publication"
                )
            validated, _ = _inspect_workspace(
                workspace,
                workflow=workflow,
                rows=rows,
                jobs=jobs,
                state=state,
                style_id=style_id,
                destination=destination,
            )
            if len(validated) != len(jobs):
                raise DesktopSingleLookBatchRecoveryError(
                    "not all recovery jobs are complete"
                )
            receipt, _, _ = _assemble_and_publish(
                workflow=workflow,
                workspace=workspace,
                destination=destination,
                state=state,
                jobs=jobs,
                validated=validated,
            )
            published = True
            _cleanup_verified_workspace(workspace, jobs=jobs, validated=validated)

        if published:
            try:
                lease_seal = _seal_file(lease_path)
            except ProductDesktopError:
                lease_seal = None
            if lease_seal is not None and lease_path.read_bytes() == _lease_payload(
                state
            ):
                remove_if_published(lease_seal.identity)
        receipt_path = destination / "batch.json"
        return DesktopBatchReceipt(
            batch_id=str(receipt["batch_id"]),
            style_id=style_id,
            look_amount=preview.look_amount,
            output_directory=destination,
            receipt_path=receipt_path,
            receipt_sha256=sha256_file(receipt_path),
            job_count=len(rows),
            receipt=receipt,
        )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHILD_SCHEMA",
    "LEASE_SCHEMA",
    "PROGRESS_SCHEMA",
    "WORKSPACE_SCHEMA",
    "DesktopBatchProgressReceipt",
    "DesktopSingleLookBatchRecoveryError",
    "export_resumable_desktop_single_look_batch",
]
