"""Resumable transaction for the deterministic three-look input batch."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

try:  # pragma: no cover - imported only to keep the module importable elsewhere.
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]

from .render_contract import (
    atomic_write_json,
    load_render_profile,
    sha256_file,
    validate_render_recipe,
)
from .three_stock_batch import render_three_stock_batch_to_directory
from .three_stock_input_batch import (
    _EXPECTED_STYLES,
    _canonical_sha256,
    _load_jobs,
    _preflight_jobs,
    _rewrite_child_for_final_destination,
)
from .three_stock_look import (
    THREE_STOCK_LOOK_CATALOG,
    resolve_three_stock_look_parameters,
)

WORKSPACE_SCHEMA = "neuro-film.resumable-three-stock-workspace.v1"
CHECKPOINT_SCHEMA = "neuro-film.resumable-three-stock-checkpoint.v1"
PROGRESS_SCHEMA = "neuro-film.resumable-three-stock-progress.v1"
RECEIPT_SCHEMA = "neuro-film.resumable-three-stock-input-batch.v1"
LEASE_SCHEMA = "neuro-film.resumable-three-stock-lease.v1"
CLAIM_CEILING = (
    "Three deterministic film-inspired Look Approximations per input; calibrated "
    "or physical stock response is not established."
)
_RESERVED_PREFIX = ".u7-8b-"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_CORE_IDENTITY_PATHS = (
    "src/inference/resumable_three_stock_input_batch.py",
    "src/inference/three_stock_input_batch.py",
    "src/inference/three_stock_batch.py",
    "src/inference/three_stock_look.py",
    "src/inference/render_contract.py",
    "src/inference/style_safe_engine.py",
    "src/inference/product_look_catalog.py",
    "src/preprocess/__init__.py",
    "src/preprocess/pipeline.py",
    "src/preprocess/raster_decode.py",
    "src/preprocess/raw_decode.py",
    "src/preprocess/output_encode.py",
    "src/preprocess/color_management.py",
    "src/preprocess/color_state.py",
    "src/preprocess/types.py",
    "scripts/pipeline_color_baseline.py",
)
_CHILD_MEMBERS = {
    "batch.json",
    *{f"{style}.png" for style in _EXPECTED_STYLES},
    *{f"{style}.recipe.json" for style in _EXPECTED_STYLES},
}
_CHILD_MANIFEST_KEYS = {
    "schema_version",
    "input_path",
    "input_sha256",
    "profile_sha256",
    "look_amount",
    "png_compression",
    "rows",
    "claim_ceiling",
}
_CHILD_ROW_KEYS = {
    "film_stock_id",
    "style_id",
    "output_path",
    "output_sha256",
    "recipe_path",
    "recipe_sha256",
}
_CHILD_CLAIM_CEILING = (
    "Three deterministic non-calibrated Look Approximations; stock separation "
    "and calibrated response are not established."
)
_STOCK_BY_STYLE = {
    row["style_id"]: row["film_stock_id"] for row in THREE_STOCK_LOOK_CATALOG
}


class ResumableThreeStockBatchError(ValueError):
    """Reject an unsafe or inconsistent resumable batch workspace."""


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
            raise ResumableThreeStockBatchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumableThreeStockBatchError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ResumableThreeStockBatchError(f"{label} must be a JSON object")
    return value


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_reparse(path: Path) -> bool:
    info = path.lstat()
    return bool(
        path.is_symlink()
        or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    )


def _require_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ResumableThreeStockBatchError(f"{label} is unavailable") from exc
    if _is_reparse(path) or not stat.S_ISDIR(info.st_mode):
        raise ResumableThreeStockBatchError(
            f"{label} must be a non-reparse regular directory"
        )


def _require_regular_single_link(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ResumableThreeStockBatchError(f"{label} is unavailable") from exc
    if (
        _is_reparse(path)
        or not stat.S_ISREG(info.st_mode)
        or getattr(info, "st_nlink", 1) != 1
    ):
        raise ResumableThreeStockBatchError(
            f"{label} must be a regular single-link file"
        )


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.fspath(path.resolve(strict=False))).casefold()


def _path_binding(path: Path) -> str:
    return hashlib.sha256(_normalized_path(path).encode("utf-8")).hexdigest()


def _software_commit(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResumableThreeStockBatchError("software commit is unavailable") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ResumableThreeStockBatchError("software commit is invalid")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ResumableThreeStockBatchError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _look_amount(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ResumableThreeStockBatchError("look_amount must be finite in [0, 1]")
    return float(value)


def _core_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _CORE_IDENTITY_PATHS:
        path = root / relative
        _require_regular_single_link(path, f"core file {relative}")
        result[relative] = sha256_file(path)
    return result


def _expected_state(
    *,
    jobs: list[dict[str, Any]],
    manifest_path: Path,
    output_directory: Path,
    root: Path,
    profile_path: Path,
    statistics_path: Path,
    guardrails_path: Path,
    look_amount: float,
    seed: int,
    tile_size: int,
    tile_workers: int,
    png_compression: int,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA,
        "manifest_sha256": sha256_file(manifest_path),
        "destination_binding_sha256": _path_binding(output_directory),
        "software_commit": _software_commit(root),
        "core_sha256": _core_hashes(root),
        "profile_sha256": sha256_file(profile_path),
        "statistics_sha256": sha256_file(statistics_path),
        "guardrails_sha256": sha256_file(guardrails_path),
        "look_amount": float(look_amount),
        "seed": seed,
        "tile_size": tile_size,
        "tile_workers": tile_workers,
        "png_compression": png_compression,
        "jobs": [
            {"job_id": row["job_id"], "input_sha256": row["input_sha256"]}
            for row in jobs
        ],
        "claim_ceiling": CLAIM_CEILING,
    }
    return {"workspace_id": _canonical_sha256(identity), **identity}


def _lease_payload(state: dict[str, Any]) -> bytes:
    return _canonical_bytes(
        {"schema_version": LEASE_SCHEMA, "workspace_id": state["workspace_id"]}
    )


def _prepare_lease_file(path: Path, expected: bytes) -> None:
    if not _lexists(path):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
    _require_regular_single_link(path, "workspace lease")


@contextmanager
def _exclusive_lease(path: Path, expected: bytes) -> Iterator[BinaryIO]:
    if os.name != "nt" or msvcrt is None:
        raise ResumableThreeStockBatchError("resumable publication requires Windows")
    _prepare_lease_file(path, expected)
    handle = path.open("r+b", buffering=0)
    try:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ResumableThreeStockBatchError(
                "workspace is held by another writer"
            ) from exc
        try:
            handle.seek(0)
            if handle.read() != expected:
                raise ResumableThreeStockBatchError(
                    "workspace lease identity mismatch"
                )
            yield handle
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _checkpoint_payload(
    workspace_id: str, rows: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "workspace_id": workspace_id,
        "completed_jobs": sorted(rows, key=lambda row: row["job_id"]),
    }


def _create_workspace(workspace: Path, state: dict[str, Any]) -> None:
    stage = workspace.with_name(
        f".{workspace.name}.u7-8b-init-{state['workspace_id']}-{uuid.uuid4().hex}"
    )
    stage.mkdir()
    try:
        atomic_write_json(stage / "resume.json", state)
        atomic_write_json(
            stage / "checkpoint.json",
            _checkpoint_payload(state["workspace_id"], []),
        )
        os.rename(stage, workspace)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _reconcile_owned_init_stages(
    workspace: Path, state: dict[str, Any]
) -> None:
    """Remove only complete state-bound sibling initialization remnants."""

    prefix = f".{workspace.name}.u7-8b-init-{state['workspace_id']}-"
    pattern = re.compile(re.escape(prefix) + r"[0-9a-f]{32}")
    temporary_pattern = re.compile(
        r"\.(?:resume|checkpoint)\.json\.[0-9]+\.tmp"
    )
    for entry in workspace.parent.iterdir():
        if pattern.fullmatch(entry.name) is None:
            continue
        _require_directory(entry, "workspace initialization remnant")
        members: set[str] = set()
        for child in entry.iterdir():
            members.add(child.name)
            if child.name not in {"resume.json", "checkpoint.json"} and (
                temporary_pattern.fullmatch(child.name) is None
            ):
                raise ResumableThreeStockBatchError(
                    "workspace initialization remnant is not safely attributable"
                )
            _require_regular_single_link(child, "initialization remnant member")
        resume_path = entry / "resume.json"
        checkpoint_path = entry / "checkpoint.json"
        if "resume.json" in members and (
            _load_json(resume_path, "initialization state") != state
        ):
            raise ResumableThreeStockBatchError(
                "workspace initialization remnant identity mismatch"
            )
        if "checkpoint.json" in members and "resume.json" not in members:
            raise ResumableThreeStockBatchError(
                "workspace initialization checkpoint lacks state identity"
            )
        if "checkpoint.json" in members and (
            _load_json(checkpoint_path, "initialization checkpoint")
            != _checkpoint_payload(state["workspace_id"], [])
        ):
            raise ResumableThreeStockBatchError(
                "workspace initialization checkpoint identity mismatch"
            )
        shutil.rmtree(entry)


def _validate_workspace_state(workspace: Path, expected: dict[str, Any]) -> None:
    _require_directory(workspace, "workspace")
    resume_path = workspace / "resume.json"
    _require_regular_single_link(resume_path, "workspace state")
    if _load_json(resume_path, "workspace state") != expected:
        raise ResumableThreeStockBatchError("workspace identity mismatch")


def _cleanup_reserved_transients(workspace: Path) -> None:
    for entry in workspace.iterdir():
        if not entry.name.startswith(_RESERVED_PREFIX):
            continue
        if _is_reparse(entry):
            raise ResumableThreeStockBatchError(
                "reserved transient must not be a link or reparse point"
            )
        info = entry.lstat()
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(entry)
        elif stat.S_ISREG(info.st_mode) and getattr(info, "st_nlink", 1) == 1:
            entry.unlink()
        else:
            raise ResumableThreeStockBatchError(
                "reserved transient has an unsupported type"
            )


def _validate_child(
    child: Path,
    *,
    job: dict[str, Any],
    output_directory: Path,
    profile: dict[str, Any],
    profile_sha256: str,
    software_commit: str,
    look_amount: float,
    seed: int,
    png_compression: int,
) -> tuple[str, list[dict[str, Any]]]:
    _require_directory(child, f"child checkpoint {job['job_id']}")
    members = {entry.name for entry in child.iterdir()}
    if members != _CHILD_MEMBERS:
        raise ResumableThreeStockBatchError(
            f"child checkpoint members drifted for job {job['job_id']}"
        )
    for name in sorted(members):
        _require_regular_single_link(
            child / name, f"child checkpoint member {job['job_id']}/{name}"
        )
    manifest_path = child / "batch.json"
    manifest = _load_json(manifest_path, f"child manifest {job['job_id']}")
    rows = manifest.get("rows")
    if (
        set(manifest) != _CHILD_MANIFEST_KEYS
        or manifest.get("schema_version")
        != "neuro-film.three-stock-file-batch.v1"
        or manifest.get("input_sha256") != job["input_sha256"]
        or _normalized_path(Path(str(manifest.get("input_path", ""))))
        != _normalized_path(job["input_path"])
        or manifest.get("profile_sha256") != profile_sha256
        or manifest.get("look_amount") != float(look_amount)
        or manifest.get("png_compression") != png_compression
        or manifest.get("claim_ceiling") != _CHILD_CLAIM_CEILING
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
        or any(set(row) != _CHILD_ROW_KEYS for row in rows)
        or tuple(row.get("style_id") for row in rows)
        != _EXPECTED_STYLES
    ):
        raise ResumableThreeStockBatchError(
            f"child manifest identity drifted for job {job['job_id']}"
        )
    receipt_rows: list[dict[str, Any]] = []
    for row in rows:
        style = row["style_id"]
        film_stock_id = _STOCK_BY_STYLE[style]
        _, expected_color_parameters = resolve_three_stock_look_parameters(
            profile,
            film_stock_id=film_stock_id,
            look_amount=look_amount,
        )
        output_name = f"{style}.png"
        recipe_name = f"{style}.recipe.json"
        output_path = child / output_name
        recipe_path = child / recipe_name
        recipe = _load_json(recipe_path, f"child recipe {job['job_id']}/{style}")
        try:
            validate_render_recipe(recipe)
        except ValueError as exc:
            raise ResumableThreeStockBatchError(
                f"child recipe is invalid for job {job['job_id']}"
            ) from exc
        final_output = output_directory / job["job_id"] / output_name
        final_recipe = output_directory / job["job_id"] / recipe_name
        output_sha256 = sha256_file(output_path)
        recipe_sha256 = sha256_file(recipe_path)
        expected_effects = {
            "grain": {"strength": 0.0, "seed": seed, "color": True},
            "halation": {
                "strength": 0.0,
                "model": "simple",
                "preset": None,
                "control_mode": "locked",
                "resolved_parameters": None,
            },
            "dust": {"strength": 0.0, "seed": seed + 17},
        }
        if (
            row["film_stock_id"] != film_stock_id
            or recipe["profile"]
            != {
                "profile_id": profile["profile_id"],
                "profile_version": profile["profile_version"],
                "sha256": profile_sha256,
            }
            or recipe["assets"] != profile["assets"]
            or recipe["render"]["style"] != style
            or recipe["render"]["seed"] != seed
            or recipe["render"]["look_amount"] != float(look_amount)
            or recipe["render"]["color_parameters"]
            != expected_color_parameters
            or recipe["render"]["effects"] != expected_effects
            or recipe["claim"]["claim_ceiling"]
            != profile["evidence"]["claim_ceiling"]
            or recipe["input"]["sha256"] != job["input_sha256"]
            or _normalized_path(Path(recipe["input"]["path"]))
            != _normalized_path(job["input_path"])
            or recipe["profile"]["sha256"] != profile_sha256
            or recipe["software"]["commit"] != software_commit
            or _normalized_path(Path(recipe["output"]["path"]))
            != _normalized_path(final_output)
            or recipe["output"]["sha256"] != output_sha256
            or _normalized_path(Path(row["output_path"]))
            != _normalized_path(final_output)
            or row["output_sha256"] != output_sha256
            or _normalized_path(Path(row["recipe_path"]))
            != _normalized_path(final_recipe)
            or row["recipe_sha256"] != recipe_sha256
        ):
            raise ResumableThreeStockBatchError(
                f"child artifact identity drifted for job {job['job_id']}"
            )
        receipt_rows.append(
            {
                "film_stock_id": film_stock_id,
                "style_id": style,
                "output_path": PurePosixPath(job["job_id"], output_name).as_posix(),
                "output_sha256": output_sha256,
                "recipe_path": PurePosixPath(job["job_id"], recipe_name).as_posix(),
                "recipe_sha256": recipe_sha256,
            }
        )
    return sha256_file(manifest_path), receipt_rows


def _load_checkpoint(
    workspace: Path, workspace_id: str
) -> list[dict[str, str]]:
    path = workspace / "checkpoint.json"
    _require_regular_single_link(path, "checkpoint ledger")
    payload = _load_json(path, "checkpoint ledger")
    if set(payload) != {"schema_version", "workspace_id", "completed_jobs"}:
        raise ResumableThreeStockBatchError("checkpoint ledger keys mismatch")
    rows = payload["completed_jobs"]
    if (
        payload["schema_version"] != CHECKPOINT_SCHEMA
        or payload["workspace_id"] != workspace_id
        or not isinstance(rows, list)
    ):
        raise ResumableThreeStockBatchError("checkpoint ledger identity mismatch")
    validated: list[dict[str, str]] = []
    previous = ""
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "job_id",
            "child_manifest_sha256",
        }:
            raise ResumableThreeStockBatchError("checkpoint ledger row is invalid")
        job_id = row["job_id"]
        manifest_sha256 = row["child_manifest_sha256"]
        if (
            not isinstance(job_id, str)
            or job_id <= previous
            or not isinstance(manifest_sha256, str)
            or _HASH.fullmatch(manifest_sha256) is None
        ):
            raise ResumableThreeStockBatchError("checkpoint ledger order is invalid")
        previous = job_id
        validated.append(dict(row))
    return validated


def _inspect_and_reconcile(
    workspace: Path,
    *,
    state: dict[str, Any],
    jobs: list[dict[str, Any]],
    output_directory: Path,
    profile: dict[str, Any],
) -> tuple[dict[str, tuple[str, list[dict[str, Any]]]], bool]:
    checkpoint_rows = _load_checkpoint(workspace, state["workspace_id"])
    claimed = {row["job_id"]: row["child_manifest_sha256"] for row in checkpoint_rows}
    job_by_id = {row["job_id"]: row for row in jobs}
    allowed_files = {"resume.json", "checkpoint.json", "batch.json"}
    actual_children: set[str] = set()
    for entry in workspace.iterdir():
        if entry.name.startswith(_RESERVED_PREFIX):
            continue
        if entry.name in allowed_files:
            if entry.name != "batch.json":
                _require_regular_single_link(entry, f"workspace member {entry.name}")
            continue
        if entry.name not in job_by_id:
            raise ResumableThreeStockBatchError("workspace contains an unexpected member")
        actual_children.add(entry.name)
    if not set(claimed).issubset(job_by_id):
        raise ResumableThreeStockBatchError("checkpoint ledger names an unknown job")
    if not set(claimed).issubset(actual_children):
        raise ResumableThreeStockBatchError("checkpoint ledger names a missing child")

    validated: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for job_id in sorted(actual_children):
        result = _validate_child(
            workspace / job_id,
            job=job_by_id[job_id],
            output_directory=output_directory,
            profile=profile,
            profile_sha256=state["profile_sha256"],
            software_commit=state["software_commit"],
            look_amount=state["look_amount"],
            seed=state["seed"],
            png_compression=state["png_compression"],
        )
        if job_id in claimed and claimed[job_id] != result[0]:
            raise ResumableThreeStockBatchError(
                "checkpoint ledger manifest identity drifted"
            )
        validated[job_id] = result

    reconciled = set(validated) != set(claimed)
    if reconciled:
        atomic_write_json(
            workspace / "checkpoint.json",
            _checkpoint_payload(
                state["workspace_id"],
                [
                    {"job_id": job_id, "child_manifest_sha256": value[0]}
                    for job_id, value in validated.items()
                ],
            ),
        )
    batch_path = workspace / "batch.json"
    if _lexists(batch_path):
        _require_regular_single_link(batch_path, "aggregate receipt")
        if len(validated) != len(jobs):
            raise ResumableThreeStockBatchError(
                "aggregate receipt exists before all jobs are complete"
            )
    return validated, reconciled


def _write_checkpoint(
    workspace: Path,
    workspace_id: str,
    validated: dict[str, tuple[str, list[dict[str, Any]]]],
) -> None:
    atomic_write_json(
        workspace / "checkpoint.json",
        _checkpoint_payload(
            workspace_id,
            [
                {"job_id": job_id, "child_manifest_sha256": value[0]}
                for job_id, value in validated.items()
            ],
        ),
    )


def _progress_receipt(
    *,
    state: dict[str, Any],
    completed: int,
    total: int,
    reused: int,
    newly_completed: int,
    reconciled: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA,
        "workspace_id": state["workspace_id"],
        "job_count": total,
        "completed_job_count": completed,
        "remaining_job_count": total - completed,
        "reused_job_count": reused,
        "newly_completed_job_count": newly_completed,
        "stale_checkpoint_reconciled": reconciled,
        "complete": completed == total,
        "destination_published": False,
        "claim_ceiling": CLAIM_CEILING,
    }


def _final_receipt(
    *,
    state: dict[str, Any],
    jobs: list[dict[str, Any]],
    validated: dict[str, tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    receipt_jobs = [
        {
            "job_id": job["job_id"],
            "input_sha256": job["input_sha256"],
            "child_manifest_path": PurePosixPath(
                job["job_id"], "batch.json"
            ).as_posix(),
            "child_manifest_sha256": validated[job["job_id"]][0],
            "rows": validated[job["job_id"]][1],
        }
        for job in jobs
    ]
    identity = {
        "schema_version": RECEIPT_SCHEMA,
        "workspace_id": state["workspace_id"],
        "job_count": len(jobs),
        "profile_sha256": state["profile_sha256"],
        "statistics_sha256": state["statistics_sha256"],
        "guardrails_sha256": state["guardrails_sha256"],
        "look_amount": state["look_amount"],
        "seed": state["seed"],
        "tile_size": state["tile_size"],
        "tile_workers": state["tile_workers"],
        "png_compression": state["png_compression"],
        "jobs": receipt_jobs,
        "claim_ceiling": CLAIM_CEILING,
    }
    return {"batch_id": _canonical_sha256(identity), **identity}


def render_resumable_three_stock_input_batch_to_directory(
    manifest_path: Path,
    workspace_directory: Path,
    output_directory: Path,
    *,
    root: Path,
    profile_path: Path,
    statistics_path: Path,
    guardrails_path: Path,
    look_amount: float = 1.0,
    seed: int = 31,
    tile_size: int = 512,
    tile_workers: int = 1,
    png_compression: int = 0,
    maximum_new_jobs: int | None = None,
) -> dict[str, Any]:
    """Resume verified jobs and create one complete no-replace batch directory."""

    if os.name != "nt":
        raise ResumableThreeStockBatchError("resumable publication requires Windows")
    manifest_path = Path(manifest_path)
    workspace = Path(workspace_directory)
    output = Path(output_directory)
    root = Path(root)
    if maximum_new_jobs is not None:
        maximum_new_jobs = _integer(
            maximum_new_jobs, "maximum_new_jobs", 1, 100
        )
    look_amount = _look_amount(look_amount)
    seed = _integer(seed, "seed", -(2**31), 2**31 - 1)
    tile_size = _integer(tile_size, "tile_size", 1, 2**31 - 1)
    tile_workers = _integer(tile_workers, "tile_workers", 1, 2**31 - 1)
    png_compression = _integer(png_compression, "png_compression", 0, 9)
    if not manifest_path.is_file():
        raise ResumableThreeStockBatchError("manifest_path must be an existing file")
    if _normalized_path(workspace) == _normalized_path(output):
        raise ResumableThreeStockBatchError("workspace and destination must differ")
    if _normalized_path(workspace.parent) != _normalized_path(output.parent):
        raise ResumableThreeStockBatchError(
            "workspace and destination must be siblings"
        )
    _require_directory(workspace.parent, "workspace parent")
    if _lexists(output):
        raise ResumableThreeStockBatchError("output destination must not exist")
    for path, label in (
        (profile_path, "profile"),
        (statistics_path, "statistics"),
        (guardrails_path, "guardrails"),
    ):
        _require_regular_single_link(Path(path), label)

    jobs = _load_jobs(manifest_path)
    _preflight_jobs(jobs)
    state = _expected_state(
        jobs=jobs,
        manifest_path=manifest_path,
        output_directory=output,
        root=root,
        profile_path=Path(profile_path),
        statistics_path=Path(statistics_path),
        guardrails_path=Path(guardrails_path),
        look_amount=look_amount,
        seed=seed,
        tile_size=tile_size,
        tile_workers=tile_workers,
        png_compression=png_compression,
    )
    lease_path = workspace.with_name(f".{workspace.name}.u7-8b.lease")
    normalized_paths = {
        _normalized_path(workspace),
        _normalized_path(output),
        _normalized_path(lease_path),
    }
    if len(normalized_paths) != 3:
        raise ResumableThreeStockBatchError(
            "workspace, destination and lease paths must be distinct"
        )
    profile = load_render_profile(Path(profile_path), root=root)
    published = False
    with _exclusive_lease(lease_path, _lease_payload(state)):
        if _lexists(output):
            raise ResumableThreeStockBatchError("output destination appeared")
        _reconcile_owned_init_stages(workspace, state)
        if not _lexists(workspace):
            _create_workspace(workspace, state)
        _validate_workspace_state(workspace, state)
        _cleanup_reserved_transients(workspace)
        validated, reconciled = _inspect_and_reconcile(
            workspace,
            state=state,
            jobs=jobs,
            output_directory=output,
            profile=profile,
        )
        reused = len(validated)
        newly_completed = 0
        job_by_id = {row["job_id"]: row for row in jobs}
        for job in jobs:
            job_id = job["job_id"]
            if job_id in validated:
                continue
            if maximum_new_jobs is not None and newly_completed >= maximum_new_jobs:
                return _progress_receipt(
                    state=state,
                    completed=len(validated),
                    total=len(jobs),
                    reused=reused,
                    newly_completed=newly_completed,
                    reconciled=reconciled,
                )
            candidate = workspace / (
                f"{_RESERVED_PREFIX}{job_id}-{uuid.uuid4().hex}.candidate"
            )
            try:
                child_manifest = render_three_stock_batch_to_directory(
                    job["input_path"],
                    candidate,
                    root=root,
                    profile_path=Path(profile_path),
                    statistics_path=Path(statistics_path),
                    guardrails_path=Path(guardrails_path),
                    look_amount=look_amount,
                    seed=seed,
                    tile_size=tile_size,
                    tile_workers=tile_workers,
                    png_compression=png_compression,
                )
                if child_manifest.get("input_sha256") != job["input_sha256"]:
                    raise ResumableThreeStockBatchError(
                        f"child input identity drifted for job {job_id}"
                    )
                _rewrite_child_for_final_destination(
                    child_manifest,
                    child_stage=candidate,
                    final_child=output / job_id,
                    job_id=job_id,
                    expected_input_sha256=job["input_sha256"],
                )
                candidate_result = _validate_child(
                    candidate,
                    job=job,
                    output_directory=output,
                    profile=profile,
                    profile_sha256=state["profile_sha256"],
                    software_commit=state["software_commit"],
                    look_amount=look_amount,
                    seed=seed,
                    png_compression=png_compression,
                )
                canonical_child = workspace / job_id
                if _lexists(canonical_child):
                    raise ResumableThreeStockBatchError(
                        f"canonical child appeared for job {job_id}"
                    )
                os.rename(candidate, canonical_child)
                validated[job_id] = _validate_child(
                    canonical_child,
                    job=job_by_id[job_id],
                    output_directory=output,
                    profile=profile,
                    profile_sha256=state["profile_sha256"],
                    software_commit=state["software_commit"],
                    look_amount=look_amount,
                    seed=seed,
                    png_compression=png_compression,
                )
                if validated[job_id] != candidate_result:
                    raise ResumableThreeStockBatchError(
                        f"published child drifted for job {job_id}"
                    )
                _write_checkpoint(workspace, state["workspace_id"], validated)
                newly_completed += 1
            except BaseException:
                if _lexists(candidate) and not _is_reparse(candidate):
                    shutil.rmtree(candidate, ignore_errors=True)
                raise

        _preflight_jobs(jobs)
        current_state = _expected_state(
            jobs=jobs,
            manifest_path=manifest_path,
            output_directory=output,
            root=root,
            profile_path=Path(profile_path),
            statistics_path=Path(statistics_path),
            guardrails_path=Path(guardrails_path),
            look_amount=look_amount,
            seed=seed,
            tile_size=tile_size,
            tile_workers=tile_workers,
            png_compression=png_compression,
        )
        if current_state != state:
            raise ResumableThreeStockBatchError(
                "input, configuration or core identity drifted before publication"
            )
        validated, _ = _inspect_and_reconcile(
            workspace,
            state=state,
            jobs=jobs,
            output_directory=output,
            profile=profile,
        )
        if len(validated) != len(jobs):
            raise ResumableThreeStockBatchError("not all jobs are complete")
        receipt = _final_receipt(state=state, jobs=jobs, validated=validated)
        batch_path = workspace / "batch.json"
        if _lexists(batch_path):
            _require_regular_single_link(batch_path, "aggregate receipt")
            if _load_json(batch_path, "aggregate receipt") != receipt:
                raise ResumableThreeStockBatchError(
                    "existing aggregate receipt identity drifted"
                )
        else:
            atomic_write_json(batch_path, receipt)
        _require_regular_single_link(workspace / "batch.json", "aggregate receipt")
        if _load_json(workspace / "batch.json", "aggregate receipt") != receipt:
            raise ResumableThreeStockBatchError("aggregate receipt drifted")
        if _lexists(output):
            raise ResumableThreeStockBatchError(
                "output destination appeared during publication"
            )
        os.rename(workspace, output)
        published = True

    if published:
        try:
            if lease_path.read_bytes() == _lease_payload(state):
                lease_path.unlink()
        except OSError:
            pass
    return receipt


__all__ = [
    "CHECKPOINT_SCHEMA",
    "PROGRESS_SCHEMA",
    "RECEIPT_SCHEMA",
    "WORKSPACE_SCHEMA",
    "ResumableThreeStockBatchError",
    "render_resumable_three_stock_input_batch_to_directory",
]
