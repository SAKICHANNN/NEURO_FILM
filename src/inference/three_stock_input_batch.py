"""Deterministic multi-input transaction for the three product look approximations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from .render_contract import atomic_write_json, sha256_file, validate_render_recipe
from .three_stock_batch import render_three_stock_batch_to_directory

INPUT_MANIFEST_SCHEMA = "neuro-film.three-stock-input-jobs.v1"
RECEIPT_SCHEMA = "neuro-film.three-stock-input-batch.v1"
MAXIMUM_JOBS = 100
_HASH = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_WINDOWS_RESERVED_JOB_BASENAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_MANIFEST_KEYS = {"jobs", "schema_version"}
_JOB_KEYS = {"input_path", "input_sha256", "job_id"}
_EXPECTED_STYLES = ("velvia_50", "portra_400", "ektar_100")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ThreeStockInputBatchError(ValueError):
    """Raised when a multi-input look transaction cannot be completed."""


def _job_id_is_safe_path_component(value: object) -> bool:
    if not isinstance(value, str) or _JOB_ID.fullmatch(value) is None:
        return False
    if value.endswith("."):
        return False
    return value.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED_JOB_BASENAMES


def _software_commit(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ThreeStockInputBatchError("software commit is unavailable") from exc
    if _COMMIT.fullmatch(value) is None:
        raise ThreeStockInputBatchError("software commit is invalid")
    return value


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ThreeStockInputBatchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_jobs(manifest_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ThreeStockInputBatchError(
            "input manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise ThreeStockInputBatchError("input manifest keys mismatch")
    if payload["schema_version"] != INPUT_MANIFEST_SCHEMA:
        raise ThreeStockInputBatchError("input manifest schema mismatch")
    raw_jobs = payload["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs or len(raw_jobs) > MAXIMUM_JOBS:
        raise ThreeStockInputBatchError(
            f"input manifest must contain 1..{MAXIMUM_JOBS} jobs"
        )
    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict) or set(raw) != _JOB_KEYS:
            raise ThreeStockInputBatchError(f"job {index} keys mismatch")
        job_id = raw["job_id"]
        input_path = raw["input_path"]
        expected_sha256 = raw["input_sha256"]
        if not _job_id_is_safe_path_component(job_id):
            raise ThreeStockInputBatchError(f"job {index} id is invalid")
        if job_id in seen_ids:
            raise ThreeStockInputBatchError("job ids must be unique")
        seen_ids.add(job_id)
        if not isinstance(input_path, str) or not input_path:
            raise ThreeStockInputBatchError(f"job {index} input_path is invalid")
        if (
            not isinstance(expected_sha256, str)
            or _HASH.fullmatch(expected_sha256) is None
        ):
            raise ThreeStockInputBatchError(f"job {index} input_sha256 is invalid")
        normalized_path = Path(input_path)
        if not normalized_path.is_absolute():
            normalized_path = manifest_path.parent / normalized_path
        jobs.append(
            {
                "job_id": job_id,
                "input_path": normalized_path.resolve(),
                "input_sha256": expected_sha256,
            }
        )
    return sorted(jobs, key=lambda row: row["job_id"])


def _preflight_jobs(jobs: list[dict[str, Any]]) -> None:
    for row in jobs:
        path = row["input_path"]
        if not path.is_file():
            raise ThreeStockInputBatchError(
                f"input file is unavailable for job {row['job_id']}"
            )
        if sha256_file(path) != row["input_sha256"]:
            raise ThreeStockInputBatchError(
                f"input hash drifted for job {row['job_id']}"
            )


def _rewrite_child_for_final_destination(
    child_manifest: dict[str, Any],
    *,
    child_stage: Path,
    final_child: Path,
    job_id: str,
    expected_input_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if child_manifest.get("schema_version") != "neuro-film.three-stock-file-batch.v1":
        raise ThreeStockInputBatchError("child manifest schema drifted")
    rows = child_manifest.get("rows")
    if (
        not isinstance(rows, list)
        or tuple(row.get("style_id") for row in rows if isinstance(row, dict))
        != _EXPECTED_STYLES
    ):
        raise ThreeStockInputBatchError("child manifest style order drifted")
    receipt_rows: list[dict[str, Any]] = []
    for row in rows:
        style = row["style_id"]
        output_name = f"{style}.png"
        recipe_name = f"{style}.recipe.json"
        staged_output = child_stage / output_name
        staged_recipe = child_stage / recipe_name
        recipe = json.loads(staged_recipe.read_text(encoding="utf-8"))
        if recipe.get("input", {}).get("sha256") != expected_input_sha256:
            raise ThreeStockInputBatchError(
                f"child recipe input identity drifted for job {job_id}"
            )
        recipe["output"]["path"] = str((final_child / output_name).resolve())
        validate_render_recipe(recipe)
        recipe_sha256 = atomic_write_json(staged_recipe, recipe)
        row["output_path"] = str((final_child / output_name).resolve())
        row["output_sha256"] = sha256_file(staged_output)
        row["recipe_path"] = str((final_child / recipe_name).resolve())
        row["recipe_sha256"] = recipe_sha256
        receipt_rows.append(
            {
                "film_stock_id": row["film_stock_id"],
                "style_id": style,
                "output_path": PurePosixPath(job_id, output_name).as_posix(),
                "output_sha256": row["output_sha256"],
                "recipe_path": PurePosixPath(job_id, recipe_name).as_posix(),
                "recipe_sha256": recipe_sha256,
            }
        )
    child_manifest_path = child_stage / "batch.json"
    atomic_write_json(child_manifest_path, child_manifest)
    return child_manifest, receipt_rows


def _publish_no_replace(stage: Path, destination: Path) -> None:
    os.rename(stage, destination)


def render_three_stock_input_batch_to_directory(
    manifest_path: Path,
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
) -> dict[str, Any]:
    """Render a complete hash-bound input set and publish one directory."""

    if os.name != "nt":
        raise ThreeStockInputBatchError("multi-input publication requires Windows")
    manifest_path = Path(manifest_path)
    output_directory = Path(output_directory)
    root = Path(root)
    if not manifest_path.is_file():
        raise ThreeStockInputBatchError("manifest_path must be an existing file")
    if output_directory.exists() or output_directory.is_symlink():
        raise ThreeStockInputBatchError("output_directory must not already exist")
    if not output_directory.parent.is_dir():
        raise ThreeStockInputBatchError("output_directory parent must exist")
    for path, label in (
        (profile_path, "profile"),
        (statistics_path, "statistics"),
        (guardrails_path, "guardrails"),
    ):
        if not Path(path).is_file():
            raise ThreeStockInputBatchError(f"{label} file is unavailable")

    jobs = _load_jobs(manifest_path)
    _preflight_jobs(jobs)
    profile_sha256 = sha256_file(profile_path)
    statistics_sha256 = sha256_file(statistics_path)
    guardrails_sha256 = sha256_file(guardrails_path)
    software_commit = _software_commit(root)

    stage = output_directory.with_name(
        f".{output_directory.name}.{os.getpid()}.{uuid.uuid4().hex}.stage"
    )
    stage.mkdir()
    published = False
    try:
        receipt_jobs: list[dict[str, Any]] = []
        for job in jobs:
            job_id = job["job_id"]
            child_stage = stage / job_id
            child_manifest = render_three_stock_batch_to_directory(
                job["input_path"],
                child_stage,
                root=root,
                profile_path=profile_path,
                statistics_path=statistics_path,
                guardrails_path=guardrails_path,
                look_amount=look_amount,
                seed=seed,
                tile_size=tile_size,
                tile_workers=tile_workers,
                png_compression=png_compression,
                software_commit=software_commit,
            )
            if child_manifest.get("input_sha256") != job["input_sha256"]:
                raise ThreeStockInputBatchError(
                    f"child input identity drifted for job {job_id}"
                )
            final_child = output_directory / job_id
            child_manifest, child_rows = _rewrite_child_for_final_destination(
                child_manifest,
                child_stage=child_stage,
                final_child=final_child,
                job_id=job_id,
                expected_input_sha256=job["input_sha256"],
            )
            child_manifest_sha256 = sha256_file(child_stage / "batch.json")
            receipt_jobs.append(
                {
                    "job_id": job_id,
                    "input_sha256": job["input_sha256"],
                    "child_manifest_path": PurePosixPath(
                        job_id, "batch.json"
                    ).as_posix(),
                    "child_manifest_sha256": child_manifest_sha256,
                    "rows": child_rows,
                }
            )

        identity = {
            "schema_version": RECEIPT_SCHEMA,
            "job_count": len(receipt_jobs),
            "profile_sha256": profile_sha256,
            "statistics_sha256": statistics_sha256,
            "guardrails_sha256": guardrails_sha256,
            "look_amount": float(look_amount),
            "seed": seed,
            "tile_size": tile_size,
            "tile_workers": tile_workers,
            "png_compression": png_compression,
            "jobs": receipt_jobs,
            "claim_ceiling": (
                "Three deterministic film-inspired Look Approximations per input; "
                "calibrated or physical stock response is not established."
            ),
        }
        receipt = {"batch_id": _canonical_sha256(identity), **identity}
        atomic_write_json(stage / "batch.json", receipt)
        _preflight_jobs(jobs)
        if (
            sha256_file(profile_path) != profile_sha256
            or sha256_file(statistics_path) != statistics_sha256
            or sha256_file(guardrails_path) != guardrails_sha256
        ):
            raise ThreeStockInputBatchError("render configuration drifted")
        if _software_commit(root) != software_commit:
            raise ThreeStockInputBatchError(
                "software commit drifted before publication"
            )
        if output_directory.exists() or output_directory.is_symlink():
            raise ThreeStockInputBatchError(
                "output_directory appeared during publication"
            )
        _publish_no_replace(stage, output_directory)
        published = True
        return receipt
    except Exception:
        if not published and stage.exists():
            shutil.rmtree(stage)
        raise


__all__ = [
    "INPUT_MANIFEST_SCHEMA",
    "MAXIMUM_JOBS",
    "RECEIPT_SCHEMA",
    "ThreeStockInputBatchError",
    "render_three_stock_input_batch_to_directory",
]
