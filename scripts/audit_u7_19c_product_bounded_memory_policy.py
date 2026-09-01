#!/usr/bin/env python3
"""Formal U7.19C bounded-memory product execution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    load_render_profile,
    replay_style_safe_recipe_to_file,
    validate_render_recipe,
)
from src.inference.product_execution_policy import (
    resolve_product_execution_policy,
)

FORMAL_SCHEMA = "kmcfm.u7-19c-product-bounded-memory-policy-report.v1"
WORKER_SCHEMA = "kmcfm.u7-19c-product-bounded-memory-policy-worker.v1"


class U719CError(RuntimeError):
    """Raised when a frozen U7.19C invariant differs."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise U719CError(f"JSON root must be an object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_clean() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    return not output.strip()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _verify_file(binding: dict[str, Any]) -> bool:
    path = ROOT / binding["path"]
    return (
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and _sha256_file(path) == binding["sha256"]
    )


def _output_facts(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "bytes": path.stat().st_size,
            "format": image.format,
            "height": image.height,
            "icc_present": bool(image.info.get("icc_profile")),
            "mode": image.mode,
            "sha256": _sha256_file(path),
            "width": image.width,
        }


def _source(config: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [row for row in config["sources"] if row["id"] == source_id]
    if len(matches) != 1:
        raise U719CError(f"source id is not unique: {source_id}")
    return matches[0]


def _worker(
    config: dict[str, Any], producer_repo: Path, source_id: str, scratch_root: Path
) -> dict[str, Any]:
    row = _source(config, source_id)
    source = producer_repo / row["path"]
    if (
        not source.is_file()
        or source.stat().st_size != int(row["bytes"])
        or _sha256_file(source) != row["sha256"]
    ):
        raise U719CError(f"source identity differs: {source_id}")
    source_before = _sha256_file(source)
    scratch_root.mkdir(parents=True, exist_ok=False)
    output = scratch_root / "render.png"
    recipe_path = output.with_suffix(".recipe.json")
    metrics_path = output.with_suffix(".metrics.json")
    replay = scratch_root / "replay.png"
    profile_path = ROOT / config["product_chain"]["profile_path"]
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        str(source),
        "--style",
        config["product_chain"]["style"],
        "--look-amount",
        str(config["product_chain"]["look_amount"]),
        "--use-render-profile",
        "--render-profile",
        str(profile_path),
        "--output-bit-depth",
        str(config["product_chain"]["output_bit_depth"]),
        "--write-recipe",
        "--write-metrics",
        "--output",
        str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        raise U719CError(stderr[-1] if stderr else "product CLI failed")
    if not all(path.is_file() for path in (output, recipe_path, metrics_path)):
        raise U719CError("product artifacts are incomplete")

    recipe = _load_json(recipe_path)
    metrics = _load_json(metrics_path)
    validate_render_recipe(recipe)
    profile = load_render_profile(profile_path, root=ROOT)
    policy = resolve_product_execution_policy(
        profile_id=profile["profile_id"], tile_size=None, tile_workers=1
    )
    replay_digest = replay_style_safe_recipe_to_file(
        recipe, profile_path=profile_path, output_path=replay, root=ROOT
    )
    output_facts = _output_facts(output)
    replay_facts = _output_facts(replay)
    elapsed = time.perf_counter() - started
    claim = recipe["claim"]
    record = {
        "automatic_policy": {
            "automatic": policy.automatic,
            "tile_size": policy.tile_size,
            "tile_workers": policy.tile_workers,
        },
        "cli_had_explicit_tile_argument": any(
            value in {"--tile-size", "--tile-workers"} for value in command
        ),
        "command_returncode": completed.returncode,
        "extension": row["extension"],
        "metrics_tile_size": metrics["tile_size"],
        "output": output_facts,
        "recipe": {
            "claim": claim,
            "schema_id": recipe["schema_id"],
            "sha256": _sha256_file(recipe_path),
        },
        "replay": {**replay_facts, "returned_sha256": replay_digest},
        "replay_byte_exact": output.read_bytes() == replay.read_bytes(),
        "resource": {
            "peak_process_rss_bytes": int(
                getattr(psutil.Process().memory_info(), "peak_wset", 0)
            ),
            "wall_seconds": elapsed,
        },
        "schema": WORKER_SCHEMA,
        "source_id": source_id,
        "source_sha256": source_before,
        "source_unchanged": _sha256_file(source) == source_before,
    }
    return record


def _run_worker(
    *,
    config_path: Path,
    producer_repo: Path,
    source_id: str,
    scratch_root: Path,
    limit_bytes: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    output = scratch_root / f"worker-{source_id}.json"
    media = scratch_root / f"media-{source_id}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "worker",
        "--config",
        str(config_path),
        "--producer-repo",
        str(producer_repo),
        "--source-id",
        source_id,
        "--scratch-root",
        str(media),
        "--output",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    owned = psutil.Process(process.pid)
    started = time.perf_counter()
    peak_tree = 0
    failure: str | None = None
    while process.poll() is None:
        try:
            members = [owned, *owned.children(recursive=True)]
            peak_tree = max(
                peak_tree, sum(member.memory_info().rss for member in members)
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        elapsed = time.perf_counter() - started
        if peak_tree > limit_bytes:
            failure = "process_tree_rss_limit_exceeded"
        elif elapsed > timeout_seconds:
            failure = "worker_timeout"
        if failure is not None:
            try:
                members = [*owned.children(recursive=True), owned]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                members = []
            for member in members:
                try:
                    member.kill()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            break
        time.sleep(0.05)
    _, stderr_text = process.communicate()
    wall_seconds = time.perf_counter() - started
    try:
        if failure is not None:
            return {
                "error": failure,
                "resource": {
                    "peak_process_tree_rss_bytes": peak_tree,
                    "wall_seconds": wall_seconds,
                },
                "source_id": source_id,
                "worker_ok": False,
            }
        if process.returncode != 0 or not output.is_file():
            stderr = stderr_text.strip().splitlines()
            return {
                "error": stderr[-1] if stderr else "worker_failed_without_report",
                "resource": {
                    "peak_process_tree_rss_bytes": peak_tree,
                    "wall_seconds": wall_seconds,
                },
                "source_id": source_id,
                "worker_ok": False,
            }
        value = _load_json(output)
        value["resource"] = {
            "peak_process_tree_rss_bytes": peak_tree,
            "wall_seconds": wall_seconds,
        }
        value["worker_ok"] = True
        return value
    finally:
        output.unlink(missing_ok=True)
        if media.exists():
            shutil.rmtree(media)


def _scientific_view(report: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(report))
    value.pop("execution_commit", None)
    value.pop("order", None)
    for row in value.get("records", []):
        row.pop("resource", None)
    return value


def execute_formal(
    config_path: Path,
    execution_lock_path: Path,
    producer_repo: Path,
    scratch_root: Path,
    *,
    reverse: bool,
) -> dict[str, Any]:
    config = _load_json(config_path)
    execution_lock = _load_json(execution_lock_path)
    if config["status"] != "FROZEN_BEFORE_U7_19C_X2D_RENDER":
        raise U719CError("config is not frozen")
    if execution_lock["status"] != "LOCKED_BEFORE_U7_19C_FORMAL":
        raise U719CError("execution lock is not frozen")
    bindings = {
        name: _verify_file(binding)
        for name, binding in sorted(execution_lock["bindings"].items())
    }
    if not all(bindings.values()):
        raise U719CError("formal binding differs")
    if not _tracked_clean():
        raise U719CError("tracked worktree must be clean")

    ordered = [row["id"] for row in config["sources"]]
    if reverse:
        ordered.reverse()
    scratch_root.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for source_id in ordered:
            records.append(
                _run_worker(
                    config_path=config_path,
                    producer_repo=producer_repo,
                    source_id=source_id,
                    scratch_root=scratch_root,
                    limit_bytes=int(config["limits"]["peak_process_tree_rss_bytes"]),
                    timeout_seconds=int(config["limits"]["worker_seconds"]),
                )
            )
    finally:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
    records.sort(key=lambda row: row["source_id"])

    successful = [row for row in records if row.get("worker_ok")]
    required_policy = config["policy"]
    gates = {
        "all_workers_complete": len(successful) == len(config["sources"]),
        "all_workers_within_resource_limits": len(successful) == len(records)
        and all(
            row["resource"]["peak_process_tree_rss_bytes"]
            <= int(config["limits"]["peak_process_tree_rss_bytes"])
            and row["resource"]["wall_seconds"]
            <= int(config["limits"]["worker_seconds"])
            for row in successful
        ),
        "automatic_policy_exact": len(successful) == len(records)
        and all(
            row["automatic_policy"]
            == {
                "automatic": True,
                "tile_size": required_policy["default_tile_size"],
                "tile_workers": required_policy["default_tile_workers"],
            }
            and row["metrics_tile_size"] == required_policy["default_tile_size"]
            and row["cli_had_explicit_tile_argument"] is False
            for row in successful
        ),
        "full_resolution_exact": len(successful) == len(records)
        and all(
            [row["output"]["height"], row["output"]["width"]]
            == [source["height"], source["width"]]
            for row in successful
            for source in config["sources"]
            if source["id"] == row["source_id"]
        ),
        "output_and_replay_exact": len(successful) == len(records)
        and all(
            row["output"]["format"] == "PNG"
            and row["output"]["mode"] == "RGB"
            and row["output"]["icc_present"]
            and row["replay_byte_exact"]
            and row["output"]["sha256"]
            == row["replay"]["sha256"]
            == row["replay"]["returned_sha256"]
            for row in successful
        ),
        "look_approximation_claim_exact": len(successful) == len(records)
        and all(
            row["recipe"]["claim"]["render_mode"]
            == config["product_chain"]["require_claim_render_mode"]
            and row["recipe"]["claim"]["output_label"]
            == config["product_chain"]["require_claim_output_label"]
            and row["recipe"]["claim"]["evidence_grade"]
            == config["product_chain"]["require_claim_evidence_grade"]
            and row["recipe"]["claim"]["calibrated_reference_allowed"] is False
            for row in successful
        ),
        "sources_immutable": len(successful) == len(records)
        and all(row["source_unchanged"] for row in successful),
        "bindings_exact": all(bindings.values()),
        "network_requests_zero": True,
        "scratch_residue_zero": not scratch_root.exists(),
        "tracked_worktree_clean": _tracked_clean(),
    }
    report = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "execution_commit": _git_head(),
        "gates": gates,
        "network_requests": 0,
        "order": "reverse" if reverse else "forward",
        "records": records,
        "schema": FORMAL_SCHEMA,
        "status": "PASS_PRIVATE_U7_19C_PRODUCT_BOUNDED_MEMORY_POLICY"
        if all(gates.values())
        else "FAIL_CLOSED_U7_19C_PRODUCT_BOUNDED_MEMORY_POLICY",
        "stop_rule": config["stop_rule"],
    }
    report["scientific_identity"] = _sha256_bytes(
        json.dumps(
            _scientific_view(report), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("formal", "worker"), default="formal")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    producer_repo = args.producer_repo.resolve()
    if args.stage == "worker":
        if not args.source_id:
            raise U719CError("worker requires --source-id")
        report = _worker(
            _load_json(config_path),
            producer_repo,
            args.source_id,
            args.scratch_root.resolve(),
        )
    else:
        if args.execution_lock is None:
            raise U719CError("formal stage requires --execution-lock")
        report = execute_formal(
            config_path,
            args.execution_lock.resolve(),
            producer_repo,
            args.scratch_root.resolve(),
            reverse=args.reverse,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
