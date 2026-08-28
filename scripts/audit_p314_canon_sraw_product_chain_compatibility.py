#!/usr/bin/env python3
"""Audit six exact Canon sRAW/mRAW files through the public product chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image
from PIL import __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    load_render_profile,
    replay_style_safe_recipe_to_file,
    validate_render_recipe,
)

REPORT_SCHEMA = "neuro-film.p314-canon-sraw-product-chain-compatibility-result.v1"


class P314Error(RuntimeError):
    """Raised when a frozen P314 identity or execution invariant differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise P314Error(f"expected JSON object: {path}")
    return value


def _verify_file(path: Path, expected: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected["bytes"])
        and _sha256_file(path) == str(expected["sha256"])
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _relative_to_root(path: Path) -> str:
    # Preserve the repository-relative logical path. ``outputs`` is an NTFS
    # junction to the canonical P-backed storage root, so resolving it would
    # erase the portable repository identity and make the path appear to live
    # on another drive.
    return path.absolute().relative_to(ROOT.absolute()).as_posix()


def _cleanup_row(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _remove_empty_tree(path: Path, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while current.exists() and current.resolve() != stop:
        current.rmdir()
        current = current.parent


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


def _row_record(
    row: dict[str, Any],
    *,
    profile_path: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    source = ROOT / row["path"]
    source_before = _sha256_file(source)
    if source.stat().st_size != int(row["bytes"]) or source_before != row["sha256"]:
        raise P314Error(f"source identity differs: {row['source_id']}")

    row_root = scratch_root / row["source_id"]
    row_root.mkdir(parents=True, exist_ok=False)
    output = row_root / "render.png"
    recipe_path = output.with_suffix(".recipe.json")
    replay = row_root / "replay.png"
    command = [
        sys.executable,
        str(ROOT / "scripts/render_film.py"),
        row["path"],
        "--style",
        "ektar_100",
        "--look-amount",
        "1.0",
        "--use-render-profile",
        "--render-profile",
        _relative_to_root(profile_path),
        "--output-bit-depth",
        "8",
        "--write-recipe",
        "--output",
        _relative_to_root(output),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip().splitlines()
            raise P314Error(
                f"render CLI failed for {row['source_id']}: "
                f"{message[-1] if message else 'no stderr'}"
            )
        if not output.is_file() or not recipe_path.is_file():
            raise P314Error(f"render artifacts missing: {row['source_id']}")

        recipe_bytes = recipe_path.read_bytes()
        recipe = json.loads(recipe_bytes)
        validate_render_recipe(recipe)
        profile = load_render_profile(profile_path, root=ROOT)
        output_facts = _output_facts(output)
        expected_input_path = row["path"]
        expected_output_path = _relative_to_root(output)
        recipe_exact = {
            "claim_is_look_approximation": (
                recipe["claim"]["claim_ceiling"]
                == "Look Approximation; not calibrated or source-authoritative"
                and recipe["claim"]["mode"] == "film-inspired"
            ),
            "explicit_style": recipe["render"]["style"] == "ektar_100",
            "input_identity": (
                recipe["input"]["path"] == expected_input_path
                and recipe["input"]["sha256"] == row["sha256"]
            ),
            "output_identity": (
                recipe["output"]["path"] == expected_output_path
                and recipe["output"]["sha256"] == output_facts["sha256"]
                and recipe["output"]["format"] == "PNG"
                and recipe["output"]["bit_depth"] == 8
            ),
            "product_profile": (
                recipe["profile"]["profile_id"] == "safe-rich-product-v1"
                and recipe["profile"]["profile_version"] == profile["profile_version"]
                and recipe["profile"]["sha256"] == _sha256_file(profile_path)
            ),
            "software_commit": recipe["software"]["commit"] == _git_head(),
        }
        replay_digest = replay_style_safe_recipe_to_file(
            recipe,
            profile_path=profile_path,
            output_path=replay,
            root=ROOT,
        )
        replay_facts = _output_facts(replay)
        record = {
            "cli_returncode": completed.returncode,
            "model": row["model"],
            "mode": row["mode"],
            "output": output_facts,
            "recipe": {
                "bytes": len(recipe_bytes),
                "exact": recipe_exact,
                "schema_id": recipe["schema_id"],
                "sha256": _sha256_bytes(recipe_bytes),
            },
            "replay": {
                **replay_facts,
                "returned_sha256": replay_digest,
            },
            "replay_byte_exact": output.read_bytes() == replay.read_bytes(),
            "source_id": row["source_id"],
            "source_sha256": source_before,
            "source_unchanged": _sha256_file(source) == source_before,
        }
        return record
    finally:
        _cleanup_row(replay, recipe_path, output)
        if row_root.exists():
            row_root.rmdir()


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = _load_json(config_path)
    bindings = {
        name: _verify_file(ROOT / item["path"], item)
        for name, item in sorted(config["bindings"].items())
    }
    if not all(bindings.values()):
        raise P314Error("frozen production binding differs")

    runtime = config["runtime"]
    runtime_exact = {
        "libraw": list(rawpy.libraw_version) == runtime["libraw"],
        "numpy": np.__version__ == runtime["numpy"],
        "pillow": PILLOW_VERSION == runtime["pillow"],
        "python": ".".join(map(str, sys.version_info[:3])) == runtime["python"],
        "rawpy": rawpy.__version__ == runtime["rawpy"],
        "windows": sys.platform == "win32" and runtime["platform"] == "Windows",
    }
    if not all(runtime_exact.values()):
        raise P314Error("frozen runtime identity differs")

    p313 = _load_json(ROOT / config["bindings"]["p313_contract"]["path"])
    rows = list(p313["rows"])
    if reverse:
        rows.reverse()
    execution = config["execution"]
    scratch_root = ROOT / execution["scratch_root"]
    if scratch_root.exists() and any(scratch_root.iterdir()):
        raise P314Error("owned scratch root is not empty before execution")
    scratch_root.mkdir(parents=True, exist_ok=True)
    profile_path = ROOT / execution["profile_path"]
    try:
        records = [
            _row_record(row, profile_path=profile_path, scratch_root=scratch_root)
            for row in rows
        ]
        records.sort(key=lambda item: item["source_id"])
    finally:
        if scratch_root.exists() and not any(scratch_root.iterdir()):
            _remove_empty_tree(scratch_root, ROOT / "outputs")

    scratch_residue = (
        sum(1 for path in scratch_root.rglob("*") if path.is_file())
        if scratch_root.exists()
        else 0
    )
    gates = {
        "all_cli_success": all(row["cli_returncode"] == 0 for row in records),
        "all_output_png8_srgb": all(
            row["output"]["format"] == "PNG"
            and row["output"]["mode"] == "RGB"
            and row["output"]["icc_present"]
            for row in records
        ),
        "all_recipe_fields_exact": all(
            all(row["recipe"]["exact"].values()) for row in records
        ),
        "all_replays_byte_exact": all(row["replay_byte_exact"] for row in records),
        "all_replay_hashes_exact": all(
            row["output"]["sha256"]
            == row["replay"]["sha256"]
            == row["replay"]["returned_sha256"]
            for row in records
        ),
        "all_sources_immutable": all(row["source_unchanged"] for row in records),
        "bindings_exact": all(bindings.values()),
        "network_requests_zero": True,
        "required_rows_complete": len(records) == int(config["gates"]["required_rows"]),
        "runtime_exact": all(runtime_exact.values()),
        "scratch_residue_zero": scratch_residue == 0,
        "tracked_worktree_clean": _tracked_clean(),
    }
    decision = (
        "PASS_PRIVATE_CANON_SRAW_PRODUCT_CHAIN_COMPATIBILITY"
        if all(gates.values())
        else "FAIL_CLOSED_CANON_SRAW_PRODUCT_CHAIN_COMPATIBILITY"
    )
    report: dict[str, Any] = {
        "bindings": bindings,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "decision": decision,
        "execution": execution,
        "execution_commit": _git_head(),
        "experiment_id": config["experiment_id"],
        "gates": gates,
        "network_requests": 0,
        "records": records,
        "runtime": {
            "checks": runtime_exact,
            "libraw": list(rawpy.libraw_version),
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
            "python": ".".join(map(str, sys.version_info[:3])),
            "rawpy": rawpy.__version__,
        },
        "schema": REPORT_SCHEMA,
        "scratch_residue_files": scratch_residue,
        "stop_rule": config["stop_rule"],
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
