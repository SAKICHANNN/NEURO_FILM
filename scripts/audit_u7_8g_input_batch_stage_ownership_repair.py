"""Committed-head audit for U7.8G outer batch-stage ownership repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import three_stock_input_batch as batch_module
from src.inference.render_contract import sha256_file
from src.inference.three_stock_input_batch import (
    INPUT_MANIFEST_SCHEMA,
    render_three_stock_input_batch_to_directory,
)

CONFIG = ROOT / "configs/u7_8g_input_batch_stage_ownership_repair_v1.json"
PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
STATISTICS = ROOT / "configs/film_color_stats.json"
GUARDRAILS = ROOT / "configs/color_guardrails.json"
SCRATCH = ROOT / "tmp/u7_8g_input_batch_stage_ownership_repair"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


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


def _render(manifest: Path, destination: Path) -> dict[str, Any]:
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


def _replacement_control(root: Path, manifest: Path) -> dict[str, Any]:
    destination = root / "published"
    witness: dict[str, Path] = {}

    def replace_then_fail(_input: Path, child_stage: Path, **_kwargs: Any):
        stage = Path(child_stage).parent
        owned_backup = stage.with_name(f"{stage.name}.owned-backup")
        stage.rename(owned_backup)
        stage.mkdir()
        foreign = stage / "foreign.bin"
        foreign.write_bytes(b"foreign-owner")
        witness.update(stage=stage, owned_backup=owned_backup, foreign=foreign)
        raise RuntimeError("injected stage identity replacement")

    rejected = False
    try:
        with patch.object(
            batch_module,
            "render_three_stock_batch_to_directory",
            replace_then_fail,
        ):
            _render(manifest, destination)
    except RuntimeError as exc:
        if "stage identity replacement" not in str(exc):
            raise
        rejected = True
    return {
        "failure_rejected": rejected,
        "foreign_stage_preserved": witness["stage"].is_dir(),
        "foreign_payload_preserved": (
            witness["foreign"].read_bytes() == b"foreign-owner"
        ),
        "displaced_owned_stage_preserved": witness["owned_backup"].is_dir(),
        "final_destination_absent": not destination.exists(),
    }


def _ordinary_failure_control(root: Path, manifest: Path) -> dict[str, Any]:
    destination = root / "published"

    def fail_child(*_args: Any, **_kwargs: Any):
        raise RuntimeError("ordinary child failure")

    rejected = False
    try:
        with patch.object(
            batch_module,
            "render_three_stock_batch_to_directory",
            fail_child,
        ):
            _render(manifest, destination)
    except RuntimeError as exc:
        if "ordinary child failure" not in str(exc):
            raise
        rejected = True
    return {
        "failure_rejected": rejected,
        "owned_stage_residue_count": len(list(root.glob(".published.*.stage"))),
        "final_destination_absent": not destination.exists(),
    }


def _success_control(
    root: Path,
    manifest: Path,
    expected_png: Mapping[str, str],
) -> dict[str, Any]:
    destination = root / "published"
    first = _render(manifest, destination)
    first_receipt = (destination / "batch.json").read_bytes()
    first_png = {
        path.name: sha256_file(path)
        for path in sorted((destination / "source").glob("*.png"))
    }
    shutil.rmtree(destination)
    second = _render(manifest, destination)
    second_receipt = (destination / "batch.json").read_bytes()
    second_png = {
        path.name: sha256_file(path)
        for path in sorted((destination / "source").glob("*.png"))
    }
    return {
        "receipt_replay_exact": first == second
        and first_receipt == second_receipt,
        "receipt_sha256": _sha256(first_receipt),
        "png_sha256": first_png,
        "png_replay_exact": first_png == second_png,
        "png_baseline_exact": first_png == dict(expected_png),
        "style_order": [row["style_id"] for row in first["jobs"][0]["rows"]],
    }


def _forbid_network(*_args: Any, **_kwargs: Any):
    raise AssertionError("network access is forbidden in U7.8G")


def _run_control_order(
    order: str,
    controls: Mapping[str, Callable[[], dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    names = list(controls)
    if order == "reverse":
        names.reverse()
    results = {name: controls[name]() for name in names}
    return {name: results[name] for name in sorted(results)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    implementation = bindings["implementation_commit"]
    source_hashes = bindings["implementation_git_lf_sha256"]
    for path, expected in source_hashes.items():
        if _sha256(_git_bytes(implementation, path)) != expected:
            raise RuntimeError(f"implementation binding drift: {path}")
        if _sha256(_git_bytes("HEAD", path)) != expected:
            raise RuntimeError(f"current implementation drift: {path}")
    if SCRATCH.exists() or SCRATCH.is_symlink():
        raise RuntimeError("U7.8G scratch already exists")
    SCRATCH.mkdir(parents=True)
    try:
        source = SCRATCH / "source.png"
        manifest = SCRATCH / "jobs.json"
        _source(source)
        _manifest(manifest, source)
        if sha256_file(source) != config["baseline"]["input_sha256"]:
            raise RuntimeError("baseline input drift")
        immutable_paths = [source, manifest, CONFIG, PROFILE, STATISTICS, GUARDRAILS]
        before = {str(path): sha256_file(path) for path in immutable_paths}

        controls: dict[str, Callable[[], dict[str, Any]]] = {
            "ordinary_failure": lambda: _ordinary_failure_control(
                SCRATCH / "ordinary", manifest
            ),
            "replacement_failure": lambda: _replacement_control(
                SCRATCH / "replacement", manifest
            ),
            "success": lambda: _success_control(
                SCRATCH / "success",
                manifest,
                config["baseline"]["successful_png_sha256"],
            ),
        }
        for name in controls:
            (SCRATCH / name.replace("_failure", "")).mkdir()
        with patch.object(socket, "create_connection", _forbid_network):
            results = _run_control_order(args.order, controls)

        after = {str(path): sha256_file(path) for path in immutable_paths}
        source_immutable = before == after
        gates = {
            "identity_replaced_stage_preserved": results["replacement_failure"][
                "foreign_stage_preserved"
            ],
            "foreign_payload_preserved": results["replacement_failure"][
                "foreign_payload_preserved"
            ],
            "ordinary_owned_stage_residue_count": results["ordinary_failure"][
                "owned_stage_residue_count"
            ],
            "final_destination_absent_after_failures": results[
                "replacement_failure"
            ]["final_destination_absent"]
            and results["ordinary_failure"]["final_destination_absent"],
            "successful_png_bytes_exact": results["success"]["png_baseline_exact"]
            and results["success"]["png_replay_exact"],
            "successful_receipt_replay_exact": results["success"][
                "receipt_replay_exact"
            ],
            "source_and_configuration_inputs_immutable": source_immutable,
            "network_requests": 0,
        }
        expected_gates = config["gates"]
        passed = all(
            gates[name] == expected for name, expected in expected_gates.items()
        )
        scientific = {
            "schema": config["schema"],
            "node_id": config["node_id"],
            "decision": (
                "PASS_PRIVATE_U7_8G_INPUT_BATCH_STAGE_OWNERSHIP_REPAIR"
                if passed
                else "FAIL_CLOSED_U7_8G_INPUT_BATCH_STAGE_OWNERSHIP_REPAIR"
            ),
            "controls": results,
            "gate_results": gates,
            "claim_ceiling": config["claim_ceiling"],
            "stop_rule": config["stop_rule"],
        }
        report = {
            **scientific,
            "bindings": {
                **bindings,
                "execution_commit": _git_commit(),
                "config_sha256": sha256_file(CONFIG),
            },
            "stable_identity": _sha256(_canonical_json(scientific)),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(_canonical_json(report))
    finally:
        if SCRATCH.exists():
            shutil.rmtree(SCRATCH)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
