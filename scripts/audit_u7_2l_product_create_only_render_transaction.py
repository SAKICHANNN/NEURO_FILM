"""Formal audit for the U7.2L product primary-image transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_render_transaction import (
    ProductRenderTransactionError,
    preflight_product_primary_output,
)
from src.preprocess import output_encode

SCRIPT = ROOT / "scripts/render_film.py"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
CONTRACT_COMMIT = "04a1ee41"
IMPLEMENTATION_COMMIT = "efdca7e7"
IMPLEMENTATION_PATHS = (
    "scripts/render_film.py",
    "src/preprocess/output_encode.py",
    "src/inference/product_render_transaction.py",
    "tests/test_u7_2l_product_create_only_render_transaction.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _command(source: Path, output: Path, style: str = "velvia_50") -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        str(source),
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--output",
        str(output),
    ]


def _run(
    source: Path, output: Path, style: str = "velvia_50"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(source, output, style),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _owned_stages(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob(".*.tmp"))


def _windows_junction_gate(work: Path) -> bool:
    if os.name != "nt":
        return True
    target = work / "junction-target"
    target.mkdir()
    destination = work / "reparse-output.png"
    target_arg = str(target).replace("'", "''")
    destination_arg = str(destination).replace("'", "''")
    created = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$null = New-Item -ItemType Junction "
                f"-Path '{destination_arg}' -Target '{target_arg}'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        return False
    try:
        completed = _run(work / "missing-reparse-input.png", destination)
        return (
            completed.returncode != 0
            and "product output destination must not already exist" in completed.stderr
            and destination.is_dir()
        )
    finally:
        destination.rmdir()


def _late_destination_gate(work: Path) -> bool:
    destination = work / "late.png"
    foreign = b"late-foreign-destination"
    real_publish = output_encode.publish_create_only

    def inject(stage: Path, final: Path):
        final.write_bytes(foreign)
        return real_publish(stage, final)

    output_encode.publish_create_only = inject
    try:
        try:
            output_encode.save_srgb8(
                np.zeros((3, 4, 3), dtype=np.float32),
                destination,
                create_only=True,
            )
        except FileExistsError:
            pass
        else:
            return False
    finally:
        output_encode.publish_create_only = real_publish
    return destination.read_bytes() == foreign and not _owned_stages(work)


def _encoder_failure_gate(work: Path) -> bool:
    destination = work / "encoder-failure.png"

    def fail_save(*_args, **_kwargs) -> None:
        raise RuntimeError("u7.2l injected encoder failure")

    with patch.object(Image.Image, "save", fail_save):
        try:
            output_encode.save_srgb8(
                np.zeros((3, 4, 3), dtype=np.float32),
                destination,
                create_only=True,
            )
        except RuntimeError as exc:
            if str(exc) != "u7.2l injected encoder failure":
                return False
        else:
            return False
    return not destination.exists() and not _owned_stages(work)


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="neuro-film-u7-2l-") as raw_work:
        work = Path(raw_work)
        source = work / "source.png"
        _source(source)
        source_before = source.read_bytes()

        product_results: dict[str, dict[str, Any]] = {}
        for style in order:
            output = work / f"{style}.png"
            completed = _run(source, output, style)
            product_results[style] = {
                "returncode": completed.returncode,
                "output_sha256": _sha256(output) if output.exists() else None,
                "source_immutable": source.read_bytes() == source_before,
                "owned_stage_count": len(_owned_stages(work)),
            }

        same = work / "same-missing.png"
        same_run = _run(same, same)
        regular = work / "regular.png"
        regular.write_bytes(b"foreign-regular")
        regular_run = _run(work / "missing-regular-input.png", regular)
        hardlink = work / "hardlink.png"
        os.link(regular, hardlink)
        hardlink_run = _run(work / "missing-hardlink-input.png", hardlink)

        with patch("pathlib.Path.lstat", return_value=os.stat_result((0,) * 10)):
            try:
                preflight_product_primary_output(
                    work / "semantic-input.png", work / "semantic-output.png"
                )
            except ProductRenderTransactionError:
                semantic_lstat_reject = True
            else:
                semantic_lstat_reject = False

        concurrent = work / "concurrent.png"
        processes = [
            subprocess.Popen(
                _command(source, concurrent),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        for process in processes:
            process.communicate(timeout=60)
        concurrent_codes = sorted(int(process.returncode) for process in processes)

        legacy = work / "legacy.png"
        legacy.write_bytes(b"replace-me")
        legacy_run = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(source),
                "--style",
                "velvia_50",
                "--use-render-profile",
                "--render-profile",
                str(LEGACY_PROFILE),
                "--output",
                str(legacy),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        expected = config["prechange_output_sha256"]
        product_exact = all(
            product_results[style]["returncode"] == 0
            and product_results[style]["output_sha256"] == expected[style]
            and product_results[style]["source_immutable"]
            and product_results[style]["owned_stage_count"] == 0
            for style in config["available_product_looks"]
        )
        prechange_commit = subprocess.check_output(
            ["git", "rev-parse", f"{CONTRACT_COMMIT}^"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        source_locks = config["source_locks"]
        prechange_locks_exact = all(
            hashlib.sha256(_git_blob(prechange_commit, path)).hexdigest()
            == source_locks[key]
            for key, path in (
                ("render_film_sha256", "scripts/render_film.py"),
                ("output_encode_sha256", "src/preprocess/output_encode.py"),
                ("create_only_file_sha256", "src/film_physics/create_only_file.py"),
                (
                    "product_profile_sha256",
                    "configs/render_profiles/safe_rich_product_v1.json",
                ),
                (
                    "legacy_profile_sha256",
                    "configs/render_profiles/safe_rich_v1.json",
                ),
            )
        )
        implementation_exact = all(
            hashlib.sha256(_git_blob(IMPLEMENTATION_COMMIT, path)).hexdigest()
            == _sha256(ROOT / path)
            for path in IMPLEMENTATION_PATHS
        )
        gates = {
            "prechange_source_locks_exact": prechange_locks_exact,
            "implementation_files_exact": implementation_exact,
            "fixture_exact": hashlib.sha256(source_before).hexdigest()
            == config["fixture"]["sha256"],
            "same_path_rejects_before_decode": same_run.returncode != 0
            and "product output must not identify the input path" in same_run.stderr
            and not same.exists(),
            "existing_regular_rejects_before_decode": regular_run.returncode != 0
            and regular.read_bytes() == b"foreign-regular",
            "hardlink_alias_rejects_before_decode": hardlink_run.returncode != 0
            and hardlink.read_bytes() == b"foreign-regular",
            "symlink_entry_semantics_reject": semantic_lstat_reject,
            "windows_reparse_entry_rejects": _windows_junction_gate(work),
            "late_foreign_destination_preserved": _late_destination_gate(work),
            "encoder_failure_residue_zero": _encoder_failure_gate(work),
            "concurrent_exactly_one_success": concurrent_codes == [0, 1]
            and _sha256(concurrent) == expected["velvia_50"],
            "three_product_outputs_byte_exact": product_exact,
            "source_immutable": source.read_bytes() == source_before,
            "legacy_replace_behavior_exact": legacy_run.returncode == 0
            and _sha256(legacy) == expected["legacy_velvia_50"],
            "owned_stage_residue_zero": not _owned_stages(work),
        }
        report = {
            "schema_id": "neuro-film.u7-2l-product-create-only-render-transaction-result.v1",
            "experiment_id": "U7.2L",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "bindings": {
                "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "contract_commit": CONTRACT_COMMIT,
                "implementation_commit": IMPLEMENTATION_COMMIT,
            },
            "product_results": dict(sorted(product_results.items())),
            "concurrent_returncodes": concurrent_codes,
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "decision": (
                "Enable create-only primary-image publication only for the exact "
                "safe-rich-product-v1 profile; retain legacy publication semantics."
            ),
            "claim_ceiling": config["claim_ceiling"],
        }
    report["owned_runtime_residue_zero"] = not Path(raw_work).exists()
    if not report["owned_runtime_residue_zero"]:
        report["status"] = "FAIL_CLOSED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_2l_product_create_only_render_transaction_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    order = tuple(config["available_product_looks"])
    if args.order == "reverse":
        order = tuple(reversed(order))
    report = build_report(config_path=args.config, order=order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
