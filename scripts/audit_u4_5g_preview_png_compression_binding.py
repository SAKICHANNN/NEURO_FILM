#!/usr/bin/env python3
"""Run the frozen U4.5G preview PNG compression binding audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import (
    ThreeStockPreviewError,
    render_three_stock_previews_to_directory,
)
from src.preprocess import save_srgb8


class U45GError(RuntimeError):
    """Raised when the frozen preview compression audit cannot execute."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_execution_bindings(config: dict[str, Any]) -> bool:
    bindings = config.get("execution_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return False
    return all(
        (ROOT / item["path"]).is_file()
        and (ROOT / item["path"]).stat().st_size == int(item["bytes"])
        and sha256_file(ROOT / item["path"]) == item["sha256"]
        and _git_output("rev-parse", f"{item['commit']}^{{commit}}") == item["commit"]
        and _git_output("rev-parse", f"{item['commit']}:{item['path']}")
        == item["git_blob"]
        for item in bindings.values()
    )


def _semantic_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(manifest))
    for row in normalized["rows"]:
        row["output_path"] = Path(row["output_path"]).name
    return normalized


def _decode_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise U45GError(f"unexpected preview encoding: {path.name}")
        pixels = np.asarray(image, dtype=np.uint8)
        profile = image.info.get("icc_profile", b"")
    return {
        "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "icc_sha256": hashlib.sha256(profile).hexdigest(),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def _run_cli(
    config: dict[str, Any],
    source: Path,
    destination: Path,
    *,
    compression: int | None,
) -> dict[str, Any]:
    render = config["render"]
    command = [
        sys.executable,
        str(ROOT / "scripts/render_three_stock_preview.py"),
        str(source),
        str(destination),
        "--max-preview-pixels",
        str(render["max_preview_pixels"]),
        "--look-amount",
        str(render["look_amount"]),
        "--seed",
        str(render["seed"]),
        "--tile-size",
        str(render["tile_size"]),
        "--tile-workers",
        str(render["tile_workers"]),
        "--jpeg-scaled-decode",
    ]
    if compression is not None:
        command.extend(("--png-compression", str(compression)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if completed.returncode != 0:
        raise U45GError(
            f"preview CLI failed with {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        manifest = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise U45GError("preview CLI emitted invalid JSON") from exc
    rows: dict[str, dict[str, Any]] = {}
    for row in manifest.get("rows", []):
        style_id = str(row["style_id"])
        rows[style_id] = _decode_png(destination / f"{style_id}.preview.png")
        if rows[style_id]["file_sha256"] != row["output_sha256"]:
            raise U45GError("manifest output hash does not match preview bytes")
    if sorted(rows) != ["ektar_100", "portra_400", "velvia_50"]:
        raise U45GError("preview CLI did not emit the exact three look rows")
    return {
        "compression": 6 if compression is None else compression,
        "invocation": "default" if compression is None else f"level-{compression}",
        "manifest": _semantic_manifest(manifest),
        "rows": rows,
    }


def _invalid_controls(source: Path, scratch: Path) -> dict[str, bool]:
    controls: dict[str, bool] = {}
    original_inspector = __import__(
        "src.inference.three_stock_preview", fromlist=["inspect_input"]
    ).inspect_input
    preview_module = sys.modules["src.inference.three_stock_preview"]

    def forbidden_inspection(path: Path) -> object:
        raise AssertionError(f"input inspection occurred for {path}")

    preview_module.inspect_input = forbidden_inspection
    try:
        for value in (-1, 10, True, 3.0):
            name = f"invalid-{str(value).lower().replace('.', '_')}"
            try:
                render_three_stock_previews_to_directory(
                    source,
                    scratch / name,
                    root=ROOT,
                    profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
                    statistics_path=ROOT / "configs/film_color_stats.json",
                    guardrails_path=ROOT / "configs/color_guardrails.json",
                    png_compression=value,  # type: ignore[arg-type]
                )
            except ThreeStockPreviewError:
                controls[name] = not (scratch / name).exists()
            else:
                controls[name] = False
    finally:
        preview_module.inspect_input = original_inspector

    try:
        save_srgb8(
            np.zeros((2, 3, 3), dtype=np.float32),
            scratch / "wrong-extension.jpg",
            png_compression=6,
        )
    except ValueError:
        controls["non_png_compression_rejects"] = not (
            scratch / "wrong-extension.jpg"
        ).exists()
    else:
        controls["non_png_compression_rejects"] = False
    return controls


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    if not source.is_file() or sha256_file(source) != config["source"]["sha256"]:
        raise U45GError("frozen source identity mismatch")
    immutable_paths = [
        source,
        ROOT / "configs/render_profiles/safe_rich_v1.json",
        ROOT / "configs/film_color_stats.json",
        ROOT / "configs/color_guardrails.json",
        ROOT / "src/preprocess/output_encode.py",
        ROOT / "src/inference/three_stock_preview.py",
        ROOT / "scripts/render_three_stock_preview.py",
        config_path,
        Path(__file__),
    ]
    before = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in immutable_paths}
    cases: list[tuple[str, int | None]] = [
        ("default", None),
        ("level-0", 0),
        ("level-6", 6),
        ("level-9", 9),
    ]
    if reverse:
        cases.reverse()
    scratch = Path(tempfile.mkdtemp(prefix="u4_5g_", dir=ROOT / "tmp"))
    try:
        records = [
            _run_cli(config, source, scratch / name, compression=compression)
            for name, compression in cases
        ]
        records.sort(key=lambda item: item["invocation"])
        invalid = _invalid_controls(source, scratch)
        by_name = {record["invocation"]: record for record in records}
        default = by_name["default"]
        zero = by_name["level-0"]
        six = by_name["level-6"]
        nine = by_name["level-9"]
        styles = sorted(default["rows"])
        historical = {
            row["style_id"]: row["sha256"]
            for row in config["historical_default_level_6"]["rows"]
        }
        after = {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in immutable_paths
        }
        for child in scratch.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        owned_residue_empty = not any(scratch.iterdir())
        gates = {
            "execution_bindings_exact": _verify_execution_bindings(config),
            "source_and_runtime_inputs_immutable": before == after,
            "default_explicit_level_6_output_identity": all(
                default["rows"][style]["file_sha256"]
                == six["rows"][style]["file_sha256"]
                for style in styles
            ),
            "default_explicit_level_6_manifest_semantic_identity": (
                default["manifest"] == six["manifest"]
            ),
            "historical_level_6_output_identity": all(
                default["rows"][style]["file_sha256"] == historical[style]
                for style in styles
            ),
            "all_levels_decoded_rgb8_identity": all(
                len(
                    {
                        record["rows"][style]["pixel_sha256"]
                        for record in records
                    }
                )
                == 1
                for style in styles
            ),
            "all_levels_icc_identity": all(
                len(
                    {record["rows"][style]["icc_sha256"] for record in records}
                )
                == 1
                for style in styles
            ),
            "level_0_vs_9_encoded_difference": all(
                zero["rows"][style]["file_sha256"]
                != nine["rows"][style]["file_sha256"]
                for style in styles
            ),
            "level_9_bytes_not_greater_than_level_0": all(
                nine["rows"][style]["bytes"] <= zero["rows"][style]["bytes"]
                for style in styles
            ),
            "manifest_compression_binding": all(
                record["manifest"]["png_compression"] == record["compression"]
                for record in records
            ),
            "invalid_predecode_rejection": all(invalid.values()),
            "owned_residue_empty": owned_residue_empty,
        }
        report = {
            "schema": "kmcfm.u4-5g-preview-png-compression-binding-result.v1",
            "status": (
                "PASS_PRIVATE_U4_5G_PREVIEW_PNG_COMPRESSION_BINDING"
                if all(gates.values())
                else "FAIL_CLOSED_U4_5G_PREVIEW_PNG_COMPRESSION_BINDING"
            ),
            "execution_order": "reverse" if reverse else "forward",
            "bindings": {
                "config_sha256": sha256_file(config_path),
                "runner_sha256": sha256_file(Path(__file__)),
                "source_sha256": sha256_file(source),
                "output_encode_sha256": sha256_file(
                    ROOT / "src/preprocess/output_encode.py"
                ),
                "three_stock_preview_sha256": sha256_file(
                    ROOT / "src/inference/three_stock_preview.py"
                ),
                "preview_cli_sha256": sha256_file(
                    ROOT / "scripts/render_three_stock_preview.py"
                ),
            },
            "runtime": {
                "platform": sys.platform,
                "python": ".".join(map(str, sys.version_info[:3])),
                "pillow": pillow_version,
            },
            "records": records,
            "invalid_controls": invalid,
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        scientific = {
            key: value
            for key, value in report.items()
            if key not in {"execution_order", "status"}
        }
        report["scientific_identity"] = f"sha256:{_canonical_sha256(scientific)}"
        report["owned_residue_empty"] = owned_residue_empty
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(args.config, reverse=args.order == "reverse")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
