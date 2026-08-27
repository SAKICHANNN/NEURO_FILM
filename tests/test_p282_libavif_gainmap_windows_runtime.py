from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.audit_p282_libavif_gainmap_windows_runtime import (
    build_base_command,
    build_metadata_command,
    build_tonemap_command,
    parse_alternate_headroom,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p282_libavif_gainmap_windows_runtime_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_p282_runtime_source_lock_is_exact_and_minimal() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    root = ROOT / runtime["root"]
    manifest_path = ROOT / runtime["manifest"]
    assert _sha256(manifest_path) == runtime["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["name"]: row for row in runtime["files"]}
    manifest_rows = {row["name"]: row for row in manifest["files"]}
    assert set(manifest_rows) == set(expected)
    for name, row in expected.items():
        assert {key: manifest_rows[name][key] for key in row} == row
    assert {path.name for path in root.iterdir() if path.is_file()} == set(expected) | {
        "SOURCE.json"
    }
    for name, row in expected.items():
        path = root / name
        assert path.stat().st_size == row["size"]
        assert _sha256(path) == row["sha256"]
        blob = subprocess.run(
            ["git", "hash-object", str(path)],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        assert blob == row["git_blob"]


def test_p282_fixture_roles_and_commands_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["fixtures"]["valid"]) == 3
    assert len(config["fixtures"]["invalid"]) == 3
    assert all(
        (row["width"], row["height"]) == (400, 300)
        for row in config["fixtures"]["valid"]
    )
    assert config["commands"]["info"] == [
        "avifdec.exe",
        "--info",
        "{input}",
    ]
    assert config["commands"]["metadata"] == [
        "avifgainmaputil.exe",
        "printmetadata",
        "{input}",
        "--jobs",
        "1",
    ]


def test_p282_command_builders_and_metadata_parser_are_exact() -> None:
    executable = Path("avifgainmaputil.exe")
    source = Path("input.avif")
    output = Path("output.png")
    assert (
        parse_alternate_headroom(" * Alternate headroom:  2.5 (as fraction: 5/2)")
        == 2.5
    )
    assert build_metadata_command(executable, source) == [
        "avifgainmaputil.exe",
        "printmetadata",
        "input.avif",
        "--jobs",
        "1",
    ]
    assert build_tonemap_command(executable, source, output, 2.5) == [
        "avifgainmaputil.exe",
        "tonemap",
        "input.avif",
        "output.png",
        "--headroom",
        "2.5",
        "--jobs",
        "1",
        "--depth",
        "12",
        "--speed",
        "10",
    ]
    assert build_base_command(Path("avifdec.exe"), source, output) == [
        "avifdec.exe",
        "--jobs",
        "1",
        "--depth",
        "16",
        "--png-compress",
        "0",
        "input.avif",
        "output.png",
    ]


def test_p282_evidence_is_exact_when_present() -> None:
    path = ROOT / "docs/evidence/P282_LIBAVIF_GAINMAP_WINDOWS_RUNTIME_RESULT.json"
    if not path.is_file():
        return
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] in {
        "PASS_PRIVATE_LIBAVIF_GAINMAP_WINDOWS_RUNTIME",
        "FAIL_CLOSED_LIBAVIF_GAINMAP_WINDOWS_RUNTIME",
    }
    assert evidence["execution"]["forward_reverse_report_exact"] is True
    assert evidence["runtime"]["retained_file_count"] == 3
