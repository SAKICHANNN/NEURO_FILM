from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.audit_p278_libavif_gainmap_avif_compatibility import build_decoder_command

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p278_libavif_gainmap_avif_compatibility_v1.json"


def test_p278_source_lock_is_exact_and_minimal() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = ROOT / config["source"]["root"]
    manifest_path = ROOT / config["source"]["manifest"]
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == config["source"]["manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["name"]: row for row in manifest["files"]}
    assert {path.name for path in root.iterdir() if path.is_file()} == set(expected) | {
        "SOURCE.json"
    }
    for name, row in expected.items():
        path = root / name
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        blob = subprocess.run(
            ["git", "hash-object", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert blob == row["git_blob"]


def test_p278_roles_and_decoder_command_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["valid_fixtures"]) == 3
    assert len(config["invalid_fixtures"]) == 3
    assert all(
        (row["width"], row["height"]) == (400, 300) for row in config["valid_fixtures"]
    )
    assert build_decoder_command(Path("decoder.exe"), "input.avif", "output.raw") == [
        "decoder.exe",
        "-m",
        "1",
        "-j",
        "input.avif",
        "-o",
        "0",
        "-O",
        "4",
        "-z",
        "output.raw",
    ]


def test_p278_evidence_is_exact_when_present() -> None:
    path = ROOT / "docs/evidence/P278_LIBAVIF_GAINMAP_AVIF_COMPATIBILITY_RESULT.json"
    if not path.is_file():
        return
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] in {
        "PASS_PRIVATE_LIBAVIF_GAINMAP_AVIF_DECODER_COMPATIBILITY",
        "FAIL_CLOSED_LIBAVIF_GAINMAP_AVIF_DECODER_COMPATIBILITY",
    }
    assert evidence["execution"]["forward_reverse_report_exact"] is True
    assert evidence["source"]["retained_file_count"] == 8
