from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_p277_makehdr_iso21496_decoder_compatibility import (
    build_decoder_command,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p277_makehdr_iso21496_decoder_compatibility_v1.json"


def test_p277_source_lock_is_exact_and_minimal() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    root = ROOT / config["source"]["root"]
    manifest_path = ROOT / config["source"]["manifest"]
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        == config["source"]["manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {row["name"]: row for row in manifest["files"]}
    assert set(expected) == {
        "README.txt",
        "interior-window-hdr.jpg",
        "interior-window-sdr.jpg",
        "night-street-hdr.jpg",
        "night-street-sdr.jpg",
        "sunrise-valley-hdr.jpg",
        "sunrise-valley-sdr.jpg",
    }
    assert {path.name for path in root.iterdir() if path.is_file()} == set(expected) | {
        "SOURCE.json"
    }
    for name, row in expected.items():
        path = root / name
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_p277_pairs_and_decoder_command_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert [row["scene"] for row in config["pairs"]] == [
        "interior-window",
        "night-street",
        "sunrise-valley",
    ]
    assert all((row["width"], row["height"]) == (2000, 1342) for row in config["pairs"])
    assert build_decoder_command(Path("decoder.exe"), "input.jpg", "output.raw") == [
        "decoder.exe",
        "-m",
        "1",
        "-j",
        "input.jpg",
        "-o",
        "0",
        "-O",
        "4",
        "-z",
        "output.raw",
    ]


def test_p277_evidence_is_exact_when_present() -> None:
    path = (
        ROOT / "docs/evidence/P277_MAKEHDR_ISO21496_DECODER_COMPATIBILITY_RESULT.json"
    )
    if not path.is_file():
        return
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["status"] in {
        "PASS_PRIVATE_INDEPENDENT_ISO21496_JPEG_DECODER_COMPATIBILITY",
        "FAIL_CLOSED_INDEPENDENT_ISO21496_JPEG_DECODER_COMPATIBILITY",
    }
    assert evidence["execution"]["forward_reverse_report_exact"] is True
    assert evidence["source"]["retained_file_count"] == 7
