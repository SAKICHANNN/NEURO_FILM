from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

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
    assert {path.name for path in root.iterdir() if path.is_file()} == set(
        expected
    ) | {"SOURCE.json"}
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
