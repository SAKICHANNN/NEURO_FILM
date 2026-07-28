from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.eval.haldclut_archive import (
    HaldArchiveError,
    acquire_archive,
    audit_archive,
    finalize_repeat_evidence,
    hald_geometry,
    load_config_snapshot,
    run_acquisition,
    sha256_file,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2aj0b_haldclut_acquisition_v1.json"


def _png_bytes(side: int = 8) -> bytes:
    values = np.arange(side * side * 3, dtype=np.uint8).reshape(side, side, 3)
    stream = io.BytesIO()
    Image.fromarray(values, mode="RGB").save(stream, "PNG")
    return stream.getvalue()


def _write_fixture(
    path: Path,
    *,
    unsafe: bool = False,
    format_mismatch: bool = False,
) -> dict[str, bytes]:
    files = {
        "HaldCLUT/README.txt": (
            b"RawTherapee Film Simulation Collection version 2015-09-20\n"
            b"CC BY-SA 4.0\n"
        ),
        "HaldCLUT/Hald_CLUT_Identity_2.png": _png_bytes(),
        "HaldCLUT/Color/Kodak/Test.png": _png_bytes(),
        "HaldCLUT/Color/CreativePack-1/Test.png": _png_bytes(),
        "HaldCLUT/Black-and-White/Test.png": _png_bytes(),
    }
    if unsafe:
        files["../escape.png"] = _png_bytes()
    if format_mismatch:
        stream = io.BytesIO()
        Image.fromarray(
            np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3),
            mode="RGB",
        ).save(stream, "TIFF")
        files["HaldCLUT/Color/Kodak/Test.png"] = stream.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return files


def _fixture_config(path: Path, files: dict[str, bytes]) -> dict:
    base = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = {
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        readme = archive.getinfo("HaldCLUT/README.txt")
    # EOCD/central-directory expectations are supplied by a first local parse
    # only for this synthetic unit fixture, never for the formal source.
    raw = path.read_bytes()
    position = raw.rfind(b"PK\x05\x06")
    (
        _signature,
        _disk,
        _directory_disk,
        _disk_entries,
        entries,
        directory_bytes,
        directory_offset,
        _comment_bytes,
    ) = __import__("struct").unpack("<4s4H2LH", raw[position : position + 22])
    directory = raw[directory_offset : directory_offset + directory_bytes]
    png_files = sum(name.endswith(".png") for name in files)
    color_files = sum(name.startswith("HaldCLUT/Color/") for name in files)
    black_and_white_files = sum(
        name.startswith("HaldCLUT/Black-and-White/") for name in files
    )
    creative_files = sum(
        name.startswith("HaldCLUT/Color/CreativePack-1/") for name in files
    )
    primary_files = color_files - creative_files
    base["archive"].update(
        {
            "expected_bytes": path.stat().st_size,
            "maximum_bytes": path.stat().st_size,
            "expected_md5": hashes["md5"],
            "destination": path.name,
            "partial_destination": f"{path.name}.part",
        }
    )
    base["expected_inventory"].update(
        {
            "entries": entries,
            "files": len(infos),
            "directories": 0,
            "png_files": png_files,
            "tiff_files": 0,
            "text_files": 1,
            "color_files": color_files,
            "black_and_white_files": black_and_white_files,
            "creative_pack_color_files": creative_files,
            "primary_noncreative_color_files": primary_files,
            "central_directory_offset": directory_offset,
            "central_directory_bytes": directory_bytes,
            "central_directory_sha256": hashlib.sha256(directory).hexdigest(),
            "readme_bytes": readme.file_size,
            "readme_crc32": f"{readme.CRC:08x}",
            "readme_sha256": hashlib.sha256(
                files["HaldCLUT/README.txt"]
            ).hexdigest(),
        }
    )
    base["primary_universe"]["expected_files"] = primary_files
    base["decode"]["expected_image_files"] = png_files
    base["decode"]["allowed_extensions"] = [".png"]
    return base


def test_frozen_contract_validates() -> None:
    validate_contract(json.loads(CONFIG.read_text(encoding="utf-8")))


def test_hald_geometry_is_exact_and_fails_closed() -> None:
    assert hald_geometry(8, 8) == {
        "width": 8,
        "height": 8,
        "hald_level": 2,
        "cube_side": 4,
    }
    with pytest.raises(HaldArchiveError, match="square"):
        hald_geometry(8, 7)
    with pytest.raises(HaldArchiveError, match="integer cube"):
        hald_geometry(9, 9)


def test_existing_exact_archive_is_reused_without_network(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"exact frozen bytes")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"].update(
        {
            "expected_bytes": path.stat().st_size,
            "maximum_bytes": path.stat().st_size,
            "expected_md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "destination": path.name,
            "partial_destination": f"{path.name}.part",
        }
    )
    observed_path, identity = acquire_archive(root=tmp_path, config=config)
    assert observed_path == path
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_wrong_archive_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"user bytes must remain")
    before = path.read_bytes()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"].update(
        {
            "expected_bytes": len(before),
            "maximum_bytes": len(before),
            "expected_md5": "0" * 32,
            "destination": path.name,
            "partial_destination": f"{path.name}.part",
        }
    )
    with pytest.raises(HaldArchiveError, match="MD5 mismatch"):
        acquire_archive(root=tmp_path, config=config)
    assert path.read_bytes() == before


def test_complete_valid_partial_is_promoted_without_network(tmp_path: Path) -> None:
    payload = b"complete valid partial"
    partial = tmp_path / "fixture.bin.part"
    partial.write_bytes(payload)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"].update(
        {
            "expected_bytes": len(payload),
            "maximum_bytes": len(payload),
            "expected_md5": hashlib.md5(payload).hexdigest(),
            "destination": "fixture.bin",
            "partial_destination": partial.name,
        }
    )
    observed_path, identity = acquire_archive(root=tmp_path, config=config)
    assert observed_path.read_bytes() == payload
    assert not partial.exists()
    assert identity["sha256"] == hashlib.sha256(payload).hexdigest()


def test_fixture_archive_audit_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "fixture.zip"
    files = _write_fixture(path)
    config = _fixture_config(path, files)
    first = audit_archive(archive_path=path, config=config)
    second = audit_archive(archive_path=path, config=deepcopy(config))
    assert first == second
    assert first["automatic_pass"] is True
    assert first["photograph_rendered"] is False
    assert first["operator_applied"] is False
    assert first["primary_paths"] == ["HaldCLUT/Color/Kodak/Test.png"]
    assert len(first["image_records"]) == 4


def test_unsafe_zip_path_fails_before_candidate_use(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    files = _write_fixture(path, unsafe=True)
    config = _fixture_config(path, files)
    with pytest.raises(HaldArchiveError, match="traversal"):
        audit_archive(archive_path=path, config=config)


def test_image_suffix_must_match_actual_format(tmp_path: Path) -> None:
    path = tmp_path / "format-mismatch.zip"
    files = _write_fixture(path, format_mismatch=True)
    config = _fixture_config(path, files)
    with pytest.raises(HaldArchiveError, match="format/suffix mismatch"):
        audit_archive(archive_path=path, config=config)


def test_contract_paths_may_not_escape_execution_root() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"]["destination"] = "../outside.zip"
    with pytest.raises(HaldArchiveError, match="canonical relative"):
        validate_contract(config)


def test_config_snapshot_binds_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(CONFIG.read_bytes())
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, observed = load_config_snapshot(path, expected_sha256=expected)
    assert loaded["experiment_id"] == "u5-r2aj0b-haldclut-acquisition-v1"
    assert observed == expected
    with pytest.raises(HaldArchiveError, match="snapshot SHA-256 mismatch"):
        load_config_snapshot(path, expected_sha256="0" * 64)


class _FakeRangeResponse:
    status = 206
    headers = {
        "Content-Range": "bytes 4-9/10",
        "Content-Length": "6",
    }

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _count: int) -> bytes:
        return b""

    def geturl(self) -> str:
        return "https://rawtherapee.com/shared/HaldCLUT.zip"


class _MalformedLengthResponse(_FakeRangeResponse):
    headers = {
        "Content-Range": "bytes 3-9/10",
        "Content-Length": "not-an-integer",
    }


class _RedirectedResponse(_FakeRangeResponse):
    def geturl(self) -> str:
        return "https://example.invalid/mirror/HaldCLUT.zip"


def test_bad_resume_range_quarantines_task_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / "fixture.bin.part"
    partial.write_bytes(b"abc")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"].update(
        {
            "expected_bytes": 10,
            "maximum_bytes": 10,
            "expected_md5": "0" * 32,
            "destination": "fixture.bin",
            "partial_destination": partial.name,
        }
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeRangeResponse(),
    )
    with pytest.raises(HaldArchiveError, match="partial quarantined"):
        acquire_archive(root=tmp_path, config=config)
    assert not partial.exists()
    quarantined = list(tmp_path.glob("fixture.bin.part.quarantine-range-mismatch-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"abc"


def test_oversize_partial_is_quarantined_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = tmp_path / "fixture.bin.part"
    partial.write_bytes(b"01234567890")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"].update(
        {
            "expected_bytes": 10,
            "maximum_bytes": 10,
            "expected_md5": "0" * 32,
            "destination": "fixture.bin",
            "partial_destination": partial.name,
        }
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network must not be called"),
    )
    with pytest.raises(HaldArchiveError, match="exceeds byte ceiling"):
        acquire_archive(root=tmp_path, config=config)
    assert not partial.exists()
    quarantined = list(tmp_path.glob("fixture.bin.part.quarantine-size-overflow-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"01234567890"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_MalformedLengthResponse(), "malformed-content-length"),
        (_RedirectedResponse(), "redirect-mismatch"),
    ],
)
def test_invalid_resume_response_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeRangeResponse,
    reason: str,
) -> None:
    partial = tmp_path / "fixture.bin.part"
    partial.write_bytes(b"abc")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["archive"].update(
        {
            "expected_bytes": 10,
            "maximum_bytes": 10,
            "expected_md5": "0" * 32,
            "destination": "fixture.bin",
            "partial_destination": partial.name,
        }
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: response,
    )
    with pytest.raises(HaldArchiveError, match="partial quarantined"):
        acquire_archive(root=tmp_path, config=config)
    assert not partial.exists()
    quarantined = list(tmp_path.glob(f"fixture.bin.part.quarantine-{reason}-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"abc"


def test_single_run_cannot_open_repeat_gate(tmp_path: Path) -> None:
    path = tmp_path / "fixture.zip"
    files = _write_fixture(path)
    config = _fixture_config(path, files)
    config_sha256 = "a" * 64
    software_commit = "b" * 40
    first = run_acquisition(
        root=tmp_path,
        config=config,
        config_sha256=config_sha256,
        output_dir=tmp_path / "run_a",
        software_commit=software_commit,
    )
    assert first["report"]["single_run_pass"] is True
    assert first["report"]["automatic_pass"] is False
    assert first["report"]["structural_audit_ready"] is False


def test_two_exact_runs_are_required_for_structural_readiness(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.zip"
    files = _write_fixture(path)
    config = _fixture_config(path, files)
    config_sha256 = "a" * 64
    software_commit = "b" * 40
    for name in ("run_a", "run_b"):
        run_acquisition(
            root=tmp_path,
            config=config,
            config_sha256=config_sha256,
            output_dir=tmp_path / name,
            software_commit=software_commit,
        )
    decision = finalize_repeat_evidence(
        first_dir=tmp_path / "run_a",
        second_dir=tmp_path / "run_b",
        config=config,
        config_sha256=config_sha256,
        software_commit=software_commit,
    )
    assert decision["automatic_pass"] is True
    assert decision["structural_audit_ready"] is True
    assert decision["run_a"]["manifest_sha256"] == sha256_file(
        tmp_path / "run_a/manifest.json"
    )

    report_b = tmp_path / "run_b/automatic_report.json"
    report_b.write_bytes(report_b.read_bytes() + b" ")
    rejected = finalize_repeat_evidence(
        first_dir=tmp_path / "run_a",
        second_dir=tmp_path / "run_b",
        config=config,
        config_sha256=config_sha256,
        software_commit=software_commit,
    )
    assert rejected["automatic_pass"] is False
    assert rejected["structural_audit_ready"] is False

    report_a = tmp_path / "run_a/automatic_report.json"
    report_b.write_bytes(report_a.read_bytes())
    malicious = json.loads(report_a.read_text(encoding="utf-8"))
    malicious["single_run_pass"] = "false"
    malicious["operator_applied"] = True
    malicious["photograph_rendered"] = True
    malicious_bytes = (
        json.dumps(malicious, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    report_a.write_bytes(malicious_bytes)
    report_b.write_bytes(malicious_bytes)
    laundered = finalize_repeat_evidence(
        first_dir=tmp_path / "run_a",
        second_dir=tmp_path / "run_b",
        config=config,
        config_sha256=config_sha256,
        software_commit=software_commit,
    )
    assert laundered["automatic_pass"] is False
    assert laundered["structural_audit_ready"] is False


def test_runner_help_loads_from_repo_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_u5_r2aj0b_haldclut_acquisition.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in completed.stdout
    assert "--output" in completed.stdout
