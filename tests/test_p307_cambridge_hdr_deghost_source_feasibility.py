from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.audit_p307_cambridge_hdr_deghost_source_feasibility import (
    P307Error,
    _fetch_range,
    _normalized_text,
    analyze_members,
)
from src.real_film.ppisp_capture_pair_source_lock import parse_central_directory

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p307_cambridge_hdr_deghost_source_feasibility_v1.json"


def _members() -> list:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for role in ("ghosted", "ground_truth"):
            for file_format, suffix in (("jpg", ".jpg"), ("raw", ".cr2")):
                for index in range(1, 6):
                    marker = "gt3" if role == "ghosted" and index == 3 else str(index)
                    name = (
                        "exposure_stacks_part2/losm/image_set1/"
                        f"{role}/{file_format}/setup1_losm_{marker}{suffix}"
                    )
                    archive.writestr(name, f"{role}-{file_format}-{index}".encode())
    payload = stream.getvalue()
    start = payload.index(b"PK\x01\x02")
    end = payload.index(b"PK\x05\x06")
    return parse_central_directory(payload[start:end])


def test_p307_analyzer_accepts_complete_paired_group() -> None:
    facts = analyze_members(_members(), ["losm"])
    assert facts["group_count"] == 1
    assert facts["complete_group_count"] == 1
    assert not facts["incomplete_groups"]
    assert not facts["invalid_role_overlaps"]
    assert facts["unexpected_file_count"] == 0


def test_p307_analyzer_rejects_unexpected_category() -> None:
    facts = analyze_members(_members(), ["nrm"])
    assert facts["unexpected_file_count"] == 20
    assert facts["missing_group_count"] == 4


def test_p307_contract_excludes_r1dn_and_payload_reads() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["forbidden"]["archives"] == ["exposure_stacks_part1.zip"]
    assert config["forbidden"]["categories"] == ["complex", "handheld", "lolm"]
    assert config["forbidden"]["member_payload_reads"] == 0
    assert config["fresh_roles"]["expected_group_count"] == 24
    assert config["fresh_roles"]["categories"] == [
        "losm",
        "multiview",
        "nrm",
        "occlusion",
        "solm",
        "sosm",
    ]


def test_p307_metadata_text_normalization_is_crlf_stable() -> None:
    assert _normalized_text("test stacks, with motion\r\nand misalignment") == (
        "test stacks, with motion and misalignment"
    )


def test_p307_fetch_range_rejects_nonexact_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status = 200

        def __init__(self) -> None:
            self.headers = {"Content-Range": None}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size: int) -> bytes:
            return b"0123456789"

        def geturl(self) -> str:
            return "https://example.invalid/archive.zip"

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(P307Error, match="exact frozen Range"):
        _fetch_range("https://example.invalid/archive.zip", 4, 9, 10)
