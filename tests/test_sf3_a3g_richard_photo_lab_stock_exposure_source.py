from __future__ import annotations

import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

from src.real_film.ppisp_capture_pair_source_lock import (
    locate_central_directory,
    parse_central_directory,
)
from src.real_film.richard_photo_lab_stock_source import run_source_lock


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _zip_payload() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory, sequence in (
            ("2_KODAKPORTRA400", "Portra400-Neg Sequence"),
            ("5_KODAKEKTAR100", "Ektar100-Neg Sequence"),
        ):
            root = "Richards-Film-Stock-And-Exposure-Comparisons/" + directory + "/"
            archive.writestr(root + "NORMAL.jpg", b"scan")
            archive.writestr(root + sequence + "/NORMAL.jpg", b"negative")
    return output.getvalue()


def _contract(tmp_path: Path, archive: bytes, page: bytes) -> Path:
    tail = archive
    offset, size, count = locate_central_directory(tail, archive_size=len(archive))
    central = archive[offset : offset + size]
    members = parse_central_directory(central)
    inventory = [
        {
            "name": member.name,
            "flags": member.flags,
            "method": member.method,
            "crc32": member.crc32,
            "compressed_size": member.compressed_size,
            "uncompressed_size": member.uncompressed_size,
            "local_offset": member.local_offset,
        }
        for member in members
    ]
    config = {
        "schema": "neuro-film.sf3-a3g-richard-photo-lab-stock-exposure-source-lock-contract.v1",
        "experiment_id": "test",
        "source": {
            "page_url": "https://example/page",
            "page_size": len(page),
            "page_sha256": _sha256(page),
            "required_page_phrases": ["same scene"],
            "archive_url": "https://example/archive.zip",
            "archive_size": len(archive),
            "central_offset": offset,
            "central_size": size,
            "central_member_count": count,
            "tail_sha256": _sha256(tail),
            "central_sha256": _sha256(central),
            "central_inventory_sha256": _canonical_sha256(inventory),
            "jpeg_member_count": 4,
        },
        "required_stocks": [
            {
                "film_stock_id": "kodak_portra_400",
                "archive_directory": "2_KODAKPORTRA400",
                "minimum_full_resolution_scan_count": 1,
                "minimum_negative_sequence_count": 1,
                "required_full_resolution_normal_name": "NORMAL.jpg",
                "required_negative_normal_name": "Portra400-Neg Sequence/NORMAL.jpg",
            },
            {
                "film_stock_id": "kodak_ektar_100",
                "archive_directory": "5_KODAKEKTAR100",
                "minimum_full_resolution_scan_count": 1,
                "minimum_negative_sequence_count": 1,
                "required_full_resolution_normal_name": "NORMAL.jpg",
                "required_negative_normal_name": "Ektar100-Neg Sequence/NORMAL.jpg",
            },
        ],
        "network_limits": {
            "tail_probe_bytes": len(archive),
            "maximum_total_bytes": len(page) + len(archive),
            "member_payload_reads": 0,
            "image_member_reads": 0,
            "pixel_decodes": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_source_lock_is_exact_and_reads_no_members(tmp_path: Path) -> None:
    archive = _zip_payload()
    page = b"same scene"
    contract = _contract(tmp_path, archive, page)

    def range_reader(_url: str, start: int, end: int, archive_size: int) -> bytes:
        assert archive_size == len(archive)
        return archive[start : end + 1]

    report = run_source_lock(
        contract,
        range_reader=range_reader,
        url_reader=lambda _url: page,
    )
    assert report["decision"] == "PASS"
    assert report["member_payload_reads"] == 0
    assert report["image_member_reads"] == 0
    assert report["pixel_decodes"] == 0
    assert all(report["gates"].values())


def test_source_lock_fails_when_one_required_normal_is_missing(tmp_path: Path) -> None:
    archive = _zip_payload()
    page = b"same scene"
    contract = _contract(tmp_path, archive, page)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["required_stocks"][0]["required_full_resolution_normal_name"] = (
        "missing.jpg"
    )
    contract.write_text(json.dumps(payload), encoding="utf-8")

    report = run_source_lock(
        contract,
        range_reader=lambda _url, start, end, _size: archive[start : end + 1],
        url_reader=lambda _url: page,
    )
    assert report["decision"] == "FAIL"
    assert not report["gates"]["all_stock_ladders_present"]


def test_project_contract_is_source_lock_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "configs/sf3_a3g_richard_photo_lab_stock_exposure_source_lock_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert config["network_limits"]["member_payload_reads"] == 0
    assert config["network_limits"]["image_member_reads"] == 0
    assert config["network_limits"]["pixel_decodes"] == 0
    assert {row["film_stock_id"] for row in config["required_stocks"]} == {
        "kodak_portra_400",
        "kodak_ektar_100",
        "kodak_tri_x_400",
        "ilford_hp5_plus",
    }
