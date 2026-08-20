from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

from src.real_film.ntire_night_capture_pair_source_lock import (
    analyze_pair_graph,
    run_source_lock,
)
from src.real_film.ppisp_capture_pair_source_lock import parse_central_directory


def _archive(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def _central(archive: bytes) -> bytes:
    eocd = archive.rfind(b"PK\x05\x06")
    values = struct.unpack_from("<4s4H2IH", archive, eocd)
    return archive[values[6] : values[6] + values[5]]


def test_pair_graph_requires_exact_four_way_stems() -> None:
    raw = _archive(
        {
            "raw/0.png": b"r",
            "metadata/0.json": b"{}",
            "processed/0.jpg": b"p",
            "raw/1.png": b"r",
        }
    )
    target = _archive({"sony/0.JPG": b"t", "sony/1.JPG": b"t"})
    facts = analyze_pair_graph(
        parse_central_directory(_central(raw)),
        parse_central_directory(_central(target)),
    )
    assert facts["complete_group_count"] == 1
    assert facts["raw_png_count"] == 2
    assert facts["unsafe_member_count"] == 0


def test_source_lock_uses_only_two_archive_tails(tmp_path: Path) -> None:
    raw = _archive(
        {
            **{f"raw/{i}.png": b"r" for i in range(64)},
            **{f"metadata/{i}.json": b"{}" for i in range(64)},
            **{f"processed/{i}.jpg": b"p" for i in range(64)},
        }
    )
    target = _archive({f"sony/{i}.JPG": b"t" for i in range(64)})
    source = {
        "zenodo_record": 1,
        "doi": "10.test/1",
        "api": "https://example.test/api",
        "license": "CC-BY-4.0",
        "publication_date": "2025-01-01",
        "raw_archive": {
            "name": "raw.zip",
            "bytes": len(raw),
            "md5": "a",
            "url": "https://example.test/raw",
        },
        "target_archive": {
            "name": "sony.zip",
            "bytes": len(target),
            "md5": "b",
            "url": "https://example.test/sony",
        },
    }
    fields = [
        "black_level",
        "white_level",
        "noise_profile",
        "cfa_pattern",
        "orientation",
        "white_balance",
        "crop_bounds",
    ]
    config = {
        "schema": "neuro-film.sf3-a0u-ntire-night-capture-pair-source-lock-contract.v1",
        "experiment_id": "test",
        "source": source,
        "official_paper": {
            "capture_facts": [
                "black level white level noise profile cfa pattern orientation white balance crop bounds"
            ]
        },
        "read_budget": {
            "maximum_tail_bytes_per_archive": max(len(raw), len(target)),
            "maximum_total_archive_bytes": len(raw) + len(target),
        },
        "required_source_gates": {"capture_metadata_fields_present": fields},
        "decision_if_pass": "PASS",
        "bounded_final_candidate_counter_before": 1,
        "bounded_final_candidate_counter_after": 1,
        "claim_ceiling": "test",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    api = {
        "id": 1,
        "metadata": {
            "doi": "10.test/1",
            "publication_date": "2025-01-01",
            "license": {"id": "cc-by-4.0"},
        },
        "files": [
            {
                "key": "raw.zip",
                "size": len(raw),
                "checksum": "md5:a",
                "links": {"content": source["raw_archive"]["url"]},
            },
            {
                "key": "sony.zip",
                "size": len(target),
                "checksum": "md5:b",
                "links": {"content": source["target_archive"]["url"]},
            },
        ],
    }
    ranges = []

    def range_reader(url: str, start: int, end: int, size: int) -> bytes:
        ranges.append((url, start, end, size))
        payload = raw if url.endswith("raw") else target
        return payload[start : end + 1]

    report = run_source_lock(
        path, range_reader=range_reader, url_reader=lambda _: json.dumps(api).encode()
    )
    assert report["automatic_pass"] is True
    assert report["complete_group_count"] == 64
    assert report["member_payload_bytes"] == report["pixel_decodes"] == 0
    assert len(ranges) == 2
