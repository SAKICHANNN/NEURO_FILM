from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_p268_hdrplus_burst_source_feasibility import (
    P268Error,
    _burst_ids,
    _canonical_bytes,
    _objects,
    _page_facts,
    _selected_burst,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p268_hdrplus_burst_source_feasibility_v1.json"
EVIDENCE = ROOT / "docs/evidence/P268_HDRPLUS_BURST_SOURCE_FEASIBILITY_RESULT.json"


def test_p268_config_forbids_bulk_download_and_freezes_selection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["official"]["curated_bursts"] == 153
    assert config["budgets"]["maximum_full_subset_download_bytes"] == 0
    assert config["budgets"]["maximum_object_body_bytes_read"] == 0
    assert config["selection"]["selected_burst_id"] == "0382_20150912_131056_666"
    assert config["selection"]["selected_burst_id_sha256"] == (
        "01d46d74ca18831b4bff9cf73bd79e186d901c765444157c8ae9edcf16a56751"
    )


def test_p268_selection_is_order_independent() -> None:
    values = ["safe_burst", "another-burst", "third"]
    assert _selected_burst(values) == _selected_burst(list(reversed(values)))


def test_p268_burst_ids_reject_unsafe_or_duplicate_prefixes() -> None:
    prefix = "gs://bucket/bursts/"
    with pytest.raises(P268Error, match="unsafe burst ID"):
        _burst_ids([{"type": "prefix", "url": f"{prefix}bad/nested/"}], prefix)
    with pytest.raises(P268Error, match="duplicate burst ID"):
        _burst_ids(
            [
                {"type": "prefix", "url": f"{prefix}same/"},
                {"type": "prefix", "url": f"{prefix}same/"},
            ],
            prefix,
        )


def test_p268_objects_preserve_only_immutable_metadata() -> None:
    rows = [
        {"type": "prefix", "url": "gs://bucket/root/"},
        {
            "type": "cloud_object",
            "metadata": {
                "name": "root/payload_N000.dng",
                "size": "123",
                "generation": "456",
                "metageneration": "1",
                "crc32c": "crc",
                "md5Hash": "md5",
                "contentType": "image/x-adobe-dng",
                "updated": "2020-01-01T00:00:00Z",
                "mediaLink": "must-not-persist",
            },
        },
    ]
    assert _objects(rows, "root/") == [
        {
            "relative_path": "payload_N000.dng",
            "bytes": 123,
            "generation": "456",
            "metageneration": "1",
            "crc32c": "crc",
            "md5_base64": "md5",
            "content_type": "image/x-adobe-dng",
            "updated": "2020-01-01T00:00:00Z",
        }
    ]


def test_p268_page_facts_are_explicit_and_canonical() -> None:
    dataset = b" ".join(
        value.encode("utf-8")
        for value in [
            "3640 bursts",
            "28461 images",
            "153 bursts, 37 GiB",
            "765 GiB",
            "payload_N&lt;frame&gt;.dng",
            "lens_shading_map_N&lt;frame&gt;.tiff",
            "rgb2rgb.txt",
            "merged.dng",
            "final.jpg",
            "reference_frame.txt",
            "Creative Commons license (CC-BY-SA)",
            "https://creativecommons.org/licenses/by-sa/4.0/",
        ]
    )
    paper = b"do not use bracketed exposures frames of constant exposure Bayer raw frames Camera2 API"
    facts = _page_facts(dataset, paper)
    assert all(facts["dataset_required_statements"].values())
    assert all(facts["paper_required_statements"].values())
    assert facts["cc_by_sa_4_link_present"] is True
    assert _canonical_bytes(facts) == _canonical_bytes(
        json.loads(_canonical_bytes(facts))
    )


def test_p268_evidence_binds_exact_formal_reports() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_PRIVATE_HDRPLUS_ONE_BURST_SOURCE_FEASIBILITY"
    assert evidence["result"]["two_complete_reports_byte_exact"] is True
    assert evidence["result"]["object_payload_body_bytes_read"] == 0
    assert evidence["result"]["dng_jpeg_tiff_pixel_decodes"] == 0
    reports = evidence["formal_reports"]
    assert reports[0]["sha256"] == reports[1]["sha256"]
    for report in reports:
        payload = (ROOT / report["path"]).read_bytes()
        assert len(payload) == report["bytes"]
        assert hashlib.sha256(payload).hexdigest() == report["sha256"]
