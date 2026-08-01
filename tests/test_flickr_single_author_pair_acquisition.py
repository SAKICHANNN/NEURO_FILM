from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from src.eval.flickr_single_author_pair_acquisition import (
    FlickrPairAcquisitionError,
    acquire,
    audit,
    selected_rows,
)


def _jpeg(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (640, 400), colour).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers = {"content-type": "image/jpeg"}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.payload[:7]
        yield self.payload[7:]

    def close(self) -> None:
        return None


class _Session:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **_: object) -> _Response:
        self.calls.append(url)
        return _Response(self.payloads[url])


def _config() -> dict:
    return {
        "schema": "neuro-film.u5-r2bo1-flickr-single-author-pair-acquisition.v1",
        "node": "test",
        "parent": {"stable_evidence_id": "evidence", "report_sha256": "0" * 64},
        "acquisition": {
            "expected_pairs": 1,
            "expected_files": 2,
            "allowed_license_ids": [1],
            "maximum_bytes_per_file": 1_000_000,
            "maximum_bytes_total": 2_000_000,
            "maximum_long_edge": 1024,
            "minimum_pixels": 200_000,
            "maximum_metadata_dimension_delta_per_axis": 2,
            "required_format": "JPEG",
            "allowed_modes": ["RGB"],
            "require_single_frame": True,
            "timeout_seconds": 1,
            "retries": 1,
            "retry_backoff_seconds": 0,
            "request_interval_seconds": 0,
        },
        "integrity_gates": {
            "required_files": 2,
            "required_complete_pairs": 1,
            "maximum_exact_duplicate_groups": 0,
            "maximum_cross_pair_dhash_le_4": 0,
        },
        "branches": {"pass": "open", "fail": "close"},
        "claim_ceiling": "test",
    }


def _parent() -> dict:
    records = []
    for role, photo_id in (("digital", "1"), ("film", "2")):
        records.append(
            {
                "photo_id": photo_id,
                "role": role,
                "license_id": 1,
                "public_photo_media": True,
                "derivative_url_l": f"https://example.test/{photo_id}.jpg",
                "derivative_width_l": 640,
                "derivative_height_l": 400,
                "page_url": f"https://example.test/page/{photo_id}",
            }
        )
    return {
        "stable_evidence_id": "evidence",
        "automatic_pass": True,
        "records": records,
        "complete_pairs": [
            {
                "pair_id": "family/scene-01",
                "family_id": "family",
                "scene_id": 1,
                "digital_photo_id": "1",
                "film_photo_id": "2",
                "stock_label": "unknown",
                "both_allowed_license": True,
                "both_public_photo_media": True,
                "both_bounded_derivatives": True,
            }
        ],
    }


def test_acquire_replay_and_audit(tmp_path: Path) -> None:
    config = _config()
    rows = selected_rows(_parent(), config)
    session = _Session({rows[0]["derivative_url"]: _jpeg((10, 20, 30)), rows[1]["derivative_url"]: _jpeg((200, 100, 20))})
    manifest = acquire(rows, root=tmp_path, config=config, session=session)
    assert len(session.calls) == 2
    report = audit(manifest, root=tmp_path, config=config)
    assert report["automatic_pass"]
    replay = acquire(rows, root=tmp_path, config=config, prior_manifest=manifest, session=_Session({}))
    assert replay == manifest


def test_acquire_rejects_untracked_existing_file(tmp_path: Path) -> None:
    config = _config()
    rows = selected_rows(_parent(), config)
    target = tmp_path / "family" / "scene-01-digital-1.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(_jpeg((1, 2, 3)))
    with pytest.raises(FlickrPairAcquisitionError, match="untracked"):
        acquire(rows, root=tmp_path, config=config, session=_Session({}))


def test_selected_rows_rejects_rights_drift() -> None:
    parent = _parent()
    parent["records"][0]["license_id"] = 0
    with pytest.raises(FlickrPairAcquisitionError, match="metadata limits"):
        selected_rows(parent, _config())


def test_decode_dimension_tolerance_is_bounded(tmp_path: Path) -> None:
    config = _config()
    parent = _parent()
    parent["records"][0]["derivative_width_l"] = 642
    rows = selected_rows(parent, config)
    session = _Session(
        {
            rows[0]["derivative_url"]: _jpeg((10, 20, 30)),
            rows[1]["derivative_url"]: _jpeg((200, 100, 20)),
        }
    )
    assert acquire(rows, root=tmp_path, config=config, session=session)["complete"]
    parent["records"][0]["derivative_width_l"] = 643
    rows = selected_rows(parent, config)
    with pytest.raises(FlickrPairAcquisitionError, match="dimensions drift"):
        acquire(rows, root=tmp_path / "fail", config=config, session=session)


def test_audit_rejects_byte_drift(tmp_path: Path) -> None:
    config = _config()
    rows = selected_rows(_parent(), config)
    session = _Session({rows[0]["derivative_url"]: _jpeg((10, 20, 30)), rows[1]["derivative_url"]: _jpeg((200, 100, 20))})
    manifest = acquire(rows, root=tmp_path, config=config, session=session)
    (tmp_path / manifest["rows"][0]["local_path"]).write_bytes(b"drift")
    with pytest.raises(FlickrPairAcquisitionError, match="identity drift"):
        audit(manifest, root=tmp_path, config=config)
