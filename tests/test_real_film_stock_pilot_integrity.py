from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.real_film.stock_pilot_integrity import (
    StockPilotIntegrityError,
    audit_stock_pilot_integrity,
)


def _write_png(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_stock_pilot_integrity_decodes_and_rejects_sealed_roll(tmp_path: Path) -> None:
    root = tmp_path / "pilots"
    preview = "negative-preview-8bit/blue-intact/rollA-01.preview.png"
    target = root / Path(*Path(preview).parts)
    _write_png(target, (12, 34, 56))
    digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    size = target.stat().st_size
    acquisition = {
        "manifest_id": "test",
        "file_count": 1,
        "bytes": size,
        "files": [
            {
                "film_stock_id": "kodak_ga_100_5095",
                "source_label": "Kodak GA 100 5095",
                "roll_id": "rollA",
                "frame_id": "rollA-01",
                "lane": "negative_preview",
                "path": preview,
                "size": size,
                "sha256": digest,
            }
        ],
    }
    download_report = {
        "files": 1,
        "bytes": size,
        "all_sizes_and_lfs_sha256_verified": True,
        "manifest_external_lane_files": 0,
    }
    metadata_report = {
        "stock_summaries": {
            "kodak_ga_100_5095": {
                "source_label": "Kodak GA 100 5095",
                "display_operator_candidate": False,
                "metadata_identifiability_candidate": True,
                "display_proxy_rolls": 0,
            }
        }
    }
    frames = [
        {
            "filename": "rollA-01",
            "roll_id": "rollA",
            "film_type": "Kodak GA 100 5095",
            "preview_path": preview,
            "pseudogt_path": None,
            "location": "test-lab",
            "scene_property": {"is_daytime": "day", "is_indoor": "outdoor"},
            "partition": "blue-intact",
            "alignment_available": False,
        }
    ]
    rolls = [{
        "roll_id": "rollA",
        "film_type": "Kodak GA 100 5095",
        "contains_official_test_frame": False,
    }]
    report = audit_stock_pilot_integrity(
        download_root=root,
        acquisition=acquisition,
        download_report=download_report,
        metadata_report=metadata_report,
        frame_rows=frames,
        roll_rows=rolls,
        software_commit="test",
    )
    assert report["passed"] is True
    assert report["checks"]["decoded_files"] == 1
    assert report["eligibility_by_stock"]["kodak_ga_100_5095"]["negative_preview_identifiability_eligible"] is False

    sealed_rolls = [{
        "roll_id": "rollA",
        "film_type": "Kodak GA 100 5095",
        "contains_official_test_frame": True,
    }]
    try:
        audit_stock_pilot_integrity(
            download_root=root,
            acquisition=acquisition,
            download_report=download_report,
            metadata_report=metadata_report,
            frame_rows=frames,
            roll_rows=sealed_rolls,
            software_commit="test",
        )
        raise AssertionError("expected sealed-roll rejection")
    except StockPilotIntegrityError as exc:
        assert "official-test roll" in str(exc)


def test_stock_pilot_integrity_detects_exact_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "pilots"
    color = (200, 10, 10)
    paths = [
        "negative-preview-8bit/blue-intact/rollA-01.preview.png",
        "negative-preview-8bit/blue-intact/rollA-02.preview.png",
    ]
    files = []
    for remote in paths:
        target = root / Path(*Path(remote).parts)
        _write_png(target, color)
        digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
        files.append(
            {
                "film_stock_id": "fujifilm_nph_400",
                "source_label": "Fuji NPH400",
                "roll_id": "rollA",
                "frame_id": Path(remote).stem.replace(".preview", ""),
                "lane": "negative_preview",
                "path": remote,
                "size": target.stat().st_size,
                "sha256": digest,
            }
        )
    acquisition = {"manifest_id": "test", "file_count": 2, "bytes": sum(f["size"] for f in files), "files": files}
    download_report = {
        "files": 2,
        "bytes": acquisition["bytes"],
        "all_sizes_and_lfs_sha256_verified": True,
        "manifest_external_lane_files": 0,
    }
    frames = [
        {
            "filename": files[0]["frame_id"],
            "roll_id": "rollA",
            "film_type": "Fuji NPH400",
            "preview_path": paths[0],
            "pseudogt_path": None,
            "location": "a",
            "scene_property": {"is_daytime": "day", "is_indoor": "outdoor"},
            "partition": "blue-intact",
            "alignment_available": False,
        },
        {
            "filename": files[1]["frame_id"],
            "roll_id": "rollA",
            "film_type": "Fuji NPH400",
            "preview_path": paths[1],
            "pseudogt_path": None,
            "location": "b",
            "scene_property": {"is_daytime": "night", "is_indoor": "indoor"},
            "partition": "blue-intact",
            "alignment_available": False,
        },
    ]
    report = audit_stock_pilot_integrity(
        download_root=root,
        acquisition=acquisition,
        download_report=download_report,
        metadata_report={
            "stock_summaries": {
                "fujifilm_nph_400": {"source_label": "Fuji NPH400"}
            }
        },
        frame_rows=frames,
        roll_rows=[{
            "roll_id": "rollA",
            "film_type": "Fuji NPH400",
            "contains_official_test_frame": False,
        }],
        software_commit="test",
    )
    assert report["passed"] is False
    assert report["checks"]["exact_sha_duplicate_groups"] == 1


def test_stock_pilot_integrity_rejects_manifest_metadata_mislabel(tmp_path: Path) -> None:
    root = tmp_path / "pilots"
    preview = "negative-preview-8bit/blue-intact/rollA-01.preview.png"
    target = root / Path(*Path(preview).parts)
    _write_png(target, (12, 34, 56))
    digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
    size = target.stat().st_size
    acquisition = {
        "manifest_id": "test",
        "file_count": 1,
        "bytes": size,
        "files": [{
            "film_stock_id": "fujifilm_nph_400",
            "source_label": "Fuji NPH400",
            "roll_id": "wrong-roll",
            "frame_id": "rollA-01",
            "lane": "negative_preview",
            "path": preview,
            "size": size,
            "sha256": digest,
        }],
    }
    download_report = {
        "files": 1,
        "bytes": size,
        "all_sizes_and_lfs_sha256_verified": True,
        "manifest_external_lane_files": 0,
    }
    frames = [{
        "filename": "rollA-01",
        "film_type": "Fuji NPH400",
        "roll_id": "rollA",
        "preview_path": preview,
        "pseudogt_path": None,
        "alignment_available": False,
    }]
    try:
        audit_stock_pilot_integrity(
            download_root=root,
            acquisition=acquisition,
            download_report=download_report,
            metadata_report={
                "stock_summaries": {
                    "fujifilm_nph_400": {"source_label": "Fuji NPH400"}
                }
            },
            frame_rows=frames,
            roll_rows=[{
                "roll_id": "rollA",
                "film_type": "Fuji NPH400",
                "contains_official_test_frame": False,
            }],
            software_commit="test",
        )
        raise AssertionError("expected manifest/metadata mismatch rejection")
    except StockPilotIntegrityError as exc:
        assert "roll mismatch" in str(exc)
