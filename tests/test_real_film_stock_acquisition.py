from __future__ import annotations

import pytest

from src.real_film.stock_acquisition import StockAcquisitionError, build_stock_pilot_acquisition


HASHES = {"registry": "0" * 64, "frames": "1" * 64, "rolls": "2" * 64,
          "remote_inventory": "3" * 64}


def _registry() -> dict:
    stocks = []
    for label in ("a", "b"):
        stocks.append({
            "film_stock_id": label, "display_name": label, "manufacturer": "x",
            "product_line": "y", "nominal_iso": 100, "film_type": "color_negative",
            "emulsion_generation": "unknown", "catalog_code": "unknown",
            "market_status": "unknown", "source_dataset": "blueneg", "source_label": label,
            "label_status": "research_eligible_dataset_declared", "data_evidence_grade": "S1-candidate",
            "expert_evidence_grade": "none", "roll_count": 1, "frame_count": 1,
            "aligned_public_frame_count": 1, "selected_first_pilot": True, "claim_ceiling": "test",
        })
    return {
        "schema_version": 1, "registry_id": "test",
        "source_datasets": {"blueneg": {"revision": "rev", "required_credit": "credit"}},
        "coverage": {"named_stock_data_s1_or_higher": 0, "named_stock_experts_s2_or_higher": 0,
                     "calibrated_stock_experts_s3": 0, "coverage_accounts_must_not_be_merged": True},
        "stocks": stocks,
    }


def _inputs(seal_a: bool = False) -> tuple[list[dict], list[dict], dict]:
    frames, rolls, files = [], [], []
    for label in ("a", "b"):
        frame = f"{label}1"
        preview = f"negative-preview-8bit/x/{frame}.png"
        proxy = f"pseudogt-8bit/x/{frame}.png"
        frames.append({"film_type": label, "roll_id": frame, "filename": frame,
                       "preview_path": preview, "pseudogt_path": proxy, "alignment_available": True})
        rolls.append({"film_type": label, "roll_id": frame, "public_non_test_pseudogt_frames": 1,
                      "contains_official_test_frame": seal_a and label == "a"})
        files.extend([{"path": preview, "size": 10, "sha256": "a" * 64},
                      {"path": proxy, "size": 11, "sha256": "b" * 64}])
    inventory = {"repo_id": "ttgroup/blueneg-release", "revision": "rev", "files": files}
    return frames, rolls, inventory


def test_builds_stock_scoped_manifest() -> None:
    frames, rolls, inventory = _inputs()
    manifest, report = build_stock_pilot_acquisition(
        _registry(), frames, rolls, inventory, software_commit="abc", input_sha256=HASHES
    )
    assert manifest["file_count"] == 4
    assert manifest["bytes"] == 42
    assert report["source_crosscheck"]["passed"] is True


def test_seals_entire_official_test_roll() -> None:
    frames, rolls, inventory = _inputs(seal_a=True)
    manifest, report = build_stock_pilot_acquisition(
        _registry(), frames, rolls, inventory, software_commit="abc", input_sha256=HASHES
    )
    assert {row["film_stock_id"] for row in manifest["files"]} == {"b"}
    assert report["stock_summaries"]["a"]["sealed_official_test_rolls"] == 1


def test_rejects_missing_lfs_digest() -> None:
    frames, rolls, inventory = _inputs()
    inventory["files"][0]["sha256"] = None
    with pytest.raises(StockAcquisitionError, match="missing LFS"):
        build_stock_pilot_acquisition(
            _registry(), frames, rolls, inventory, software_commit="abc", input_sha256=HASHES
        )


def test_alignment_without_public_proxy_is_reported_not_downloaded() -> None:
    frames, rolls, inventory = _inputs()
    inventory["files"] = [row for row in inventory["files"] if row["path"] != frames[0]["pseudogt_path"]]
    registry = _registry()
    registry["stocks"][0]["aligned_public_frame_count"] = 0
    rolls[0]["public_non_test_pseudogt_frames"] = 0
    manifest, report = build_stock_pilot_acquisition(
        registry, frames, rolls, inventory, software_commit="abc", input_sha256=HASHES
    )
    assert manifest["file_count"] == 3
    assert report["stock_summaries"]["a"]["aligned_but_proxy_unavailable_frames"] == 1
