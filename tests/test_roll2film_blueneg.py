from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.roll2film.blueneg import (
    BlueNegContractError,
    BlueNegEvidenceConfig,
    build_blueneg_metadata_evidence,
)


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> BlueNegEvidenceConfig:
    root = tmp_path / "blueneg"
    license_hash = _write(root / "LICENSE", b"Copyrighted by Tien-Tsin Wong")
    readme_hash = _write(root / "README.md", b"fixture")
    rows = []
    inventory = []
    films = {"film-a": ["a1", "a2"], "film-b": ["b1", "b2"]}
    for film, rolls in films.items():
        for roll in rolls:
            for index in range(4):
                filename = f"{roll}-{index}"
                preview = f"negative-preview-8bit/x/{filename}.preview.png"
                pseudo = f"pseudogt-8bit/x/{filename}.pseudogt.png"
                rows.append(
                    {
                        "filename": filename,
                        "partition": "x",
                        "is_testset": False,
                        "date": "20000101",
                        "roll_id": roll,
                        "film_type": film,
                        "preview_path": preview,
                        "pseudogt_path": pseudo,
                        "location": "fixture",
                        "scene_property": {"is_indoor": "outdoor", "is_daytime": "day"},
                    }
                )
                inventory.extend(
                    [
                        {"path": preview, "size": 10, "sha256": "a" * 64},
                        {"path": pseudo, "size": 5, "sha256": "b" * 64},
                    ]
                )
    meta_payload = json.dumps(rows).encode()
    meta_hash = _write(root / "meta.json", meta_payload)
    (root / "remote_inventory.json").write_text(
        json.dumps(
            {
                "repo_id": "fixture/repo",
                "revision": "f" * 40,
                "files": inventory,
            }
        ),
        encoding="utf-8",
    )
    return BlueNegEvidenceConfig(
        root=root,
        output_dir=tmp_path / "evidence",
        repo_id="fixture/repo",
        revision="f" * 40,
        metadata_sha256={
            "LICENSE": license_hash,
            "README.md": readme_hash,
            "meta.json": meta_hash,
        },
        expected={
            "metadata_rows": 16,
            "rolls": 4,
            "film_types": 2,
            "preview_files": 16,
            "preview_bytes": 160,
            "pseudogt_files": 16,
            "pseudogt_bytes": 80,
        },
        split_seed=7,
    )


def test_blueneg_evidence_is_whole_roll_and_byte_deterministic(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    first = build_blueneg_metadata_evidence(config)
    second = build_blueneg_metadata_evidence(config)

    assert first.report == second.report
    assert first.report["whole_roll_overlap"] == 0
    assert first.report["operator_rolls"] == 4
    assert first.report["matched_control_operator_rolls"] == 4
    assert first.report["roll_pool_counts"] == {
        "confirmatory_roll": 2,
        "development_roll": 2,
    }
    acquisition = json.loads(first.paths["acquisition"].read_text(encoding="utf-8"))
    assert acquisition["file_count"] == 32
    assert acquisition["bytes"] == 240


def test_blueneg_seals_entire_roll_containing_official_test_frame(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    meta_path = config.root / "meta.json"
    rows = json.loads(meta_path.read_text(encoding="utf-8"))
    rows[0]["is_testset"] = True
    payload = json.dumps(rows).encode()
    meta_path.write_bytes(payload)
    config = BlueNegEvidenceConfig(
        **{
            **config.__dict__,
            "metadata_sha256": {
                **config.metadata_sha256,
                "meta.json": hashlib.sha256(payload).hexdigest(),
            },
        }
    )
    result = build_blueneg_metadata_evidence(config)
    rolls = [json.loads(line) for line in result.paths["rolls"].read_text().splitlines()]
    sealed = next(row for row in rolls if row["roll_id"] == "a1")

    assert sealed["research_pool"] == "official_test_roll_lockbox"
    assert sealed["operator_eligible"] is False


def test_blueneg_fails_closed_on_remote_lane_size_drift(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    inventory_path = config.root / "remote_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["files"][0]["size"] += 1
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(BlueNegContractError, match="byte count mismatch"):
        build_blueneg_metadata_evidence(config)
