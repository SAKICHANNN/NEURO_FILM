from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.filmcase.lineage import ManifestAuditError, audit_manifest, write_audit_outputs


def _image(path: Path, value: int) -> None:
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def _manifest_row(path: str, sha256: str, split: str = "train") -> dict[str, object]:
    return {
        "path": path,
        "source": "Flickr API",
        "license": "unknown-flickr-user-content",
        "split": split,
        "task": "film_lora_training",
        "style": "velvia_50",
        "redistributable": False,
        "width": 32,
        "height": 32,
        "bytes": 100,
        "sha256": sha256,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_unresolved_rows_are_quarantined_and_do_not_enter_case_memory(tmp_path: Path) -> None:
    image_dir = tmp_path / "data" / "film_domain" / "velvia_50"
    image_dir.mkdir(parents=True)
    _image(image_dir / "unknown.jpg", 40)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_manifest_row("data/film_domain/velvia_50/unknown.jpg", "a" * 64)])

    result = audit_manifest(manifest, root=tmp_path)

    row = result.rows[0]
    assert row["lineage_status"] == "unresolved"
    assert row["filmcase_split"] == "quarantine"
    assert row["filmcase_eligibility"] == "quarantined_missing_group_lineage"
    assert result.report["gate_status"] == "blocked_missing_group_lineage"


def test_source_grouped_rows_share_a_group_split_and_are_research_only(tmp_path: Path) -> None:
    image_dir = tmp_path / "data" / "film_domain" / "velvia_50"
    image_dir.mkdir(parents=True)
    _image(image_dir / "one.jpg", 10)
    _image(image_dir / "two.jpg", 200)
    (image_dir / "metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"file_name": "one.jpg", "photo_id": "one", "owner_id": "same-owner"}),
                json.dumps({"file_name": "two.jpg", "photo_id": "two", "owner_id": "same-owner"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row("data/film_domain/velvia_50/one.jpg", "1" * 64, "train"),
            _manifest_row("data/film_domain/velvia_50/two.jpg", "2" * 64, "val"),
        ],
    )

    result = audit_manifest(manifest, root=tmp_path)

    assert {row["lineage_status"] for row in result.rows} == {"resolved_group"}
    assert {row["source_group_id"] for row in result.rows} == {"uploader:same-owner"}
    assert len({row["filmcase_split"] for row in result.rows}) == 1
    assert {row["filmcase_eligibility"] for row in result.rows} == {"quarantined_missing_group_lineage"}
    assert result.report["gate_status"] == "ready_for_group_split"


def test_duplicate_audit_reports_exact_and_perceptual_duplicates(tmp_path: Path) -> None:
    image_dir = tmp_path / "data" / "film_domain" / "velvia_50"
    image_dir.mkdir(parents=True)
    _image(image_dir / "one.jpg", 100)
    _image(image_dir / "two.jpg", 100)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row("data/film_domain/velvia_50/one.jpg", "d" * 64),
            _manifest_row("data/film_domain/velvia_50/two.jpg", "d" * 64),
        ],
    )

    result = audit_manifest(manifest, root=tmp_path)

    assert result.report["duplicate_audit"]["exact_duplicate_groups"] == 1
    assert result.report["duplicate_audit"]["perceptual_duplicate_pairs"] == 1
    assert any(issue["kind"] == "exact_duplicate" for issue in result.report["issues"])


def test_audit_outputs_are_new_files_and_legacy_manifest_is_unchanged(tmp_path: Path) -> None:
    image_dir = tmp_path / "data" / "film_domain" / "velvia_50"
    image_dir.mkdir(parents=True)
    _image(image_dir / "one.jpg", 12)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_manifest_row("data/film_domain/velvia_50/one.jpg", "f" * 64)])
    before = manifest.read_bytes()

    result = audit_manifest(manifest, root=tmp_path)
    report_path = tmp_path / "outputs" / "report.json"
    v2_path = tmp_path / "outputs" / "manifest_v2.jsonl"
    write_audit_outputs(result, report_path=report_path, manifest_v2_path=v2_path)

    assert manifest.read_bytes() == before
    assert report_path.exists()
    assert v2_path.exists()
    assert json.loads(v2_path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_invalid_manifest_row_fails_loudly(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"source": "missing path"}) + "\n", encoding="utf-8")

    with pytest.raises(ManifestAuditError, match="missing path"):
        audit_manifest(manifest, root=tmp_path)
