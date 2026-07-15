from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.roll2film.manifests import (
    FILMSET_DOMAINS,
    FilmSetEvidenceConfig,
    FilmSetManifestError,
    PairBlindFilmSetView,
    build_filmset_evidence,
)


def _fixture(root: Path, train_count: int = 12, test_count: int = 3) -> None:
    duplicate = np.random.default_rng(999).integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    for split, count in (("train", train_count), ("test", test_count)):
        for directory in FILMSET_DOMAINS.values():
            (root / split / directory).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            if split == "train" and index in (0, 1):
                source = duplicate
            else:
                source = np.random.default_rng(index + (0 if split == "train" else 100)).integers(
                    0,
                    256,
                    size=(24, 32, 3),
                    dtype=np.uint8,
                )
            name = f"image_{split}_{index:03d}.png"
            for domain, directory in FILMSET_DOMAINS.items():
                rendered = source if domain == "input" else np.roll(source, shift=list(FILMSET_DOMAINS).index(domain), axis=2)
                Image.fromarray(rendered).save(root / split / directory / name)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_filmset_evidence_is_pair_blind_and_test_payload_stays_sealed(tmp_path: Path) -> None:
    root = tmp_path / "FilmSet"
    output = tmp_path / "evidence"
    _fixture(root)
    config = FilmSetEvidenceConfig(
        root=root,
        output_dir=output,
        expected_train=12,
        expected_test=3,
        expected_files=None,
        expected_bytes=None,
        software_commit="test-commit",
    )
    result = build_filmset_evidence(config)
    view = PairBlindFilmSetView.from_manifests(
        result.manifest_paths["source_train"],
        result.manifest_paths["target_train"],
    )

    source_ids = {row["content_id"] for row in view.source_rows}
    target_ids = {row["content_id"] for row in view.target_rows}
    assert not source_ids & target_ids
    assert result.report["decision"] == "pass"
    assert result.report["duplicate_canaries"]["largest_cluster"] >= 2
    assert all(value == 0 for value in result.report["leakage"]["content_id_intersections"].values())
    assert all(
        value == 0
        for value in result.report["leakage"]["duplicate_cluster_intersections"].values()
    )

    final_rows = _read_jsonl(result.manifest_paths["final_628_lockbox"])
    assert len(final_rows) == 3 * 4
    assert {row["payload_access"] for row in final_rows} == {"final_evaluator_only"}
    assert all(row["width"] is None and row["dhash64"] is None for row in final_rows)


def test_training_loader_rejects_lockbox_manifest(tmp_path: Path) -> None:
    root = tmp_path / "FilmSet"
    output = tmp_path / "evidence"
    _fixture(root)
    result = build_filmset_evidence(
        FilmSetEvidenceConfig(
            root=root,
            output_dir=output,
            expected_train=12,
            expected_test=3,
            expected_files=None,
            expected_bytes=None,
        )
    )

    with pytest.raises(FilmSetManifestError, match="lockbox role rejected"):
        PairBlindFilmSetView.from_manifests(
            result.manifest_paths["internal_dev_lockbox"],
            result.manifest_paths["target_train"],
        )


def test_filmset_evidence_is_byte_deterministic_with_resume_cache(tmp_path: Path) -> None:
    root = tmp_path / "FilmSet"
    output = tmp_path / "evidence"
    _fixture(root)
    config = FilmSetEvidenceConfig(
        root=root,
        output_dir=output,
        expected_train=12,
        expected_test=3,
        expected_files=None,
        expected_bytes=None,
        software_commit="same-commit",
    )
    first = build_filmset_evidence(config)
    first_hashes = first.report["manifest_sha256"]
    first_report = (output / "report.json").read_bytes()
    second = build_filmset_evidence(config)

    assert second.report["manifest_sha256"] == first_hashes
    assert (output / "report.json").read_bytes() == first_report


def test_filmset_evidence_fails_closed_on_basename_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "FilmSet"
    _fixture(root)
    (root / "train" / "Velvia" / "image_train_000.png").rename(
        root / "train" / "Velvia" / "wrong_name.png"
    )

    with pytest.raises(FilmSetManifestError, match="basename parity"):
        build_filmset_evidence(
            FilmSetEvidenceConfig(
                root=root,
                output_dir=tmp_path / "evidence",
                expected_train=12,
                expected_test=3,
                expected_files=None,
                expected_bytes=None,
            )
        )
