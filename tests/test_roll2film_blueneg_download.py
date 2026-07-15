from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.roll2film.blueneg_download import (
    BlueNegDownloadConfig,
    BlueNegDownloadError,
    download_blueneg_acquisition,
    sha256_file,
)


def _fixture(tmp_path: Path) -> tuple[BlueNegDownloadConfig, Path]:
    root = tmp_path / "data"
    remote = tmp_path / "remote"
    files = []
    for relative, payload in (
        ("negative-preview-8bit/x/a.png", b"preview"),
        ("pseudogt-8bit/x/a.png", b"target"),
    ):
        source = remote / Path(*relative.split("/"))
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        files.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "repo_id": "fixture/repo",
                "revision": "a" * 40,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata_report.json"
    metadata.write_text("{}", encoding="utf-8")
    return (
        BlueNegDownloadConfig(
            root=root,
            acquisition_manifest=acquisition,
            acquisition_manifest_sha256=sha256_file(acquisition),
            metadata_report=metadata,
            metadata_report_sha256=sha256_file(metadata),
            output_report=tmp_path / "download_report.json",
            repo_id="fixture/repo",
            revision="a" * 40,
            expected_files=2,
            expected_bytes=13,
            workers=2,
            software_commit="fixture",
        ),
        remote,
    )


def test_exact_download_repairs_corruption_and_reruns_byte_identically(
    tmp_path: Path,
) -> None:
    config, remote = _fixture(tmp_path)
    corrupt = config.root / "negative-preview-8bit" / "x" / "a.png"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"bad")
    force_values = []

    def fetch(relative: str, force: bool) -> Path:
        force_values.append(force)
        target = config.root / Path(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / Path(*relative.split("/")), target)
        return target

    first = download_blueneg_acquisition(config, fetch)
    first_bytes = config.output_report.read_bytes()
    second = download_blueneg_acquisition(config, fetch)

    assert first["all_sizes_and_lfs_sha256_verified"] is True
    assert first["report_sha256"] == second["report_sha256"]
    assert config.output_report.read_bytes() == first_bytes
    assert sorted(force_values) == [False, True]


def test_download_rejects_manifest_external_lane_file(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    extra = config.root / "pseudogt-8bit" / "x" / "extra.png"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"extra")

    with pytest.raises(BlueNegDownloadError, match="manifest-external"):
        download_blueneg_acquisition(config, lambda _path, _force: extra)


def test_download_rejects_path_traversal(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    payload = json.loads(config.acquisition_manifest.read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "negative-preview-8bit/../../escape.png"
    config.acquisition_manifest.write_text(json.dumps(payload), encoding="utf-8")
    config = BlueNegDownloadConfig(
        **{
            **config.__dict__,
            "acquisition_manifest_sha256": sha256_file(config.acquisition_manifest),
        }
    )

    with pytest.raises(BlueNegDownloadError, match="unsafe acquisition path"):
        download_blueneg_acquisition(config, lambda _path, _force: tmp_path)
