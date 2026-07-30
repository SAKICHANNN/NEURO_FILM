from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import src.eval.bw_uniform_grain_source as source_module
from src.eval.bw_uniform_grain_source import (
    BWUniformGrainSourceError,
    acquire_files,
    normalize_api_payload,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4z_bw_uniform_grain_source_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _api_payload(config: dict) -> dict:
    pages = []
    for index, row in enumerate(config["files"], start=1):
        pages.append(
            {
                "pageid": index,
                "title": row["title"],
                "imageinfo": [
                    {
                        "url": row["original_url"],
                        "descriptionurl": row["file_page_url"],
                        "sha1": row["api_sha1"],
                        "size": row["expected_bytes"],
                        "width": row["width"],
                        "height": row["height"],
                        "mime": row["mime"],
                        "user": row["uploader"],
                        "timestamp": row["upload_timestamp"],
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC0"},
                            "LicenseUrl": {
                                "value": (
                                    "http://creativecommons.org/publicdomain/"
                                    "zero/1.0/deed.en"
                                )
                            },
                            "UsageTerms": {
                                "value": (
                                    "Creative Commons Zero, Public Domain Dedication"
                                )
                            },
                        },
                    }
                ],
            }
        )
    return {"query": {"pages": pages}}


def test_frozen_source_is_bounded_new_and_fit_forbidden() -> None:
    config = _config()
    validate_contract(config)
    assert len(config["files"]) == 3
    assert {row["film_stock_id"] for row in config["files"]} == {
        "kodak-tmax-100",
        "kodak-trix-400",
    }
    assert config["selection"]["expected_total_bytes"] == 110_295_676
    assert config["training_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["stock_calibration_allowed"] is False


def test_live_metadata_must_match_every_frozen_identity() -> None:
    config = _config()
    rows = normalize_api_payload(_api_payload(config), config)
    assert len(rows) == 3
    drift = _api_payload(config)
    drift["query"]["pages"][0]["imageinfo"][0]["timestamp"] = "2025-01-01"
    with pytest.raises(BWUniformGrainSourceError, match="drift"):
        normalize_api_payload(drift, config)


def test_contract_rejects_invented_process_or_expanded_authority() -> None:
    config = deepcopy(_config())
    config["files"][0]["process_type"] = "D-76"
    with pytest.raises(BWUniformGrainSourceError, match="process_type"):
        validate_contract(config)
    config = deepcopy(_config())
    config["operator_fitting_allowed"] = True
    with pytest.raises(BWUniformGrainSourceError, match="fitting"):
        validate_contract(config)


def test_contract_rejects_unbounded_or_duplicate_derivative() -> None:
    config = deepcopy(_config())
    config["selection"]["maximum_total_bytes"] = 1
    with pytest.raises(BWUniformGrainSourceError, match="byte budget"):
        validate_contract(config)
    config = deepcopy(_config())
    config["files"][0]["title"] = "File:Tmax100 GrainNoProfile.png"
    with pytest.raises(BWUniformGrainSourceError, match="file set"):
        validate_contract(config)


def test_acquisition_manifest_paths_are_root_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    def fake_download(row: dict, destination: Path) -> dict:
        return {
            "path": str(destination),
            "bytes": row["expected_bytes"],
            "sha1": row["api_sha1"],
            "sha256": "0" * 64,
            "reused": False,
        }

    monkeypatch.setattr(source_module, "_download_exact", fake_download)
    manifest = acquire_files(tmp_path, config)
    assert all(not Path(row["path"]).is_absolute() for row in manifest["rows"])
    assert manifest["rows"][0]["path"].startswith(
        "data/external/wikimedia_bw_uniform_grain_v1/"
    )
