import json
from pathlib import Path

import pytest

from src.real_film.filmr import FilmRContractError, build_pair_manifest, validate_article


def _config() -> dict:
    return {
        "article_id": 1,
        "article_version": 2,
        "doi": "doi",
        "expected_files": 2,
        "expected_bytes": 2,
        "expected_pairs": 1,
        "license": {"name": "CC BY 4.0", "url": "https://license"},
        "claim_ceiling": "test ceiling",
    }


def _article() -> dict:
    return {
        "id": 1,
        "version": 2,
        "doi": "doi",
        "license": {"name": "CC BY 4.0", "url": "https://license"},
        "files": [
            {"id": 1, "name": "velvia50_half_1.jpg", "size": 1, "supplied_md5": "0" * 32, "download_url": "https://one"},
            {"id": 2, "name": "velvia50_half_1_restored.jpg", "size": 1, "supplied_md5": "0" * 32, "download_url": "https://two"},
        ],
    }


def test_article_contract_rejects_drift() -> None:
    article = _article()
    assert len(validate_article(article, _config())) == 2
    article["license"]["name"] = "unknown"
    with pytest.raises(FilmRContractError, match="license name mismatch"):
        validate_article(article, _config())


def test_pair_manifest_keeps_group_fields_unknown(tmp_path: Path) -> None:
    article = _article()
    files = validate_article(article, _config())
    for item in files:
        (tmp_path / item["name"]).write_bytes(b"x")
        item["supplied_md5"] = "9dd4e461268c8034f5c8564e155c67a6"
    rows = build_pair_manifest(files, output_dir=tmp_path, config=_config())
    assert rows[0]["filename_family_claim"] == "velvia50"
    assert rows[0]["physical_roll_id"] is None
    assert rows[0]["scanner_id"] is None
