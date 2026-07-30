from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.public_paired_film_source import (
    PublicPairedFilmSourceError,
    audit_public_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (
        ROOT / "configs/u5_r2bi0_public_paired_film_source_refresh_v1.json"
    ).read_text(encoding="utf-8")
)


def _fetcher(*, with_data: bool = False, with_license: bool = False):
    repo = CONFIG["sources"][0]
    emulsion = CONFIG["sources"][1]
    payloads = {
        repo["repository_api_url"]: {
            "default_branch": "master",
            "license": {"spdx_id": "MIT"} if with_license else None,
        },
        repo["repository_api_url"] + "/commits/master": {
            "sha": "a" * 40,
            "commit": {"tree": {"sha": "b" * 40}},
        },
        repo["tree_api_template"].format(tree_sha="b" * 40): {
            "truncated": False,
            "tree": [
                {"path": "README.md"},
                *([{"path": "data/pairs.json"}] if with_data else []),
                *([{"path": "LICENSE"}] if with_license else []),
            ],
        },
        emulsion["project_page_url"]: (
            '<a href="https://zenodo.org/records/1">data</a>'
            '<a href="https://creativecommons.org/licenses/by/4.0/">cc</a>'
            if with_data and with_license
            else '<a href="https://github.com/musicofmusix">GitHub</a>'
        ),
        emulsion["author_repositories_api_url"]: (
            [{"name": "emulating-emulsion"}] if with_data else [{"name": "sairi"}]
        ),
    }

    def fetch(url: str, _config: dict) -> bytes:
        value = payloads[url]
        if isinstance(value, str):
            return value.encode()
        return json.dumps(value).encode()

    return fetch


def test_current_shape_closes_without_data_or_license() -> None:
    report = audit_public_sources(CONFIG, fetcher=_fetcher())
    assert report["eligible_source_count"] == 0
    assert report["decision"] == "no_eligible_source_current_public_surface_closed"
    assert report["network_facts"]["image_or_dataset_payload_requests"] == 0


def test_dataset_and_license_are_both_required() -> None:
    for data, license_ in ((True, False), (False, True)):
        report = audit_public_sources(
            CONFIG, fetcher=_fetcher(with_data=data, with_license=license_)
        )
        assert report["eligible_source_count"] == 0
    report = audit_public_sources(
        CONFIG, fetcher=_fetcher(with_data=True, with_license=True)
    )
    assert report["eligible_source_count"] == 2


def test_contract_rejects_payload_or_training_drift() -> None:
    for key in ("dataset_download_allowed", "image_request_allowed", "training_allowed"):
        changed = json.loads(json.dumps(CONFIG))
        changed[key] = True
        with pytest.raises(PublicPairedFilmSourceError):
            audit_public_sources(changed, fetcher=_fetcher())
