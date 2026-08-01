from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_pair_preflight import (
    _asset_url,
    select_pair_names,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ay3s0_fivek_fresh_pair_preflight_v1.json"
)


def test_contract_is_head_only_and_bounded() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    assert config["network_preflight"]["method"] == "HEAD"
    assert config["network_preflight"]["maximum_requests"] == 128
    assert config["image_download_allowed"] is False


def test_selection_is_deterministic_disjoint_and_sorted() -> None:
    licensed = [f"a{index:04d}" for index in range(100)]
    retained = {"a0001", "a0002", "a0003"}
    first = select_pair_names(
        licensed_names=licensed,
        retained_names=retained,
        seed=17,
        count=16,
    )
    second = select_pair_names(
        licensed_names=list(reversed(licensed)),
        retained_names=retained,
        seed=17,
        count=16,
    )
    assert first == second
    assert first == sorted(first)
    assert not set(first) & retained


def test_official_asset_url_percent_encodes_spaces() -> None:
    url = _asset_url(
        "https://data.csail.mit.edu/graphics/fivek/",
        "img/dng/a0532-jmacdscf0021 (1).dng",
    )
    assert url == (
        "https://data.csail.mit.edu/graphics/fivek/"
        "img/dng/a0532-jmacdscf0021%20%281%29.dng"
    )
