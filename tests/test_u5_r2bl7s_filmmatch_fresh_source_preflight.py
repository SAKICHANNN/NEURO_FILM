from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import (
    _comparison_rows,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bl7s_filmmatch_fresh_source_preflight_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_bl7s_contract_is_frozen_and_unseen_by_bound_prior_manifests() -> None:
    config = _config()
    validate_contract(ROOT, config)
    candidates = config["candidates"]
    assert len(candidates) == 18
    assert len({row["make"] for row in candidates}) == 18
    assert sum(row["bytes_reported_mb"] for row in candidates) < 420
    prior_ids: set[int] = set()
    for binding in config["preflight"]["comparison_manifests"]:
        text = (ROOT / binding["path"]).read_text(
            encoding="utf-8", errors="ignore"
        )
        prior_ids.update(
            int(value)
            for value in re.findall(
                r"raw\.pixls\.us/getfile\.php/(\d+)/", text
            )
        )
    assert not ({row["repository_id"] for row in candidates} & prior_ids)


def test_cumulative_prior_inventory_deduplicates_historical_replays() -> None:
    rows = _comparison_rows(ROOT, _config())
    identities = [(row["decoded_sha256"], row["dhash64"]) for row in rows]
    assert len(rows) == len(set(identities))
    assert len(rows) >= 100
