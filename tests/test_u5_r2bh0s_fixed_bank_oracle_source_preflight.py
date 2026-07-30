from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2bh0s_fixed_bank_oracle_source_preflight_v1.json"
)
PRIOR_CONFIGS = (
    ROOT / "configs/u5_r2ai1s_rawpixls_confirmation_source_preflight_v1.json",
    ROOT / "configs/u5_r2ao7s_fresh_rawpixls_source_preflight_v1.json",
    ROOT / "configs/u6_p8bn_fresh_strength_population_v1.json",
    ROOT / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json",
)
LEGACY = (
    ROOT
    / "outputs/color_baseline/velvia50_rawpixls20_s0p58_gamutsafe/manifest.json"
)


def _repository_ids(payload: object) -> set[int]:
    return {
        int(value)
        for value in re.findall(
            r"raw\.pixls\.us/getfile\.php/(\d+)/(?:nice|exif)/",
            json.dumps(payload),
        )
    }


def test_bh0s_contract_is_frozen_before_pixel_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert len(rows) == 12
    assert len({row["make"] for row in rows}) == 12
    assert len({row["repository_id"] for row in rows}) == 12
    assert sum(row["bytes_reported_mb"] for row in rows) == 229.31
    assert config["maximum_download_bytes"] == 256 * 1024 * 1024
    assert config["next_leaf"]["complete_ranking_required"]
    assert len(config["next_leaf"]["fixed_arms"]) == 5
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["next_leaf"]["router_training_allowed"]


def test_bh0s_sources_are_disjoint_from_every_prior_rawpixls_pool() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    current_ids = {row["repository_id"] for row in config["candidates"]}
    current_hashes = {row["sha256"] for row in config["candidates"]}
    current_models = {
        (row["make"].casefold(), row["model"].casefold())
        for row in config["candidates"]
    }
    prior_ids: set[int] = set()
    prior_hashes: set[str] = set()
    prior_models: set[tuple[str, str]] = set()
    for path in (*PRIOR_CONFIGS, LEGACY):
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior_ids.update(_repository_ids(payload))
        rows = payload.get("candidates", []) if isinstance(payload, dict) else payload
        for row in rows:
            for key in ("sha256", "raw_sha256"):
                value = row.get(key)
                if isinstance(value, str) and len(value) == 64:
                    prior_hashes.add(value)
            if "make" in row and "model" in row:
                prior_models.add(
                    (row["make"].casefold(), row["model"].casefold())
                )
    assert current_ids.isdisjoint(prior_ids)
    assert current_hashes.isdisjoint(prior_hashes)
    assert current_models.isdisjoint(prior_models)


def test_bh0s_selection_excludes_obvious_nonphotographic_metadata() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = re.compile(r"\b(scanner|chart|flat|target|film)\b", re.I)
    for row in config["candidates"]:
        text = " ".join(
            str(row[key]) for key in ("make", "model", "mode", "filename")
        )
        assert forbidden.search(text) is None
