from __future__ import annotations

import json
from pathlib import Path
import re

from src.eval.rawpixls_confirmation_preflight import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk1s_log_chroma_fresh_source_v1.json"
PRIOR_CONFIGS = (
    ROOT / "configs/u5_r2ai1s_rawpixls_confirmation_source_preflight_v1.json",
    ROOT / "configs/u5_r2ao7s_fresh_rawpixls_source_preflight_v1.json",
    ROOT / "configs/u6_p8bn_fresh_strength_population_v1.json",
    ROOT / "configs/u6_p8bp_fresh_native_standard_confirmation_v1.json",
    ROOT / "configs/u5_r2bh0s_fixed_bank_oracle_source_preflight_v1.json",
    ROOT / "configs/u5_r2bh1s_global_policy_source_preflight_v1.json",
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


def test_bk1s_contract_is_frozen_before_pixel_access() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    rows = config["candidates"]
    assert config["status"] == "contract_frozen_acquisition_ready"
    assert config["source"]["api_snapshot_sha256"] == (
        "8dc5f5c74e20cc3a38f4548d53bba4bb516835e2a859e80bf86e7ac6283b3aa3"
    )
    assert len(rows) == 12
    assert len({row["make"].casefold() for row in rows}) == 12
    assert len({row["repository_id"] for row in rows}) == 12
    assert sum(row["bytes_reported_mb"] for row in rows) == 191.15
    assert config["maximum_download_bytes"] == 220200960
    assert config["next_leaf"]["fixed_arms"] == [
        "fixed_bk0_log_chroma",
        "fixed_ao6_colour_only_t15_c35",
        "safe_rich",
    ]
    assert not config["training_allowed"]
    assert not config["operator_fitting_allowed"]
    assert not config["next_leaf"]["strength_retuning_allowed"]
    assert not config["next_leaf"]["router_training_allowed"]


def test_bk1s_sources_are_disjoint_from_all_prior_rawpixls_pools() -> None:
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


def test_bk1s_selection_excludes_obvious_nonphotographic_metadata() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    forbidden = re.compile(r"\b(scanner|chart|flat|target|film|buggy)\b", re.I)
    for row in config["candidates"]:
        text = " ".join(
            str(row[key]) for key in ("make", "model", "mode", "filename")
        )
        assert forbidden.search(text) is None
