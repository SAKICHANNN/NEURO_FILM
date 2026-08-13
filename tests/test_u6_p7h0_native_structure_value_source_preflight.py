from __future__ import annotations

import json
import re
from pathlib import Path

from src.eval.rawpixls_confirmation_preflight import validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p7h0_native_structure_value_source_preflight_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_p7h0_contract_is_bound_and_program_wide_fresh() -> None:
    config = _config()
    validate_contract(ROOT, config)
    candidates = config["candidates"]
    assert len(candidates) == 12
    assert len({row["make"] for row in candidates}) == 12
    assert sum(row["bytes_reported_mb"] for row in candidates) < 360.0

    prior_ids: set[int] = set()
    prior_hashes: set[str] = set()
    prior_models: set[tuple[str, str]] = set()
    for binding in config["preflight"]["comparison_manifests"]:
        payload = json.loads((ROOT / binding["path"]).read_text(encoding="utf-8"))
        text = json.dumps(payload)
        prior_ids.update(
            int(value)
            for value in re.findall(r"raw\.pixls\.us/getfile\.php/(\d+)/", text)
        )
        for row in payload:
            if not isinstance(row, dict):
                continue
            for key in ("raw_sha256", "sha256", "source_sha256"):
                value = row.get(key)
                if isinstance(value, str) and len(value) == 64:
                    prior_hashes.add(value.casefold())
            if row.get("make") and row.get("model"):
                prior_models.add(
                    (str(row["make"]).casefold(), str(row["model"]).casefold())
                )

    assert {row["repository_id"] for row in candidates}.isdisjoint(prior_ids)
    assert {row["sha256"] for row in candidates}.isdisjoint(prior_hashes)
    assert {
        (str(row["make"]).casefold(), str(row["model"]).casefold())
        for row in candidates
    }.isdisjoint(prior_models)


def test_p7h0_opens_only_the_fixed_four_arm_value_ablation() -> None:
    config = _config()
    assert config["training_allowed"] is False
    assert config["operator_fitting_allowed"] is False
    assert config["next_leaf"]["source_context_built_from_original_only"] is True
    assert config["next_leaf"]["blind_review_allowed_before_automatic_pass"] is False
    assert config["next_leaf"]["fixed_arms"] == [
        "fixed_ao6_colour_only_t15_c35",
        "matched_scanner_only_ao6_t15_c35",
        "p4hu_scanner_physical_only_diagnostic",
        "p4hu_scanner_then_fixed_ao6_t15_c35",
    ]
