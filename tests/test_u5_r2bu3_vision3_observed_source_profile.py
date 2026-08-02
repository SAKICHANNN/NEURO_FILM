from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vision3_observed_source_profile import (
    ObservedSourceCompilerError,
    canonical_json,
    compile_observed_bundle,
    load_contract,
)
from src.film_physics.observed_source_profile import (
    EXECUTION_AUTHORITY,
    Vision3ObservedSourceProfileBundle,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bu3_vision3_observed_source_profile_v1.json"


def test_contract_rejects_execution_authority_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["bundle"]["execution_authority"] = "renderable"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ObservedSourceCompilerError, match="contract drift"):
        load_contract(changed)


def test_compiler_is_exact_roundtrippable_and_non_renderable() -> None:
    config = load_contract(CONFIG)
    first_report, first_bundle = compile_observed_bundle(config, ROOT)
    second_report, second_bundle = compile_observed_bundle(config, ROOT)
    assert canonical_json(first_report) == canonical_json(second_report)
    assert canonical_json(first_bundle) == canonical_json(second_bundle)
    rebuilt = Vision3ObservedSourceProfileBundle.from_dict(first_bundle)
    assert rebuilt.to_dict() == first_bundle
    assert rebuilt.identity() == first_report["bundle_id"]
    assert rebuilt.execution_authority == EXECUTION_AUTHORITY
    assert first_report["forbidden_serialized_fields_found"] == []


def test_compiler_preserves_observed_sample_counts_and_parent_ids() -> None:
    report, bundle = compile_observed_bundle(load_contract(CONFIG), ROOT)
    assert report["compiler_pass"] is True
    assert report["gate_results"] == {
        "mtf_parent": True,
        "granularity_parent": True,
        "stock_and_channel_counts": True,
        "exact_roundtrip": True,
        "forbidden_serialized_fields_absent": True,
        "execution_authority_is_non_renderable": True,
    }
    assert len(bundle["profiles"]) == 3
    for profile in bundle["profiles"]:
        assert all(
            len(profile["mtf_curves"][channel]["coordinates"]) >= 8
            and len(profile["granularity_curves"][channel]["coordinates"]) >= 16
            for channel in profile["channel_order"]
        )
