from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_pair_acquisition import (
    resolve_configured_owned_root,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq0s1_fivek_casebank_acquisition_v1.json"


def test_casebank_acquisition_binds_two_exact_passing_preflights() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["preflight_manifest"]["rows"]) == 512
    assert config["preflight"]["expected_assets"] == 1024
    assert config["preflight"]["expected_bytes"] == 29_627_684_806
    assert (
        config["preflight"]["manifest_sha256"]
        == config["preflight"]["repeat_manifest_sha256"]
    )
    assert (
        config["preflight"]["report_sha256"]
        == config["preflight"]["repeat_report_sha256"]
    )


def test_casebank_acquisition_uses_live_repository_data_junction() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    resolved = resolve_configured_owned_root(ROOT, config["ownership"])
    assert resolved == ROOT / "data/external/fivek-bq0-casebank512-v1"
    assert resolved.resolve().is_relative_to((ROOT / "data").resolve())
    assert config["ownership"]["foreign_resource_mutation_allowed"] is False
