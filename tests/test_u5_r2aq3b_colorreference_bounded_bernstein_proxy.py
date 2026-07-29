from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_u5_r2aq3b_colorreference_bounded_bernstein_proxy import (
    CONFIG_SHA256,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2aq3b_colorreference_bounded_bernstein_proxy_v1.json"
)


def test_contract_freezes_one_bounded_family() -> None:
    config = load_config(CONFIG, CONFIG_SHA256)
    assert config["candidate_family"]["degrees"] == [1, 2, 3]
    assert config["candidate_family"]["post_fit_clipping"] is False
    assert config["target_space"]["coefficient_lower_bound"] == 0.0
    assert config["target_space"]["coefficient_upper_bound"] == 1.2
    assert config["render_allowed"] is False
    assert config["production_integration_allowed"] is False


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["candidate_family"]["degrees"].append(4)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(path, CONFIG_SHA256)
