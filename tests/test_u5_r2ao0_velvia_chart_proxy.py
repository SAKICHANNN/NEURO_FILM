from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.run_u5_r2ao0_balica_velvia_chart_proxy_source import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.velvia_chart_proxy import extract_chart_proxy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao0_balica_velvia_chart_proxy_source_v1.json"


def test_frozen_source_contract_is_exact() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["article"]["doi"] == "10.1016/j.daach.2026.e00550"
    assert config["article"]["license"] == "CC BY 4.0"
    assert config["observed_evidence"]["film_stock_id"].endswith("Velvia 50")
    assert config["operator_fitting_if_passes"]
    assert not config["stock_response_claim_allowed"]


def test_extractor_fails_closed_on_nonmatching_asset(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "wrong.jpg"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="byte count"):
        extract_chart_proxy(path, config)
