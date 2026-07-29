from __future__ import annotations

import json
from pathlib import Path

from src.eval.global_material_amplitude import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4x_global_material_amplitude_v1.json"


def test_p4x_binds_one_global_label_blind_proxy() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    observed, p4t = validate_contract(ROOT, config)
    assert observed["automatic_pass"] is True
    assert observed["stock_labels_used"] is False
    assert p4t["candidate"]["sigma_yx_pixels"] == [0.9, 0.65]
    assert config["proxy_model"]["per_channel_scale_allowed"] is False
    assert config["proxy_model"]["stock_specific_fit_allowed"] is False
    assert config["execution"]["photographic_render_allowed"] is False
