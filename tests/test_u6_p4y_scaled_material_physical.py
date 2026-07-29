from __future__ import annotations

import json
from pathlib import Path

from src.eval.scaled_material_physical import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4y_scaled_material_physical_v1.json"


def test_p4y_binds_exact_proxy_scale_without_photographs() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    p4r, _, p4s, p4d = validate_contract(ROOT, config)
    assert p4r["decision"] == "retain_generic_scanner_convolved_signature_only"
    assert p4s["model"]["amplitude_fit_allowed"] is False
    assert p4d["model"]["display_rgb_noise_allowed"] is False
    assert config["candidate"]["source_amplitude_scale"] == 0.125
    assert config["candidate"]["grain_optical_density_by_rgb_layer"] == [
        0.005,
        0.00625,
        0.004375,
    ]
    assert config["execution"]["photographic_render_allowed"] is False
