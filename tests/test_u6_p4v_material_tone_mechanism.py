from __future__ import annotations

import json
from pathlib import Path

from src.eval.material_tone_mechanism import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4v_material_tone_mechanism_v1.json"


def test_p4v_binds_only_the_closed_p4u_candidate() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runtime, _, p4u, p4t = validate_contract(ROOT, config)
    assert len(runtime.eligible_ids) == 16
    assert p4u["execution"]["post_result_retuning_allowed"] is False
    assert p4t["candidate"]["sigma_yx_pixels"] == [0.9, 0.65]
    assert config["candidate"]["parameter_changes_allowed"] is False
    assert config["execution"]["new_render_files_allowed"] is False
