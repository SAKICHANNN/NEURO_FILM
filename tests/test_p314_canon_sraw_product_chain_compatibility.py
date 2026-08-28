from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p314_canon_sraw_product_chain_compatibility import (
    _canonical_bytes,
    _load_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p314_canon_sraw_product_chain_compatibility_v1.json"


def test_p314_contract_is_exact_six_file_no_core_change_product_chain() -> None:
    config = _load_json(CONFIG)
    p313 = json.loads(
        (ROOT / config["bindings"]["p313_contract"]["path"]).read_text(encoding="utf-8")
    )
    execution = config["execution"]
    assert len(p313["rows"]) == 6
    assert len({row["model"] for row in p313["rows"]}) == 6
    assert execution == {
        "dust": 0.0,
        "grain": 0.0,
        "halation": 0.0,
        "look_amount": 1.0,
        "output_bit_depth": 8,
        "output_format": "PNG",
        "profile_path": "configs/render_profiles/safe_rich_product_v1.json",
        "scratch_root": "outputs/eval/p314_canon_sraw_product_chain_v1",
        "style": "ektar_100",
        "tile_size": None,
    }
    assert "candidate 3" in config["claim_ceiling"]
    assert "no vendor-exact colour" in config["claim_ceiling"]


def test_p314_canonical_report_encoding_is_stable() -> None:
    value = {"z": [3, 2, 1], "a": {"value": True}}
    assert _canonical_bytes(value) == (
        b'{\n  "a": {\n    "value": true\n  },\n  "z": [\n'
        b"    3,\n    2,\n    1\n  ]\n}\n"
    )
