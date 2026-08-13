from __future__ import annotations

import json
from pathlib import Path

from src.eval.sigmoid_scanner_ao6_value_d1 import evaluate, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4im_sigmoid_scanner_ao6_value_d1_v1.json"


def test_p4im_exact_replay_and_parent_guards(tmp_path: Path) -> None:
    contract = load_contract(CONFIG)
    first = evaluate(contract, ROOT, contact_path=tmp_path / "a.png")
    second = evaluate(contract, ROOT, contact_path=tmp_path / "b.png")
    assert first == second
    assert not first["automatic_pass"]
    assert not first["blind_review_allowed"]
    assert not first["checks"]["chroma"]
    assert not first["checks"]["isolated"]
    assert first["checks"]["material_increment"]
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    changed = json.loads(json.dumps(contract))
    changed["ao6"]["required_bundle_sha256"] = "0" * 64
    try:
        evaluate(changed, ROOT)
    except ValueError as error:
        assert "bundle drift" in str(error)
    else:
        raise AssertionError("AO6 bundle drift was accepted")
