from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.eval.fivek_semantic_pseudopair_flow import (
    FiveKSemanticPseudoPairError,
    _resize_crop,
    build_pseudo_pairs,
    load_config,
    load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bk24_fivek_semantic_pseudopair_flow_v1.json"


def test_contract_and_inventory_are_frozen() -> None:
    config = load_config(ROOT, CONFIG)
    rows = load_rows(ROOT, config)
    assert len(rows) == 64
    assert config["source"]["fit_rows"] == 48
    assert config["source"]["confirmation_population_used"] is False
    assert config["feature_model"]["input"].startswith("luma")


def test_closed_decision_binds_repeat_exact_reports() -> None:
    decision = __import__("json").loads(
        (
            ROOT
            / "configs/u5_r2bk24_fivek_semantic_pseudopair_flow_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["decision"] == "retain_k1_identity_close_fixed_semantic_pseudopair"
    assert decision["repeat_report_byte_identical"] is True
    assert decision["automatic_pass"] is False
    assert decision["observations"]["semantic_pass_fraction"] == 0.0625


def test_contract_hash_drift_fails_closed() -> None:
    config = load_config(ROOT, CONFIG)
    tampered = copy.deepcopy(config)
    tampered["parent"]["decision_sha256"] = "0" * 64
    path = ROOT / "tmp" / "bk24-invalid.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(__import__("json").dumps(tampered), encoding="utf-8")
    with pytest.raises(FiveKSemanticPseudoPairError, match="hash drift"):
        load_config(ROOT, path)
    path.unlink()


def test_pseudo_pairs_exclude_identity_and_preserve_control_multiset() -> None:
    rng = np.random.default_rng(24)
    features = rng.normal(size=(4, 6, 8))
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    colours = rng.uniform(size=(4, 6, 3))
    source, semantic, control, facts = build_pseudo_pairs(
        features,
        colours,
        features.copy(),
        colours.copy(),
        topk_images=2,
        regularization=0.1,
        iterations=20,
        control_seed=9,
    )
    assert source.shape == semantic.shape == control.shape == (24, 3)
    assert facts["same_identity_matches"] == 0
    assert facts["target_colour_multiset_sha256"] == facts["control_colour_multiset_sha256"]
    assert not np.array_equal(semantic, control)


def test_resize_repairs_only_float32_endpoint_epsilon() -> None:
    image = np.ones((19, 31, 3), dtype=np.float32)
    output = _resize_crop(image)
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    invalid = image.copy()
    invalid[4:8, 4:8] = 1.01
    with pytest.raises(FiveKSemanticPseudoPairError, match="material"):
        _resize_crop(invalid)
