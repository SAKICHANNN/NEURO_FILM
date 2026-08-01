from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.ao6_tone_layout_hard_retrieval import (
    AO6ToneLayoutHardRetrievalError,
    _extract_tone_layout_features,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bm3_ao6_tone_layout_hard_retrieval_v1.json"


def test_bm3_contract_binds_bm2_close_and_bm1_oracle() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["source_rows"]) == 17
    assert validated["bm1_report"]["automatic_pass"] is True


def test_bm3_descriptor_is_fixed_bounded_and_colour_blind() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    luma = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    views = [np.repeat(luma[..., None], 3, axis=-1)]
    feature = _extract_tone_layout_features(
        root=ROOT, views=views, config=config
    )
    assert feature.shape == (1, 74)
    assert np.all((feature >= 0.0) & (feature <= 1.0))
    altered = views[0].copy()
    altered[..., 1] = np.clip(altered[..., 1] + 0.1, 0.0, 1.0)
    with pytest.raises(AO6ToneLayoutHardRetrievalError):
        _extract_tone_layout_features(root=ROOT, views=[altered], config=config)


def test_bm3_rejects_chroma_or_operator_inputs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["selector"]["source_chroma_used"] = True
    with pytest.raises(AO6ToneLayoutHardRetrievalError):
        validate_contract(ROOT, config)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["selector"]["operator_signatures_used_at_inference"] = True
    with pytest.raises(AO6ToneLayoutHardRetrievalError):
        validate_contract(ROOT, config)
