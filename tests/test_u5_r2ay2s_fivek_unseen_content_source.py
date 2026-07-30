from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.fivek_unseen_content_source import _dhash, validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ay2s_fivek_unseen_content_source_v1.json"
)


def test_contract_binds_untouched_half_and_forbids_training() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    assert config["freeze"]["confirmation_row_start_one_based"] == 65
    assert config["training_allowed"] is False
    assert "camera-model OOD" in config["grouping_limit"]


def test_dhash_is_exact_and_detects_identical_content(tmp_path: Path) -> None:
    pixels = np.arange(72, dtype=np.uint8).reshape(8, 9)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(pixels, mode="L").save(first)
    Image.fromarray(pixels, mode="L").save(second)
    assert _dhash(first) == _dhash(second)
