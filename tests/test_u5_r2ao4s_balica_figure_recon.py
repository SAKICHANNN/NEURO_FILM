from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image
import pytest

from scripts.run_u5_r2ao4s_balica_figure_recon import (
    CONFIG_SHA256,
    load_config,
)
from src.real_film.figure_recon import build_vertical_contact_sheet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao4s_balica_additional_figure_recon_v1.json"


def test_config_identity_and_closed_claims() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert [row["figure"] for row in config["assets"]] == [6, 11]
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
    assert config["stock_response_claim_allowed"] is False


def test_config_hash_fail_closed() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(CONFIG, expected_sha256="0" * 64)


def test_contact_sheet_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    Image.new("RGB", (96, 48), (10, 20, 30)).save(first, format="JPEG")
    Image.new("RGB", (48, 96), (40, 50, 60)).save(second, format="JPEG")
    out_a = build_vertical_contact_sheet(
        [(6, first), (11, second)], tmp_path / "sheet_a.png"
    )
    out_b = build_vertical_contact_sheet(
        [(6, first), (11, second)], tmp_path / "sheet_b.png"
    )
    assert out_a["sha256"] == out_b["sha256"]
    assert out_a["width"] == 96
    assert out_a["height"] == 208
