from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.film_physics.bounded_photographic_profile import (
    canonical_profile_bytes,
    validate_bounded_photographic_profile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4hk_bounded_photographic_profile_bundle_v1.json"


def test_p4hk_contract_freezes_profile_scope() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["bundle"]["rank_bins"] == 65536
    assert payload["bundle"]["canonical_row_block_height"] == 128
    assert payload["bundle"]["calibrated_stock_or_scanner_claimed"] is False
    assert payload["fixture"]["runs"] == 2


def test_profile_validator_rejects_unknown_fields() -> None:
    payload = {"schema": "neuro-film.bounded-photographic-runtime-profile.v1"}
    with pytest.raises(ValueError, match="fields drift"):
        validate_bounded_photographic_profile(payload)


def test_canonical_profile_bytes_reject_nonfinite() -> None:
    with pytest.raises(ValueError):
        canonical_profile_bytes({"value": float("nan")})
