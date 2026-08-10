from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.scanner_glare_typed_chain import (
    apply_typed_scanner_glare_chain,
    load_contract,
    scanner_profile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6zg_scanner_glare_typed_chain_v1.json"


def test_contract_and_typed_chain_match_reference() -> None:
    contract = load_contract(CONTRACT)
    profile = scanner_profile(ROOT, contract)
    rng = np.random.default_rng(6206106)
    source = rng.uniform(0.03, 0.97, size=(321, 513, 3))
    reference = apply_typed_scanner_glare_chain(
        source,
        profile,
        pixel_pitch_um=1.0,
        glare_algorithm="full-fft",
        glare_row_chunk=512,
    )
    candidate = apply_typed_scanner_glare_chain(
        source,
        profile,
        pixel_pitch_um=1.0,
        glare_algorithm="channel-serial-block-fft",
        glare_row_chunk=512,
    )
    difference = candidate - reference
    assert float(np.max(np.abs(difference))) <= 1e-11
    assert float(np.sqrt(np.mean(np.square(difference)))) <= 1e-12
    assert np.all(candidate >= 0.0)
    assert np.all(candidate <= 1.0)


def test_chain_rejects_unknown_glare_algorithm() -> None:
    contract = load_contract(CONTRACT)
    profile = scanner_profile(ROOT, contract)
    with pytest.raises(ValueError, match="unsupported"):
        apply_typed_scanner_glare_chain(
            np.full((7, 9, 3), 0.5),
            profile,
            pixel_pitch_um=1.0,
            glare_algorithm="unknown",
            glare_row_chunk=3,
        )
