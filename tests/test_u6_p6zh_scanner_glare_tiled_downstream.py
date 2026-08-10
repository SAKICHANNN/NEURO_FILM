from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.scanner_glare_tiled_downstream import (
    apply_typed_scanner_glare_chain_tiled_downstream,
)
from src.eval.scanner_glare_typed_chain import (
    apply_typed_scanner_glare_chain,
    load_contract,
    scanner_profile,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p6zg_scanner_glare_typed_chain_v1.json"


def test_tiled_downstream_matches_full_chain_without_seams() -> None:
    contract = load_contract(CONTRACT)
    profile = scanner_profile(ROOT, contract)
    rng = np.random.default_rng(6206106)
    source = rng.uniform(0.03, 0.97, size=(321, 513, 3))
    reference = apply_typed_scanner_glare_chain(
        source,
        profile,
        pixel_pitch_um=1.0,
        glare_algorithm="channel-serial-block-fft",
        glare_row_chunk=512,
    )
    candidate = apply_typed_scanner_glare_chain_tiled_downstream(
        source,
        profile,
        pixel_pitch_um=1.0,
        glare_row_chunk=512,
        downstream_tile_rows=127,
    )
    assert np.array_equal(candidate, reference)
