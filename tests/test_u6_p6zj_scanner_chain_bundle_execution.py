from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.eval.scanner_chain_bundle_execution as execution
from src.eval.scanner_glare_tiled_downstream import (
    apply_typed_scanner_glare_chain_tiled_downstream,
)
from src.eval.scanner_glare_typed_chain import (
    glare_profile,
    load_contract,
    scanner_profile,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile

ROOT = Path(__file__).resolve().parents[1]
P6ZG_CONTRACT = ROOT / "configs" / "u6_p6zg_scanner_glare_typed_chain_v1.json"


def _profile() -> ScannerChainProfile:
    contract = load_contract(P6ZG_CONTRACT)
    return ScannerChainProfile(
        scanner_profile=scanner_profile(ROOT, contract),
        glare_profile=glare_profile(),
    )


def _source() -> np.ndarray:
    return np.random.default_rng(6206106).uniform(0.03, 0.97, size=(65, 67, 3))


def test_bundle_execution_is_bit_exact_with_direct_execution() -> None:
    profile = _profile()
    source = _source()
    direct = apply_typed_scanner_glare_chain_tiled_downstream(
        source,
        profile.scanner_profile,
        pixel_pitch_um=profile.pixel_pitch_um,
        glare_row_chunk=profile.glare_row_chunk,
        downstream_tile_rows=profile.downstream_tile_rows,
    )
    first = execution.apply_scanner_chain_from_bundle(
        source,
        profile.canonical_bytes(),
        expected_profile_sha256=profile.profile_sha256,
    )
    second = execution.apply_scanner_chain_from_bundle(
        source,
        profile.canonical_bytes(),
        expected_profile_sha256=profile.profile_sha256,
    )
    assert np.array_equal(first, direct)
    assert np.array_equal(second, direct)


@pytest.mark.parametrize("case", ["identity", "noncanonical", "tampered"])
def test_bundle_failures_happen_before_pixel_execution(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    raw = profile.canonical_bytes()
    expected = profile.profile_sha256
    if case == "identity":
        expected = "0" * 64
    elif case == "noncanonical":
        raw += b"\n"
    else:
        raw = raw.replace(b'"downstream_tile_rows":512', b'"downstream_tile_rows":256')

    calls = 0

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("pixel executor was called")

    monkeypatch.setattr(
        execution, "apply_typed_scanner_glare_chain_tiled_downstream", forbidden
    )
    with pytest.raises(ValueError):
        execution.apply_scanner_chain_from_bundle(
            _source(), raw, expected_profile_sha256=expected
        )
    assert calls == 0


@pytest.mark.parametrize(
    "source",
    [
        np.full((7, 9, 3), 0.5, dtype=np.float32),
        np.full((7, 9), 0.5, dtype=np.float64),
        np.full((7, 9, 3), 0.0, dtype=np.float64),
        np.full((7, 9, 3), np.nan, dtype=np.float64),
    ],
)
def test_input_failures_happen_before_pixel_execution(
    source: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("pixel executor was called")

    monkeypatch.setattr(
        execution, "apply_typed_scanner_glare_chain_tiled_downstream", forbidden
    )
    with pytest.raises((TypeError, ValueError)):
        execution.apply_scanner_chain_from_bundle(
            source,
            profile.canonical_bytes(),
            expected_profile_sha256=profile.profile_sha256,
        )
    assert calls == 0
