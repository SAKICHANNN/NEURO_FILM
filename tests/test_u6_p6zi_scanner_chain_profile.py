from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.eval.scanner_glare_typed_chain import (
    glare_profile,
    load_contract,
    scanner_profile,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile
from src.film_physics.scanner_glare import compile_scanner_glare_kernel

ROOT = Path(__file__).resolve().parents[1]
P6ZG_CONTRACT = ROOT / "configs" / "u6_p6zg_scanner_glare_typed_chain_v1.json"
P6ZI_CONTRACT = ROOT / "configs" / "u6_p6zi_scanner_chain_profile_serialization_v1.json"
EXPECTED_PROFILE_SHA256 = (
    "3ebdee37366871460b1f5feaedbe0362c59e593efd3485652a892183bba6b891"
)


def _profile() -> ScannerChainProfile:
    contract = load_contract(P6ZG_CONTRACT)
    return ScannerChainProfile(
        scanner_profile=scanner_profile(ROOT, contract),
        glare_profile=glare_profile(),
    )


def test_canonical_roundtrip_reconstructs_exact_profiles_and_kernel() -> None:
    profile = _profile()
    raw = profile.canonical_bytes()
    reconstructed = ScannerChainProfile.from_json_bytes(raw)
    assert reconstructed == profile
    assert reconstructed.canonical_bytes() == raw
    assert reconstructed.profile_sha256 == profile.profile_sha256
    assert profile.profile_sha256 == EXPECTED_PROFILE_SHA256
    assert np.array_equal(
        compile_scanner_glare_kernel(
            reconstructed.glare_profile,
            kernel_size=reconstructed.glare_kernel_size,
        ),
        compile_scanner_glare_kernel(
            profile.glare_profile, kernel_size=profile.glare_kernel_size
        ),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_level", "calibrated-reference"),
        ("input_domain", "display-linear-light"),
        ("output_domain", "display-encoded-rgb"),
        ("arithmetic", "float32-standard"),
        ("stage_order", ["spectral", "dmax", "p6za-multiscale-glare", "mtf", "noise"]),
        ("glare_algorithm", "full-fft"),
        ("product_runtime_allowed", True),
        ("scanner_calibrated", True),
    ],
)
def test_fixed_claim_domain_and_execution_fields_fail_closed(
    field: str, value: object
) -> None:
    payload = _profile().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        ScannerChainProfile.from_dict(payload)


def test_unknown_nested_and_nonexact_types_fail_closed() -> None:
    payload = _profile().to_dict()
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        ScannerChainProfile.from_dict(payload)

    payload = _profile().to_dict()
    payload["scanner_profile"]["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        ScannerChainProfile.from_dict(payload)

    payload = _profile().to_dict()
    payload["glare_profile"]["components"][0]["weight"] = 1
    with pytest.raises(ValueError, match="JSON float"):
        ScannerChainProfile.from_dict(payload)


def test_duplicate_nonfinite_and_invalid_utf8_json_fail_closed() -> None:
    raw = _profile().canonical_bytes()
    duplicate = raw[:-1] + b',"schema":"neuro_film.generic_scanner_chain_profile.v1"}'
    with pytest.raises(ValueError, match="duplicate"):
        ScannerChainProfile.from_json_bytes(duplicate)

    payload = _profile().to_dict()
    payload["pixel_pitch_um"] = float("nan")
    nonfinite = json.dumps(payload, separators=(",", ":")).encode("ascii")
    with pytest.raises(ValueError, match="nonfinite"):
        ScannerChainProfile.from_json_bytes(nonfinite)

    with pytest.raises(ValueError, match="invalid scanner profile JSON"):
        ScannerChainProfile.from_json_bytes(b'{"schema":"\xff"}')


def test_parameter_mutation_changes_identity_and_does_not_mutate_source() -> None:
    profile = _profile()
    payload = profile.to_dict()
    changed = deepcopy(payload)
    changed["downstream_tile_rows"] = 256
    reconstructed = ScannerChainProfile.from_dict(changed)
    assert reconstructed.profile_sha256 != profile.profile_sha256
    assert payload == profile.to_dict()


def test_frozen_p6zh_parent_hashes_are_exact() -> None:
    contract = json.loads(P6ZI_CONTRACT.read_text(encoding="utf-8"))
    for prefix in ("p6zh_evidence", "p6zh_implementation"):
        path = ROOT / contract["parents"][f"{prefix}_path"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == contract["parents"][f"{prefix}_sha256"]
        )
