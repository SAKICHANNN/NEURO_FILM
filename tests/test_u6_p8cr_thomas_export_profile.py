from __future__ import annotations

import copy
import ctypes
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _parent_payloads,
    _profiles,
)
from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import _gauge_payload
from src.eval.native_thomas_field_conformance import canonical_bytes
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
)
from src.film_physics.native_thomas_export_profile import (
    canonical_profile_bytes,
    compile_native_thomas_export_profile,
    load_native_thomas_export_profile,
    reconstruct_native_thomas_export_profile,
    validate_native_thomas_export_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    p4bw, prior = _parent_payloads()
    amplitude = compile_native_granularity_amplitude_profile(p4bw, prior)
    contract = json.loads(
        (ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return compile_native_thomas_export_profile(
        amplitude,
        _profiles(contract),
        _gauge_payload(),
        source_bindings={"test": "a" * 64},
    )


def test_profile_roundtrip_preserves_all_abi_struct_bytes() -> None:
    payload = _payload()
    validate_native_thomas_export_profile(payload)
    encoded = canonical_bytes(payload)
    rebuilt = json.loads(encoded)
    amplitude, fields, gauge = reconstruct_native_thomas_export_profile(rebuilt)
    original_amplitude, original_fields, original_gauge = (
        reconstruct_native_thomas_export_profile(payload)
    )
    assert ctypes.string_at(ctypes.byref(amplitude), ctypes.sizeof(amplitude)) == (
        ctypes.string_at(
            ctypes.byref(original_amplitude), ctypes.sizeof(original_amplitude)
        )
    )
    assert [bytes(row) for row in fields] == [bytes(row) for row in original_fields]
    assert ctypes.string_at(
        ctypes.byref(gauge), ctypes.sizeof(gauge)
    ) == ctypes.string_at(ctypes.byref(original_gauge), ctypes.sizeof(original_gauge))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["fields"][0].__setitem__("mean_offspring", 0.0),
            "positive",
        ),
        (
            lambda value: value["amplitude"]["channels"][0][
                "log_exposure_knots"
            ].reverse(),
            "increasing",
        ),
        (
            lambda value: value["source_bindings"].__setitem__("test", "z" * 64),
            "SHA-256",
        ),
        (
            lambda value: value.__setitem__("profile_sha256", "0" * 64),
            "identity drift",
        ),
    ],
)
def test_profile_rejects_invalid_or_tampered_payloads(mutator, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        validate_native_thomas_export_profile(payload)


def test_profile_file_loader_requires_canonical_expected_identity(
    tmp_path: Path,
) -> None:
    payload = _payload()
    path = tmp_path / "profile.json"
    path.write_bytes(canonical_profile_bytes(payload))
    assert (
        load_native_thomas_export_profile(
            path, expected_profile_sha256=payload["profile_sha256"]
        )
        == payload
    )
    with pytest.raises(ValueError, match="expected profile"):
        load_native_thomas_export_profile(
            path, expected_profile_sha256="0" * 64
        )
    path.write_bytes(canonical_profile_bytes(payload) + b"\n")
    with pytest.raises(ValueError, match="not canonical"):
        load_native_thomas_export_profile(
            path, expected_profile_sha256=payload["profile_sha256"]
        )


def test_tracked_generic_thomas_profile_asset_is_exact() -> None:
    path = ROOT / "configs/render_profiles/generic_physical_thomas_p8cr_v1.json"
    payload = load_native_thomas_export_profile(
        path,
        expected_profile_sha256=(
            "923a985aa90e029332c723a33afb793afe05cae777b3c07061d54828b91e21b4"
        ),
    )
    assert payload["claim_level"] == "generic-physical-inspired"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "ce11da8e336b322342941ec639e4dac0fe3521224cf0fc907e9841cac51cf2c0"
    )
