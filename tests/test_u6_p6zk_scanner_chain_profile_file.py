from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

import src.film_physics.scanner_chain_profile_file as profile_file
from src.eval.scanner_glare_typed_chain import (
    glare_profile,
    load_contract,
    scanner_profile,
)
from src.film_physics.scanner_chain_profile import ScannerChainProfile

ROOT = Path(__file__).resolve().parents[1]
P6ZG = ROOT / "configs/u6_p6zg_scanner_glare_typed_chain_v1.json"
PROFILE_SHA = "3ebdee37366871460b1f5feaedbe0362c59e593efd3485652a892183bba6b891"
FILE_SHA = "38ed5d2ea154eefdd3f1a264ff86db313f963e3f1ff556c57729fe7ce9c78350"


def _profile_bytes() -> bytes:
    contract = load_contract(P6ZG)
    profile = ScannerChainProfile(
        scanner_profile=scanner_profile(ROOT, contract),
        glare_profile=glare_profile(),
    )
    return profile.canonical_bytes()


def test_profile_file_bytes_and_strict_roundtrip_are_frozen() -> None:
    encoded = profile_file.encode_scanner_chain_profile_file(
        _profile_bytes(), expected_profile_sha256=PROFILE_SHA
    )
    assert len(encoded) == 1162
    assert hashlib.sha256(encoded).hexdigest() == FILE_SHA
    assert not encoded.endswith(b"\n")
    assert profile_file.decode_scanner_chain_profile_file(
        encoded,
        expected_file_sha256=FILE_SHA,
        expected_profile_sha256=PROFILE_SHA,
    ) == _profile_bytes()


def test_create_only_profile_file_preserves_foreign_destination(tmp_path: Path) -> None:
    destination = tmp_path / "profile.json"
    destination.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        profile_file.publish_scanner_chain_profile_file_create_only(
            destination, _profile_bytes(), expected_profile_sha256=PROFILE_SHA
        )
    assert destination.read_bytes() == b"foreign"
    assert list(tmp_path.iterdir()) == [destination]


def test_profile_file_loader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(
        profile_file.encode_scanner_chain_profile_file(
            _profile_bytes(), expected_profile_sha256=PROFILE_SHA
        )
    )
    link = tmp_path / "link.json"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"host cannot create a test symlink: {exc}")
    with pytest.raises(ValueError, match="non-symlink"):
        profile_file.load_scanner_chain_profile_file(
            link,
            expected_file_sha256=FILE_SHA,
            expected_profile_sha256=PROFILE_SHA,
        )


@pytest.mark.parametrize("case", ["wrong-file", "wrong-profile", "newline", "oversize"])
def test_invalid_profile_file_rejects_before_bundle_execution(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = profile_file.encode_scanner_chain_profile_file(
        _profile_bytes(), expected_profile_sha256=PROFILE_SHA
    )
    expected_file = FILE_SHA
    expected_profile = PROFILE_SHA
    if case == "wrong-file":
        expected_file = "0" * 64
    elif case == "wrong-profile":
        expected_profile = "0" * 64
    elif case == "newline":
        encoded += b"\n"
        expected_file = hashlib.sha256(encoded).hexdigest()
    else:
        encoded += b" " * 4096
        expected_file = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "profile.json"
    path.write_bytes(encoded)
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        raise AssertionError("bundle executor was called")

    monkeypatch.setattr(profile_file, "apply_scanner_chain_from_bundle", forbidden)
    source = np.full((3, 5, 3), 0.5, dtype=np.float64)
    with pytest.raises(ValueError):
        profile_file.apply_scanner_chain_from_profile_file(
            source,
            path,
            expected_file_sha256=expected_file,
            expected_profile_sha256=expected_profile,
        )
    assert calls == 0
