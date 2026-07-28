from __future__ import annotations

import hashlib
import ctypes
from pathlib import Path
import subprocess

import pytest

from scripts.build_srgb_icc_profile_c_abi_v1 import (
    encode_header,
    encode_source,
)
from scripts.build_srgb_icc_profile_native_v1 import (
    build_android,
    build_apple_objects,
    build_llvm_mingw,
    build_llvm_mingw_dll,
    build_msvc,
    build_msvc_dll,
)
from scripts.build_reference_chain_msvc import VSWHERE
from src.color_match.srgb_icc_profile import (
    SRGB_ICC_PROFILE_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
TOOL_ROOT = (
    ROOT.parent
    / "\u8ffd\u8272"
    / "outputs"
    / "tmp"
    / "tools"
)
LLVM = TOOL_ROOT / "llvm-mingw-20260616-ucrt-x86_64"
NDK = TOOL_ROOT / "android-ndk-r27d"


def _assert_runtime(executable: Path) -> None:
    expected = {
        "hash": SRGB_ICC_PROFILE_SHA256,
        "header": "588 6d6e7472 52474220 58595a20 61637370",
        "negative": "pass",
    }
    for command, value in expected.items():
        completed = subprocess.run(
            [executable, command],
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == value


def _assert_dynamic_abi(library: Path) -> None:
    dll = ctypes.CDLL(str(library))
    size = dll.nf_srgb_icc_profile_size_v1
    size.argtypes = []
    size.restype = ctypes.c_size_t
    profile_hash = dll.nf_srgb_icc_profile_sha256_v1
    profile_hash.argtypes = []
    profile_hash.restype = ctypes.c_char_p
    copy = dll.nf_srgb_icc_profile_copy_v1
    copy.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    copy.restype = ctypes.c_int

    assert size() == 588
    assert profile_hash().decode("ascii") == SRGB_ICC_PROFILE_SHA256
    buffer_type = ctypes.c_uint8 * 588
    short = buffer_type(*([0xA5] * 588))
    before = bytes(short)
    assert copy(short, 587) == 0
    assert bytes(short) == before
    assert copy(None, 588) == 0
    output = buffer_type()
    assert copy(output, 588) == 1
    assert hashlib.sha256(bytes(output)).hexdigest() == (
        SRGB_ICC_PROFILE_SHA256
    )


def test_generated_c_and_header_are_byte_exact() -> None:
    assert (
        NATIVE / "reference_srgb_icc_profile_v1.h"
    ).read_text(encoding="utf-8") == encode_header()
    assert (
        NATIVE / "reference_srgb_icc_profile_v1.c"
    ).read_text(encoding="utf-8") == encode_source()
    assert hashlib.sha256(
        encode_source().encode("utf-8")
    ).hexdigest() == hashlib.sha256(
        (NATIVE / "reference_srgb_icc_profile_v1.c").read_bytes()
    ).hexdigest()


def test_msvc_runtime_exact_profile_and_failure_boundary(
    tmp_path: Path,
) -> None:
    if not VSWHERE.is_file():
        pytest.skip("MSVC discovery is unavailable")
    executable = tmp_path / "srgb_icc_msvc.exe"
    try:
        report = build_msvc(executable)
        repeated = build_msvc(
            tmp_path / "repeat" / "srgb_icc_msvc.exe"
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"MSVC is unavailable: {exc}")
    assert report["claim_scope"] == "Windows x86_64 host compile and execution"
    assert report["executable_sha256"] == repeated["executable_sha256"]
    _assert_runtime(executable)


def test_llvm_mingw_runtime_exact_profile_and_failure_boundary(
    tmp_path: Path,
) -> None:
    if not LLVM.is_dir():
        pytest.skip("pinned LLVM-MinGW is unavailable")
    executable = tmp_path / "srgb_icc_llvm.exe"
    report = build_llvm_mingw(LLVM, executable)
    repeated = build_llvm_mingw(
        LLVM,
        tmp_path / "repeat" / "srgb_icc_llvm.exe",
    )
    assert report["llvm_version"] == "22.1.8"
    assert report["executable_sha256"] == repeated["executable_sha256"]
    _assert_runtime(executable)


def test_msvc_dynamic_abi_executes_exactly(tmp_path: Path) -> None:
    if not VSWHERE.is_file():
        pytest.skip("MSVC discovery is unavailable")
    library = tmp_path / "msvc" / "srgb_icc.dll"
    repeated_library = tmp_path / "msvc-repeat" / "srgb_icc.dll"
    report = build_msvc_dll(library)
    repeated = build_msvc_dll(repeated_library)
    assert report["dll_sha256"] == repeated["dll_sha256"]
    _assert_dynamic_abi(library)


def test_llvm_mingw_dynamic_abi_executes_exactly(
    tmp_path: Path,
) -> None:
    if not LLVM.is_dir():
        pytest.skip("pinned LLVM-MinGW is unavailable")
    library = tmp_path / "llvm" / "srgb_icc.dll"
    repeated_library = tmp_path / "llvm-repeat" / "srgb_icc.dll"
    report = build_llvm_mingw_dll(LLVM, library)
    repeated = build_llvm_mingw_dll(LLVM, repeated_library)
    assert report["dll_sha256"] == repeated["dll_sha256"]
    _assert_dynamic_abi(library)


def test_android_two_abi_link_is_not_runtime(tmp_path: Path) -> None:
    if not NDK.is_dir():
        pytest.skip("pinned Android NDK is unavailable")
    report = build_android(NDK, tmp_path)
    repeated = build_android(NDK, tmp_path / "repeat")
    assert report["claim_scope"] == (
        "Android cross-compile/link only; no device execution"
    )
    assert set(report["artifacts"]) == {"arm64-v8a", "x86_64"}
    assert all(
        artifact["exported_symbols"]
        == [
            "nf_srgb_icc_profile_copy_v1",
            "nf_srgb_icc_profile_sha256_v1",
            "nf_srgb_icc_profile_size_v1",
        ]
        for artifact in report["artifacts"].values()
    )
    assert {
        name: value["sha256"]
        for name, value in report["artifacts"].items()
    } == {
        name: value["sha256"]
        for name, value in repeated["artifacts"].items()
    }


def test_apple_arm64_objects_are_not_runtime(tmp_path: Path) -> None:
    if not LLVM.is_dir():
        pytest.skip("pinned LLVM distribution is unavailable")
    report = build_apple_objects(LLVM, tmp_path)
    repeated = build_apple_objects(LLVM, tmp_path / "repeat")
    assert report["claim_scope"] == (
        "Apple ARM64 object-only; not linked or run"
    )
    assert set(report["artifacts"]) == {"macos-arm64", "ios-arm64"}
    assert all(
        artifact["defined_symbols"]
        == [
            "nf_srgb_icc_profile_copy_v1",
            "nf_srgb_icc_profile_sha256_v1",
            "nf_srgb_icc_profile_size_v1",
        ]
        for artifact in report["artifacts"].values()
    )
    assert {
        name: value["sha256"]
        for name, value in report["artifacts"].items()
    } == {
        name: value["sha256"]
        for name, value in repeated["artifacts"].items()
    }
