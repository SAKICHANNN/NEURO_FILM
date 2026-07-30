from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.build_srgb_eotf_lut_c_v1 import (
    encode_header,
    encode_source,
    tables,
)
from scripts.build_srgb_eotf_native_v1 import (
    build_android,
    build_apple_objects,
    build_llvm_mingw_dll,
    build_msvc_dll,
)
from scripts.build_reference_chain_msvc import VSWHERE


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
TOOL_ROOT = ROOT.parent / "\u8ffd\u8272" / "outputs" / "tmp" / "tools"
LLVM = TOOL_ROOT / "llvm-mingw-20260616-ucrt-x86_64"
NDK = TOOL_ROOT / "android-ndk-r27d"


def _load(path: Path):
    dll = ctypes.CDLL(str(path))
    apply = dll.nf_srgb_eotf_f32_apply_v1
    apply.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    apply.restype = ctypes.c_int
    identity = dll.nf_srgb_eotf_f32_lut_sha256_v1
    identity.argtypes = []
    identity.restype = ctypes.c_char_p
    return apply, identity


def _assert_exact_dll(path: Path) -> None:
    table8, table16, digest = tables()
    apply, identity = _load(path)
    assert identity().decode("ascii") == digest
    for bit_depth, source, expected in (
        (8, np.arange(256, dtype=np.uint8), table8),
        (16, np.arange(65536, dtype=np.uint16), table16),
    ):
        output = np.empty(source.size, dtype=np.float32)
        assert apply(
            source.ctypes.data,
            source.size,
            bit_depth,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.size,
        ) == 1
        np.testing.assert_array_equal(output, expected)
    sentinel = np.full(16, np.float32(-7.0), dtype=np.float32)
    before = sentinel.tobytes()
    source = np.arange(16, dtype=np.uint8)
    for source_pointer, count, depth, capacity in (
        (source.ctypes.data, 16, 7, 16),
        (source.ctypes.data, 16, 8, 15),
        (source.ctypes.data, 0, 8, 16),
        (0, 16, 8, 16),
        (
            source.ctypes.data,
            ctypes.c_size_t(-1).value,
            16,
            ctypes.c_size_t(-1).value,
        ),
    ):
        assert apply(
            source_pointer,
            count,
            depth,
            sentinel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            capacity,
        ) == 0
        assert sentinel.tobytes() == before
    assert apply(
        source.ctypes.data,
        16,
        8,
        ctypes.POINTER(ctypes.c_float)(),
        16,
    ) == 0
    overlap = (ctypes.c_uint8 * 64)(*range(64))
    assert apply(
        ctypes.addressof(overlap),
        16,
        8,
        ctypes.cast(overlap, ctypes.POINTER(ctypes.c_float)),
        16,
    ) == 0
    assert bytes(overlap) == bytes(range(64))
    unaligned_output = (ctypes.c_uint8 * 65)(*range(65))
    unaligned_output_before = bytes(unaligned_output)
    assert apply(
        source.ctypes.data,
        16,
        8,
        ctypes.cast(
            ctypes.addressof(unaligned_output) + 1,
            ctypes.POINTER(ctypes.c_float),
        ),
        16,
    ) == 0
    assert bytes(unaligned_output) == unaligned_output_before
    unaligned_u16 = (ctypes.c_uint8 * 33)(*range(33))
    assert apply(
        ctypes.addressof(unaligned_u16) + 1,
        16,
        16,
        sentinel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        16,
    ) == 0
    assert sentinel.tobytes() == before


def test_generated_lut_source_and_identity_are_exact() -> None:
    source = encode_source()
    header = encode_header()
    assert (
        NATIVE / "reference_srgb_eotf_f32_v1.c"
    ).read_text(encoding="utf-8") == source
    assert (
        NATIVE / "reference_srgb_eotf_f32_v1.h"
    ).read_text(encoding="utf-8") == header
    table8, table16, digest = tables()
    payload = (
        table8.astype(">f4", copy=False).tobytes()
        + table16.astype(">f4", copy=False).tobytes()
    )
    assert hashlib.sha256(payload).hexdigest() == digest
    assert digest in header


def test_msvc_dll_exhaustively_matches_python(tmp_path: Path) -> None:
    if not VSWHERE.is_file():
        pytest.skip("MSVC discovery is unavailable")
    first = tmp_path / "msvc" / "eotf.dll"
    second = tmp_path / "msvc-repeat" / "eotf.dll"
    report = build_msvc_dll(first)
    repeated = build_msvc_dll(second)
    assert report["dll_sha256"] == repeated["dll_sha256"]
    _assert_exact_dll(first)


def test_llvm_dll_exhaustively_matches_python(tmp_path: Path) -> None:
    if not LLVM.is_dir():
        pytest.skip("LLVM-MinGW is unavailable")
    first = tmp_path / "llvm" / "eotf.dll"
    second = tmp_path / "llvm-repeat" / "eotf.dll"
    report = build_llvm_mingw_dll(LLVM, first)
    repeated = build_llvm_mingw_dll(LLVM, second)
    assert report["dll_sha256"] == repeated["dll_sha256"]
    _assert_exact_dll(first)


def test_android_links_two_abis_without_runtime(tmp_path: Path) -> None:
    if not NDK.is_dir():
        pytest.skip("Android NDK is unavailable")
    report = build_android(NDK, tmp_path)
    repeated = build_android(NDK, tmp_path / "repeat")
    assert report["claim_scope"].endswith("no device execution")
    assert {
        key: value["sha256"]
        for key, value in report["artifacts"].items()
    } == {
        key: value["sha256"]
        for key, value in repeated["artifacts"].items()
    }


def test_apple_builds_objects_without_runtime(tmp_path: Path) -> None:
    if not LLVM.is_dir():
        pytest.skip("LLVM distribution is unavailable")
    report = build_apple_objects(LLVM, tmp_path)
    repeated = build_apple_objects(LLVM, tmp_path / "repeat")
    assert report["claim_scope"].endswith("not linked or run")
    assert {
        key: value["sha256"]
        for key, value in report["artifacts"].items()
    } == {
        key: value["sha256"]
        for key, value in repeated["artifacts"].items()
    }
