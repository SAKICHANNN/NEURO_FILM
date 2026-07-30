from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.build_reference_chain_msvc import VSWHERE
from scripts.build_srgb_oetf_quantize_c_v1 import (
    accepted_linear_bounds,
    encode_header,
    encode_source,
    quantize_reference,
    thresholds,
)
from scripts.build_srgb_oetf_quantize_native_v1 import (
    build_android,
    build_apple_objects,
    build_llvm_mingw_dll,
    build_msvc_dll,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
TOOL_ROOT = ROOT.parent / "\u8ffd\u8272" / "outputs" / "tmp" / "tools"
LLVM = TOOL_ROOT / "llvm-mingw-20260616-ucrt-x86_64"
NDK = TOOL_ROOT / "android-ndk-r27d"


def _load(path: Path):
    dll = ctypes.CDLL(str(path))
    apply = dll.nf_srgb_oetf_quantize_apply_v1
    apply.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    apply.restype = ctypes.c_int
    identity = dll.nf_srgb_oetf_quantize_thresholds_sha256_v1
    identity.argtypes = []
    identity.restype = ctypes.c_char_p
    return dll, apply, identity


def _close(dll: ctypes.CDLL) -> None:
    handle = dll._handle
    if handle:
        free_library = ctypes.windll.kernel32.FreeLibrary
        free_library.argtypes = [ctypes.c_void_p]
        free_library.restype = ctypes.c_int
        assert free_library(ctypes.c_void_p(handle))
        dll._handle = 0


def _probe_values(bit_depth: int) -> np.ndarray:
    threshold = thresholds()[0 if bit_depth == 8 else 1]
    neighbours = np.unique(
        np.concatenate(
            [
                threshold,
                threshold - np.uint32(1),
                np.minimum(
                    threshold + np.uint32(1),
                    np.float32(1.0).view(np.uint32),
                ),
            ]
        )
    ).view(np.float32)
    rng = np.random.default_rng(20260728 + bit_depth)
    random_bits = rng.integers(
        0,
        int(np.float32(1.0).view(np.uint32)) + 1,
        size=1_000_000,
        dtype=np.uint32,
    )
    lower, upper = accepted_linear_bounds()
    special = np.asarray(
        [
            lower,
            -0.0,
            0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            0.0031308,
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            1.0,
            upper,
        ],
        dtype=np.float32,
    )
    return np.concatenate([special, neighbours, random_bits.view(np.float32)])


def _assert_exact_dll(path: Path) -> None:
    _, _, digest = thresholds()
    dll, apply, identity = _load(path)
    try:
        assert identity().decode("ascii") == digest
        for bit_depth in (8, 16):
            source = _probe_values(bit_depth)
            expected = quantize_reference(source, bit_depth)
            output = np.empty(source.size, dtype=expected.dtype)
            assert (
                apply(
                    source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    source.size,
                    bit_depth,
                    output.ctypes.data,
                    output.size,
                )
                == 1
            )
            np.testing.assert_array_equal(output, expected)

        source = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        lower, upper = accepted_linear_bounds()
        sentinel = np.full(16, 0xA5, dtype=np.uint8)
        before = sentinel.tobytes()
        invalid_sources = (
            np.concatenate([source[:-1], np.asarray([np.nan], np.float32)]),
            np.concatenate([source[:-1], np.asarray([np.inf], np.float32)]),
            np.concatenate(
                [
                    source[:-1],
                    np.asarray(
                        [np.nextafter(lower, np.float32(-np.inf))],
                        np.float32,
                    ),
                ]
            ),
            np.concatenate(
                [
                    source[:-1],
                    np.asarray(
                        [np.nextafter(upper, np.float32(np.inf))],
                        np.float32,
                    ),
                ]
            ),
        )
        for invalid in invalid_sources:
            assert (
                apply(
                    invalid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    invalid.size,
                    8,
                    sentinel.ctypes.data,
                    sentinel.size,
                )
                == 0
            )
            assert sentinel.tobytes() == before
        for pointer, count, depth, capacity in (
            (
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                source.size,
                7,
                sentinel.size,
            ),
            (
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                source.size,
                8,
                sentinel.size - 1,
            ),
            (
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                0,
                8,
                sentinel.size,
            ),
            (
                ctypes.POINTER(ctypes.c_float)(),
                source.size,
                8,
                sentinel.size,
            ),
            (
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_size_t(-1).value,
                16,
                ctypes.c_size_t(-1).value,
            ),
        ):
            assert (
                apply(
                    pointer,
                    count,
                    depth,
                    sentinel.ctypes.data,
                    capacity,
                )
                == 0
            )
            assert sentinel.tobytes() == before
        assert (
            apply(
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                source.size,
                8,
                0,
                source.size,
            )
            == 0
        )
        overlap_before = source.tobytes()
        assert (
            apply(
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                source.size,
                8,
                source.ctypes.data,
                source.size,
            )
            == 0
        )
        assert source.tobytes() == overlap_before
        unaligned_source = (ctypes.c_uint8 * 65)(*range(65))
        assert (
            apply(
                ctypes.cast(
                    ctypes.addressof(unaligned_source) + 1,
                    ctypes.POINTER(ctypes.c_float),
                ),
                16,
                8,
                sentinel.ctypes.data,
                sentinel.size,
            )
            == 0
        )
        assert sentinel.tobytes() == before
        output_bytes = (ctypes.c_uint8 * 33)(*range(33))
        output_before = bytes(output_bytes)
        assert (
            apply(
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                source.size,
                16,
                ctypes.addressof(output_bytes) + 1,
                source.size,
            )
            == 0
        )
        assert bytes(output_bytes) == output_before
    finally:
        _close(dll)


def test_generated_threshold_source_and_identity_are_exact() -> None:
    assert (
        NATIVE / "reference_srgb_oetf_quantize_v1.c"
    ).read_text(encoding="utf-8") == encode_source()
    header = (
        NATIVE / "reference_srgb_oetf_quantize_v1.h"
    ).read_text(encoding="utf-8")
    assert header == encode_header()
    threshold8, threshold16, digest = thresholds()
    payload = (
        b"neuro-film.srgb-oetf-quantize-thresholds.v1\0"
        + threshold8.astype(">u4", copy=False).tobytes()
        + threshold16.astype(">u4", copy=False).tobytes()
    )
    assert hashlib.sha256(payload).hexdigest() == digest
    assert digest in header


def test_thresholds_cover_every_quantization_boundary() -> None:
    for bit_depth, threshold in (
        (8, thresholds()[0]),
        (16, thresholds()[1]),
    ):
        targets = np.arange(1, threshold.size + 1, dtype=np.uint32)
        assert np.array_equal(
            quantize_reference(threshold.view(np.float32), bit_depth).astype(
                np.uint32
            ),
            targets,
        )
        assert np.array_equal(
            quantize_reference(
                (threshold - np.uint32(1)).view(np.float32),
                bit_depth,
            ).astype(np.uint32),
            targets - np.uint32(1),
        )


def test_msvc_dll_matches_all_boundaries_and_random_inputs(
    tmp_path: Path,
) -> None:
    if not VSWHERE.is_file():
        pytest.skip("MSVC discovery is unavailable")
    first = tmp_path / "msvc" / "oetf.dll"
    second = tmp_path / "msvc-repeat" / "oetf.dll"
    report = build_msvc_dll(first)
    repeated = build_msvc_dll(second)
    assert report["dll_sha256"] == repeated["dll_sha256"]
    _assert_exact_dll(first)


def test_llvm_dll_matches_all_boundaries_and_random_inputs(
    tmp_path: Path,
) -> None:
    if not LLVM.is_dir():
        pytest.skip("LLVM-MinGW is unavailable")
    first = tmp_path / "llvm" / "oetf.dll"
    second = tmp_path / "llvm-repeat" / "oetf.dll"
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
