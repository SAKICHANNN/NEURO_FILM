"""ctypes contract for the native streaming AO6 source-context reducer."""

from __future__ import annotations

import ctypes


class NativeAo6ContextStateF32V1(ctypes.Structure):
    _Vector = ctypes.c_double * 3
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("pixel_count", ctypes.c_uint64),
        ("mean", _Vector),
        ("m2", _Vector),
    ]


__all__ = ["NativeAo6ContextStateF32V1"]
