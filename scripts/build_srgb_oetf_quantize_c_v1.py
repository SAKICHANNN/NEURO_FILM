"""Generate an exact float32-linear to integer-sRGB quantizer C11 ABI."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.color_match.staging_io import display_srgb


_FLOAT_ONE_BITS = np.float32(1.0).view(np.uint32)
_IDENTITY_PREFIX = b"neuro-film.srgb-oetf-quantize-thresholds.v1\0"


def accepted_linear_bounds() -> tuple[np.float32, np.float32]:
    """Return the exact float32 endpoints accepted by display_srgb."""
    lower = np.float32(-2e-6)
    if float(lower) < -2e-6:
        lower = np.nextafter(lower, np.float32(np.inf))
    upper = np.float32(1.0 + 2e-6)
    if float(upper) > 1.0 + 2e-6:
        upper = np.nextafter(upper, np.float32(-np.inf))
    return lower, upper


def quantize_reference(values: np.ndarray, bit_depth: int) -> np.ndarray:
    """Apply the exact current staging OETF and integer quantization."""
    if bit_depth not in {8, 16}:
        raise ValueError("bit_depth must be 8 or 16")
    linear = np.asarray(values, dtype=np.float32)
    encoded, _ = display_srgb(linear, label="OETF reference")
    maximum = np.float32(255 if bit_depth == 8 else 65535)
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    return np.rint(encoded * maximum).astype(dtype)


def _threshold_bits(bit_depth: int) -> np.ndarray:
    """Return the first positive float32 bit pattern for every output code."""
    maximum = 255 if bit_depth == 8 else 65535
    targets = np.arange(1, maximum + 1, dtype=np.uint32)
    lower = np.zeros(targets.shape, dtype=np.uint32)
    upper = np.full(targets.shape, _FLOAT_ONE_BITS, dtype=np.uint32)
    while bool(np.any(upper - lower > 1)):
        middle = lower + (upper - lower) // 2
        mapped = quantize_reference(
            middle.view(np.float32),
            bit_depth,
        ).astype(np.uint32)
        at_or_above = mapped >= targets
        upper = np.where(at_or_above, middle, upper)
        lower = np.where(at_or_above, lower, middle)
    if not np.all(
        quantize_reference(upper.view(np.float32), bit_depth).astype(
            np.uint32
        )
        >= targets
    ):
        raise AssertionError("upper threshold proof failed")
    if not np.all(
        quantize_reference(lower.view(np.float32), bit_depth).astype(
            np.uint32
        )
        < targets
    ):
        raise AssertionError("lower threshold proof failed")
    return upper


def thresholds() -> tuple[np.ndarray, np.ndarray, str]:
    threshold8 = _threshold_bits(8)
    threshold16 = _threshold_bits(16)
    payload = (
        _IDENTITY_PREFIX
        + threshold8.astype(">u4", copy=False).tobytes()
        + threshold16.astype(">u4", copy=False).tobytes()
    )
    return threshold8, threshold16, hashlib.sha256(payload).hexdigest()


def encode_header() -> str:
    _, _, digest = thresholds()
    lower, upper = accepted_linear_bounds()
    lower_hex = float(lower).hex()
    upper_hex = float(upper).hex()
    return f"""#ifndef NF_SRGB_OETF_QUANTIZE_V1_H
#define NF_SRGB_OETF_QUANTIZE_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define NF_SRGB_OETF_QUANTIZE_THRESHOLDS_V1_SHA256 "{digest}"
#define NF_SRGB_OETF_QUANTIZE_LINEAR_MIN_V1 {lower_hex}f
#define NF_SRGB_OETF_QUANTIZE_LINEAR_MAX_V1 {upper_hex}f

const char *nf_srgb_oetf_quantize_thresholds_sha256_v1(void);
int nf_srgb_oetf_quantize_apply_v1(
    const float *linear_samples,
    size_t sample_count,
    uint32_t bit_depth,
    void *output,
    size_t output_capacity);

#ifdef __cplusplus
}}
#endif

#endif
"""


def _encode_bits(name: str, values: np.ndarray) -> str:
    rows = []
    for offset in range(0, values.size, 8):
        encoded = ", ".join(
            f"0x{int(value):08x}u" for value in values[offset : offset + 8]
        )
        rows.append(f"    {encoded},")
    return (
        f"static const uint32_t {name}[{values.size}u] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def encode_source() -> str:
    threshold8, threshold16, digest = thresholds()
    return f"""#include "reference_srgb_oetf_quantize_v1.h"

#include <float.h>
#include <stdint.h>

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "nf_srgb_oetf_quantize_v1 requires IEEE-754 binary32 float"
#endif

{_encode_bits("NF_OETF_Q8", threshold8)}
{_encode_bits("NF_OETF_Q16", threshold16)}
static int nf_binary32_is_little_endian(void) {{
    const float one = 1.0f;
    const unsigned char *bytes = (const unsigned char *)&one;
    return (
        sizeof(float) == 4u
        && bytes[0] == 0x00u
        && bytes[1] == 0x00u
        && bytes[2] == 0x80u
        && bytes[3] == 0x3fu
    );
}}

static uint32_t nf_load_binary32_le_bits(const float *value) {{
    const unsigned char *bytes = (const unsigned char *)value;
    return (
        (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8u)
        | ((uint32_t)bytes[2] << 16u)
        | ((uint32_t)bytes[3] << 24u)
    );
}}

static uint32_t nf_quantize_bits(
    uint32_t bits,
    const uint32_t *thresholds,
    uint32_t count) {{
    uint32_t lower = 0u;
    uint32_t upper = count;
    while (lower < upper) {{
        const uint32_t middle = lower + (upper - lower) / 2u;
        if (bits < thresholds[middle]) {{
            upper = middle;
        }} else {{
            lower = middle + 1u;
        }}
    }}
    return lower;
}}

const char *nf_srgb_oetf_quantize_thresholds_sha256_v1(void) {{
    return "{digest}";
}}

int nf_srgb_oetf_quantize_apply_v1(
    const float *linear_samples,
    size_t sample_count,
    uint32_t bit_depth,
    void *output,
    size_t output_capacity) {{
    size_t index;
    size_t input_bytes;
    size_t output_bytes;
    size_t output_sample_size;
    uintptr_t input_start;
    uintptr_t output_start;
    uintptr_t input_end;
    uintptr_t output_end;
    if (
        linear_samples == NULL
        || output == NULL
        || sample_count == 0u
        || output_capacity < sample_count
        || (bit_depth != 8u && bit_depth != 16u)
        || !nf_binary32_is_little_endian()
    ) {{
        return 0;
    }}
    output_sample_size = bit_depth == 8u ? 1u : 2u;
    if (
        sample_count > SIZE_MAX / sizeof(float)
        || sample_count > SIZE_MAX / output_sample_size
    ) {{
        return 0;
    }}
    input_bytes = sample_count * sizeof(float);
    output_bytes = sample_count * output_sample_size;
    input_start = (uintptr_t)linear_samples;
    output_start = (uintptr_t)output;
    if (
        input_start % _Alignof(float) != 0u
        || (
            bit_depth == 16u
            && output_start % _Alignof(uint16_t) != 0u
        )
        || input_start > UINTPTR_MAX - input_bytes
        || output_start > UINTPTR_MAX - output_bytes
    ) {{
        return 0;
    }}
    input_end = input_start + input_bytes;
    output_end = output_start + output_bytes;
    if (input_start < output_end && output_start < input_end) {{
        return 0;
    }}
    for (index = 0; index < sample_count; ++index) {{
        const float value = linear_samples[index];
        if (
            value != value
            || value < NF_SRGB_OETF_QUANTIZE_LINEAR_MIN_V1
            || value > NF_SRGB_OETF_QUANTIZE_LINEAR_MAX_V1
        ) {{
            return 0;
        }}
    }}
    if (bit_depth == 8u) {{
        uint8_t *destination = (uint8_t *)output;
        for (index = 0; index < sample_count; ++index) {{
            const float value = linear_samples[index];
            if (value <= 0.0f) {{
                destination[index] = 0u;
            }} else if (value >= 1.0f) {{
                destination[index] = 255u;
            }} else {{
                destination[index] = (uint8_t)nf_quantize_bits(
                    nf_load_binary32_le_bits(&linear_samples[index]),
                    NF_OETF_Q8,
                    255u
                );
            }}
        }}
    }} else {{
        uint16_t *destination = (uint16_t *)output;
        for (index = 0; index < sample_count; ++index) {{
            const float value = linear_samples[index];
            if (value <= 0.0f) {{
                destination[index] = 0u;
            }} else if (value >= 1.0f) {{
                destination[index] = 65535u;
            }} else {{
                destination[index] = (uint16_t)nf_quantize_bits(
                    nf_load_binary32_le_bits(&linear_samples[index]),
                    NF_OETF_Q16,
                    65535u
                );
            }}
        }}
    }}
    return 1;
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    args.header.write_text(encode_header(), encoding="utf-8", newline="\n")
    args.source.write_text(encode_source(), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
