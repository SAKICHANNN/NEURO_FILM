"""Generate an exact float32 sRGB EOTF LUT and freestanding C11 ABI."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from src.color_match.shared_runtime_staging_match_views import (
    _decode_srgb_samples_f32,
)


def _table(bit_depth: int) -> np.ndarray:
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    count = 256 if bit_depth == 8 else 65536
    values = np.arange(count, dtype=dtype)
    samples = np.repeat(values[:, None, None], 3, axis=2)
    samples.flags.writeable = False
    return _decode_srgb_samples_f32(
        samples,
        bit_depth=bit_depth,
    )[:, 0, 0]


def tables() -> tuple[np.ndarray, np.ndarray, str]:
    table8 = _table(8)
    table16 = _table(16)
    payload = (
        table8.astype(">f4", copy=False).tobytes()
        + table16.astype(">f4", copy=False).tobytes()
    )
    return table8, table16, hashlib.sha256(payload).hexdigest()


def encode_header() -> str:
    _, _, digest = tables()
    return f"""#ifndef NF_SRGB_EOTF_F32_V1_H
#define NF_SRGB_EOTF_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define NF_SRGB_EOTF_F32_LUT_V1_SHA256 "{digest}"

const char *nf_srgb_eotf_f32_lut_sha256_v1(void);
int nf_srgb_eotf_f32_apply_v1(
    const void *samples,
    size_t sample_count,
    uint32_t bit_depth,
    float *output,
    size_t output_capacity);

#ifdef __cplusplus
}}
#endif

#endif
"""


def _encode_bits(name: str, values: np.ndarray) -> str:
    bits = values.view(np.uint32)
    rows = []
    for offset in range(0, bits.size, 8):
        encoded = ", ".join(
            f"0x{int(value):08x}u" for value in bits[offset : offset + 8]
        )
        rows.append(f"    {encoded},")
    return (
        f"static const uint32_t {name}[{bits.size}u] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def encode_source() -> str:
    table8, table16, digest = tables()
    return f"""#include "reference_srgb_eotf_f32_v1.h"

#include <float.h>
#include <stdint.h>

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "nf_srgb_eotf_f32_v1 requires IEEE-754 binary32 float"
#endif

{_encode_bits("NF_EOTF_8", table8)}
{_encode_bits("NF_EOTF_16", table16)}
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

static void nf_store_binary32_le(float *destination, uint32_t bits) {{
    unsigned char *bytes = (unsigned char *)destination;
    bytes[0] = (unsigned char)(bits & 0xffu);
    bytes[1] = (unsigned char)((bits >> 8u) & 0xffu);
    bytes[2] = (unsigned char)((bits >> 16u) & 0xffu);
    bytes[3] = (unsigned char)((bits >> 24u) & 0xffu);
}}

const char *nf_srgb_eotf_f32_lut_sha256_v1(void) {{
    return "{digest}";
}}

int nf_srgb_eotf_f32_apply_v1(
    const void *samples,
    size_t sample_count,
    uint32_t bit_depth,
    float *output,
    size_t output_capacity) {{
    size_t index;
    size_t input_bytes;
    size_t output_bytes;
    uintptr_t input_start;
    uintptr_t output_start;
    uintptr_t input_end;
    uintptr_t output_end;
    size_t sample_size;
    if (
        samples == NULL
        || output == NULL
        || sample_count == 0u
        || output_capacity < sample_count
        || (bit_depth != 8u && bit_depth != 16u)
        || !nf_binary32_is_little_endian()
    ) {{
        return 0;
    }}
    sample_size = bit_depth == 8u ? 1u : 2u;
    if (
        sample_count > SIZE_MAX / sample_size
        || sample_count > SIZE_MAX / sizeof(float)
    ) {{
        return 0;
    }}
    input_bytes = sample_count * sample_size;
    output_bytes = sample_count * sizeof(float);
    input_start = (uintptr_t)samples;
    output_start = (uintptr_t)output;
    if (
        output_start % _Alignof(float) != 0u
        || (
            bit_depth == 16u
            && input_start % _Alignof(uint16_t) != 0u
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
    if (bit_depth == 8u) {{
        const uint8_t *input = (const uint8_t *)samples;
        for (index = 0; index < sample_count; ++index) {{
            nf_store_binary32_le(&output[index], NF_EOTF_8[input[index]]);
        }}
    }} else {{
        const uint16_t *input = (const uint16_t *)samples;
        for (index = 0; index < sample_count; ++index) {{
            nf_store_binary32_le(&output[index], NF_EOTF_16[input[index]]);
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
