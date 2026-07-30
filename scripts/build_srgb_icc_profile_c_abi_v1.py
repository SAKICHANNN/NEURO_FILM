"""Generate the freestanding C11 accessor for the pinned sRGB ICC profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.color_match.srgb_icc_profile import (
    SRGB_ICC_PROFILE_SHA256,
    SRGB_ICC_PROFILE_SIZE_BYTES,
    srgb_icc_profile_v1,
)


def encode_header() -> str:
    return f"""#ifndef NF_SRGB_ICC_PROFILE_V1_H
#define NF_SRGB_ICC_PROFILE_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define NF_SRGB_ICC_PROFILE_V1_SIZE {SRGB_ICC_PROFILE_SIZE_BYTES}u
#define NF_SRGB_ICC_PROFILE_V1_SHA256 "{SRGB_ICC_PROFILE_SHA256}"

size_t nf_srgb_icc_profile_size_v1(void);
const char *nf_srgb_icc_profile_sha256_v1(void);
int nf_srgb_icc_profile_copy_v1(uint8_t *output, size_t capacity);

#ifdef __cplusplus
}}
#endif

#endif
"""


def encode_source() -> str:
    profile = srgb_icc_profile_v1()
    rows = []
    for offset in range(0, len(profile), 12):
        encoded = ", ".join(
            f"0x{value:02x}" for value in profile[offset : offset + 12]
        )
        rows.append(f"    {encoded},")
    byte_rows = "\n".join(rows)
    return f"""#include "reference_srgb_icc_profile_v1.h"

static const uint8_t NF_PROFILE[NF_SRGB_ICC_PROFILE_V1_SIZE] = {{
{byte_rows}
}};

size_t nf_srgb_icc_profile_size_v1(void) {{
    return sizeof(NF_PROFILE);
}}

const char *nf_srgb_icc_profile_sha256_v1(void) {{
    return NF_SRGB_ICC_PROFILE_V1_SHA256;
}}

int nf_srgb_icc_profile_copy_v1(uint8_t *output, size_t capacity) {{
    size_t index;
    volatile uint8_t *destination;
    if (output == NULL || capacity < sizeof(NF_PROFILE)) {{
        return 0;
    }}
    destination = output;
    for (index = 0; index < sizeof(NF_PROFILE); ++index) {{
        destination[index] = NF_PROFILE[index];
    }}
    return 1;
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    args.header.write_text(
        encode_header(),
        encoding="utf-8",
        newline="\n",
    )
    args.source.write_text(
        encode_source(),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
