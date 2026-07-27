#include "reference_canonical_core.h"

typedef char nf_u32_must_be_32_bits[(sizeof(nf_u32) == 4) ? 1 : -1];
typedef char nf_u64_must_be_64_bits[(sizeof(nf_u64) == 8) ? 1 : -1];

typedef struct nf_sha256_context {
    nf_u32 state[8];
    nf_u8 block[64];
    nf_u32 block_size;
    nf_u64 total_size;
} nf_sha256_context;

static const nf_u32 NF_ROUND[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static nf_u32 nf_rotate_right(nf_u32 value, nf_u32 count) {
    return (value >> count) | (value << (32U - count));
}

static void nf_sha256_transform(
    nf_sha256_context* context,
    const nf_u8 block[64]
) {
    nf_u32 words[64];
    nf_u32 a;
    nf_u32 b;
    nf_u32 c;
    nf_u32 d;
    nf_u32 e;
    nf_u32 f;
    nf_u32 g;
    nf_u32 h;
    nf_u32 index;

    for (index = 0; index < 16U; ++index) {
        const nf_u32 cursor = index * 4U;
        words[index] =
            ((nf_u32)block[cursor] << 24U)
            | ((nf_u32)block[cursor + 1U] << 16U)
            | ((nf_u32)block[cursor + 2U] << 8U)
            | (nf_u32)block[cursor + 3U];
    }
    for (index = 16U; index < 64U; ++index) {
        const nf_u32 first = words[index - 15U];
        const nf_u32 second = words[index - 2U];
        const nf_u32 sigma0 =
            nf_rotate_right(first, 7U)
            ^ nf_rotate_right(first, 18U)
            ^ (first >> 3U);
        const nf_u32 sigma1 =
            nf_rotate_right(second, 17U)
            ^ nf_rotate_right(second, 19U)
            ^ (second >> 10U);
        words[index] =
            words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (index = 0; index < 64U; ++index) {
        const nf_u32 sum1 =
            nf_rotate_right(e, 6U)
            ^ nf_rotate_right(e, 11U)
            ^ nf_rotate_right(e, 25U);
        const nf_u32 choose = (e & f) ^ ((~e) & g);
        const nf_u32 temp1 =
            h + sum1 + choose + NF_ROUND[index] + words[index];
        const nf_u32 sum0 =
            nf_rotate_right(a, 2U)
            ^ nf_rotate_right(a, 13U)
            ^ nf_rotate_right(a, 22U);
        const nf_u32 majority = (a & b) ^ (a & c) ^ (b & c);
        const nf_u32 temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void nf_sha256_init(nf_sha256_context* context) {
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
    context->block_size = 0U;
    context->total_size = 0U;
}

static void nf_sha256_update(
    nf_sha256_context* context,
    const nf_u8* input,
    nf_u64 input_size
) {
    nf_u64 index;
    for (index = 0U; index < input_size; ++index) {
        context->block[context->block_size] = input[index];
        context->block_size += 1U;
        if (context->block_size == 64U) {
            nf_sha256_transform(context, context->block);
            context->block_size = 0U;
        }
    }
    context->total_size += input_size;
}

static void nf_sha256_finish(
    nf_sha256_context* context,
    nf_u8 output[32]
) {
    const nf_u64 bit_length = context->total_size * 8U;
    nf_u32 index;

    context->block[context->block_size] = 0x80U;
    context->block_size += 1U;
    if (context->block_size > 56U) {
        while (context->block_size < 64U) {
            context->block[context->block_size] = 0U;
            context->block_size += 1U;
        }
        nf_sha256_transform(context, context->block);
        context->block_size = 0U;
    }
    while (context->block_size < 56U) {
        context->block[context->block_size] = 0U;
        context->block_size += 1U;
    }
    for (index = 0U; index < 8U; ++index) {
        context->block[56U + index] =
            (nf_u8)((bit_length >> (56U - index * 8U)) & 0xffU);
    }
    nf_sha256_transform(context, context->block);

    for (index = 0U; index < 8U; ++index) {
        output[index * 4U] = (nf_u8)(context->state[index] >> 24U);
        output[index * 4U + 1U] =
            (nf_u8)(context->state[index] >> 16U);
        output[index * 4U + 2U] =
            (nf_u8)(context->state[index] >> 8U);
        output[index * 4U + 3U] = (nf_u8)context->state[index];
    }
}

int nf_reference_sha256(
    const nf_u8* input,
    nf_u64 input_size,
    nf_u8 output[32]
) {
    nf_sha256_context context;
    if (output == (nf_u8*)0) {
        return 0;
    }
    if (input_size != 0U && input == (const nf_u8*)0) {
        return 0;
    }
    nf_sha256_init(&context);
    nf_sha256_update(&context, input, input_size);
    nf_sha256_finish(&context, output);
    return 1;
}

nf_u32 nf_reference_staging_authorized(
    nf_u32 numeric_eligible,
    nf_u32 all_promoted,
    nf_u32 any_research_override
) {
    return (
        numeric_eligible != 0U
        && all_promoted != 0U
        && any_research_override == 0U
    ) ? 1U : 0U;
}
