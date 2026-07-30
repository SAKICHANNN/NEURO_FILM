#ifndef NEURO_FILM_REFERENCE_CANONICAL_CORE_H
#define NEURO_FILM_REFERENCE_CANONICAL_CORE_H

typedef unsigned char nf_u8;
typedef unsigned int nf_u32;
typedef unsigned long long nf_u64;

#if defined(__cplusplus)
extern "C" {
#endif

int nf_reference_sha256(
    const nf_u8* input,
    nf_u64 input_size,
    nf_u8 output[32]
);

nf_u32 nf_reference_staging_authorized(
    nf_u32 numeric_eligible,
    nf_u32 all_promoted,
    nf_u32 any_research_override
);

#if defined(__cplusplus)
}
#endif

#endif
