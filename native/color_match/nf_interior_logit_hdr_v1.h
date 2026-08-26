#ifndef NF_INTERIOR_LOGIT_HDR_V1_H
#define NF_INTERIOR_LOGIT_HDR_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_INTERIOR_LOGIT_HDR_BUILD)
#    define NF_INTERIOR_LOGIT_HDR_API __declspec(dllexport)
#  else
#    define NF_INTERIOR_LOGIT_HDR_API __declspec(dllimport)
#  endif
#else
#  define NF_INTERIOR_LOGIT_HDR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_INTERIOR_LOGIT_HDR_ABI_VERSION_V1 1u
#define NF_INTERIOR_LOGIT_HDR_KNOT_COUNT_V1 33u
#define NF_INTERIOR_LOGIT_HDR_CHANNELS_V1 3u
#define NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1 99u

typedef enum nf_interior_logit_hdr_status_v1 {
    NF_INTERIOR_LOGIT_HDR_OK_V1 = 0,
    NF_INTERIOR_LOGIT_HDR_INVALID_ARGUMENT_V1 = 1,
    NF_INTERIOR_LOGIT_HDR_INVALID_KNOTS_V1 = 2,
    NF_INTERIOR_LOGIT_HDR_INVALID_INPUT_V1 = 3,
    NF_INTERIOR_LOGIT_HDR_NUMERIC_FAILURE_V1 = 4,
    NF_INTERIOR_LOGIT_HDR_OVERLAP_V1 = 5
} nf_interior_logit_hdr_status_v1;

typedef struct nf_interior_logit_hdr_diagnostics_v1 {
    uint32_t abi_version;
    uint32_t identity;
    uint64_t triplet_count;
    uint64_t preserved_boundary_values;
    uint64_t strict_interior_values;
} nf_interior_logit_hdr_diagnostics_v1;

NF_INTERIOR_LOGIT_HDR_API uint32_t
nf_interior_logit_hdr_abi_version_v1(void);

/*
 * Knots are 33 row-major RGB float32 logit triplets. Samples are interleaved
 * float32 absolute Rec.2020 RGB values in [0, 10000]. Exact in-place execution
 * is allowed; every other overlap is rejected. Every failure leaves both the
 * output and diagnostics buffers untouched.
 */
NF_INTERIOR_LOGIT_HDR_API nf_interior_logit_hdr_status_v1
nf_interior_logit_hdr_apply_v1(
    const float source_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    const float reference_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    uint32_t identity,
    const float* input_rgb,
    size_t triplet_count,
    float* output_rgb,
    size_t output_value_count,
    nf_interior_logit_hdr_diagnostics_v1* diagnostics);

#ifdef __cplusplus
}
#endif

#endif
