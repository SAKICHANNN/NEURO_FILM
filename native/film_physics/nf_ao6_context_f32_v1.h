#ifndef NF_AO6_CONTEXT_F32_V1_H
#define NF_AO6_CONTEXT_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#include "nf_ao6_base_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_AO6_CONTEXT_F32_BUILD)
#    define NF_AO6_CONTEXT_F32_API __declspec(dllexport)
#  else
#    define NF_AO6_CONTEXT_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_CONTEXT_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_CONTEXT_F32_ABI_VERSION_V1 1u

typedef enum nf_ao6_context_f32_status_v1 {
    NF_AO6_CONTEXT_F32_OK_V1 = 0,
    NF_AO6_CONTEXT_F32_INVALID_ARGUMENT_V1 = 1,
    NF_AO6_CONTEXT_F32_INVALID_PROFILE_V1 = 2,
    NF_AO6_CONTEXT_F32_INVALID_STATE_V1 = 3,
    NF_AO6_CONTEXT_F32_INVALID_INPUT_V1 = 4,
    NF_AO6_CONTEXT_F32_NUMERIC_FAILURE_V1 = 5
} nf_ao6_context_f32_status_v1;

typedef struct nf_ao6_context_f32_state_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t pixel_count;
    double mean[3];
    double m2[3];
} nf_ao6_context_f32_state_v1;

NF_AO6_CONTEXT_F32_API uint32_t
nf_ao6_context_f32_abi_version_v1(void);

NF_AO6_CONTEXT_F32_API nf_ao6_context_f32_status_v1
nf_ao6_context_f32_init_v1(nf_ao6_context_f32_state_v1* state);

/*
 * Updates one row-major stream in encounter order. An invalid batch leaves
 * the accumulator unchanged. Partition size does not affect the result.
 */
NF_AO6_CONTEXT_F32_API nf_ao6_context_f32_status_v1
nf_ao6_context_f32_update_v1(
    const nf_ao6_base_f32_profile_v1* profile,
    nf_ao6_context_f32_state_v1* state,
    const float* encoded_srgb,
    size_t rgb_count);

NF_AO6_CONTEXT_F32_API nf_ao6_context_f32_status_v1
nf_ao6_context_f32_finalize_v1(
    const nf_ao6_context_f32_state_v1* state,
    nf_ao6_base_f32_context_v1* context);

#ifdef __cplusplus
}
#endif

#endif
