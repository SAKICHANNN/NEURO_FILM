#ifndef NF_AO6_CONTEXT_F32_V2_H
#define NF_AO6_CONTEXT_F32_V2_H

#include "nf_ao6_context_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_AO6_CONTEXT_F32_V2_BUILD)
#    define NF_AO6_CONTEXT_F32_V2_API __declspec(dllexport)
#  else
#    define NF_AO6_CONTEXT_F32_V2_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_CONTEXT_F32_V2_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_CONTEXT_F32_ABI_VERSION_V2 2u

/*
 * V2 stores validated density-processed Lab triplets in caller-owned scratch
 * before mutating state, preserving failure atomicity without recomputation.
 */
NF_AO6_CONTEXT_F32_V2_API nf_ao6_context_f32_status_v1
nf_ao6_context_f32_update_v2(
    const nf_ao6_base_f32_profile_v1* profile,
    nf_ao6_context_f32_state_v1* state,
    const float* encoded_srgb,
    size_t rgb_count,
    float* scratch_lab);

#ifdef __cplusplus
}
#endif

#endif
