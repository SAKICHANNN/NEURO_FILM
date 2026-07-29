#ifndef NF_AO6_BASE_F32_V3_H
#define NF_AO6_BASE_F32_V3_H

#include "nf_ao6_base_f32_v2.h"

#if defined(_WIN32)
#  if defined(NF_AO6_BASE_F32_V3_BUILD)
#    define NF_AO6_BASE_F32_V3_API __declspec(dllexport)
#  else
#    define NF_AO6_BASE_F32_V3_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_BASE_F32_V3_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_BASE_F32_ABI_VERSION_V3 3u

/*
 * Produces the exact v2 base output with one pixel computation. The caller
 * designates output_encoded_srgb as expendable scratch: invalid input leaves
 * it unchanged, while a later numeric failure may leave partial scratch.
 */
NF_AO6_BASE_F32_V3_API nf_ao6_base_f32_status_v1
nf_ao6_base_f32_apply_scratch_v3(
    const nf_ao6_base_f32_profile_v1* profile,
    const nf_ao6_base_f32_context_v1* context,
    const float* encoded_srgb,
    size_t rgb_count,
    float* output_encoded_srgb);

#ifdef __cplusplus
}
#endif

#endif
