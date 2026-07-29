#ifndef NF_AO6_RESIDUAL_F32_V2_H
#define NF_AO6_RESIDUAL_F32_V2_H

#include "nf_ao6_residual_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_AO6_RESIDUAL_F32_V2_BUILD)
#    define NF_AO6_RESIDUAL_F32_V2_API __declspec(dllexport)
#  else
#    define NF_AO6_RESIDUAL_F32_V2_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_RESIDUAL_F32_V2_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_RESIDUAL_F32_ABI_VERSION_V2 2u

/*
 * Produces exact v1 output with one pixel computation. Destination and
 * optional scale arrays are expendable scratch after input validation.
 */
NF_AO6_RESIDUAL_F32_V2_API nf_ao6_residual_f32_status_v1
nf_ao6_residual_f32_apply_scratch_v2(
    const nf_ao6_residual_f32_profile_v1* profile,
    const float* base_linear_rgb,
    size_t rgb_count,
    float* output_linear_rgb,
    float* tone_scale,
    float* chroma_scale);

#ifdef __cplusplus
}
#endif

#endif
