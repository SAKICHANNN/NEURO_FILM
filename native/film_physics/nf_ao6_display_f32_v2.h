#ifndef NF_AO6_DISPLAY_F32_V2_H
#define NF_AO6_DISPLAY_F32_V2_H

#include "nf_ao6_base_f32_v2.h"
#include "nf_ao6_display_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_AO6_DISPLAY_F32_V2_BUILD)
#    define NF_AO6_DISPLAY_F32_V2_API __declspec(dllexport)
#  else
#    define NF_AO6_DISPLAY_F32_V2_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_DISPLAY_F32_V2_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_DISPLAY_F32_ABI_VERSION_V2 2u

NF_AO6_DISPLAY_F32_V2_API nf_ao6_display_f32_status_v1
nf_ao6_display_f32_apply_v2(
    const nf_ao6_base_f32_profile_v1* base_profile,
    const nf_ao6_base_f32_context_v1* context,
    const nf_ao6_residual_f32_profile_v1* residual_profile,
    const float* input_encoded_srgb,
    size_t rgb_count,
    float* scratch_rgb,
    float* output_encoded_srgb);

#ifdef __cplusplus
}
#endif

#endif
