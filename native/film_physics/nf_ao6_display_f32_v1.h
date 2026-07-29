#ifndef NF_AO6_DISPLAY_F32_V1_H
#define NF_AO6_DISPLAY_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#include "nf_ao6_base_f32_v1.h"
#include "nf_ao6_residual_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_AO6_DISPLAY_F32_BUILD)
#    define NF_AO6_DISPLAY_F32_API __declspec(dllexport)
#  else
#    define NF_AO6_DISPLAY_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_DISPLAY_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_DISPLAY_F32_ABI_VERSION_V1 1u

typedef enum nf_ao6_display_f32_status_v1 {
    NF_AO6_DISPLAY_F32_OK_V1 = 0,
    NF_AO6_DISPLAY_F32_INVALID_ARGUMENT_V1 = 1,
    NF_AO6_DISPLAY_F32_BASE_FAILURE_V1 = 2,
    NF_AO6_DISPLAY_F32_RESIDUAL_FAILURE_V1 = 3,
    NF_AO6_DISPLAY_F32_NUMERIC_FAILURE_V1 = 4
} nf_ao6_display_f32_status_v1;

NF_AO6_DISPLAY_F32_API uint32_t
nf_ao6_display_f32_abi_version_v1(void);

/*
 * Applies the complete AO6 display approximation to one bounded row tile:
 * active source-context base -> exact sRGB EOTF -> t15/c35 residual ->
 * exact sRGB OETF. scratch_rgb must be a distinct rgb_count*3 float buffer.
 * Final output is untouched unless all stages succeed.
 */
NF_AO6_DISPLAY_F32_API nf_ao6_display_f32_status_v1
nf_ao6_display_f32_apply_v1(
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
