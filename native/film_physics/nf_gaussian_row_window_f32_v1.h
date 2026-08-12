#ifndef NF_GAUSSIAN_ROW_WINDOW_F32_V1_H
#define NF_GAUSSIAN_ROW_WINDOW_F32_V1_H

#include "nf_gaussian_rgb_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_GAUSSIAN_ROW_WINDOW_F32_BUILD)
#    define NF_GAUSSIAN_ROW_WINDOW_F32_API __declspec(dllexport)
#  else
#    define NF_GAUSSIAN_ROW_WINDOW_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_GAUSSIAN_ROW_WINDOW_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_GAUSSIAN_ROW_WINDOW_F32_ABI_VERSION_V1 1u

NF_GAUSSIAN_ROW_WINDOW_F32_API uint32_t
nf_gaussian_row_window_f32_abi_version_v1(void);

NF_GAUSSIAN_ROW_WINDOW_F32_API nf_gaussian_f32_status_v1
nf_gaussian_row_window_f32_apply_v1(
    const nf_gaussian_f32_profile_v1* profile,
    size_t full_height,
    size_t full_width,
    size_t input_logical_start,
    size_t input_height,
    const float* input_rgb,
    size_t core_logical_start,
    size_t core_height,
    float* workspace_rgb,
    size_t workspace_floats,
    float* window_output_rgb,
    size_t window_output_floats,
    float* core_output_rgb,
    size_t core_output_floats);

#ifdef __cplusplus
}
#endif

#endif
