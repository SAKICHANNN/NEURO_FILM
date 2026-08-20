#ifndef NF_SAFE_LAB_SPATIAL_TONE_F32_V1_H
#define NF_SAFE_LAB_SPATIAL_TONE_F32_V1_H

#include <stdint.h>

#if defined(_WIN32) && defined(NF_SAFE_LAB_SPATIAL_TONE_BUILD)
#define NF_SAFE_LAB_SPATIAL_TONE_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_SAFE_LAB_SPATIAL_TONE_API __declspec(dllimport)
#else
#define NF_SAFE_LAB_SPATIAL_TONE_API
#endif

enum {
    NF_SAFE_LAB_SPATIAL_TONE_OK = 0,
    NF_SAFE_LAB_SPATIAL_TONE_INVALID_ARGUMENT = 1,
    NF_SAFE_LAB_SPATIAL_TONE_NONFINITE_INPUT = 2,
    NF_SAFE_LAB_SPATIAL_TONE_NONFINITE_OUTPUT = 3
};

NF_SAFE_LAB_SPATIAL_TONE_API int nf_safe_lab_spatial_tone_f32_v1(
    const float *source_lab,
    const float *pointwise_lab,
    float *output_lab,
    float *source_workspace,
    float *target_workspace,
    uint64_t height,
    uint64_t width,
    float sigma,
    float truncate,
    float detail_strength,
    float tone_strength,
    float shadow_floor_l,
    float highlight_ceiling_l,
    uint32_t thread_count);

#endif
