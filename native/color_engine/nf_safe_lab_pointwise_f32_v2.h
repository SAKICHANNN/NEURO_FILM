#ifndef NF_SAFE_LAB_POINTWISE_F32_V2_H
#define NF_SAFE_LAB_POINTWISE_F32_V2_H

#include <stdint.h>

#if defined(_WIN32)
#define NF_COLOR_API __declspec(dllexport)
#else
#define NF_COLOR_API
#endif

enum {
    NF_SAFE_LAB_OK = 0,
    NF_SAFE_LAB_INVALID_ARGUMENT = 1,
    NF_SAFE_LAB_NONFINITE_INPUT = 2,
    NF_SAFE_LAB_NONFINITE_OUTPUT = 3
};

NF_COLOR_API int nf_safe_lab_pointwise_f32_v2(
    const float *source_lab,
    float *output_lab,
    uint64_t pixel_count,
    const float source_mean[3],
    const float source_std[3],
    const float destination_mean[3],
    const float destination_std[3],
    float strength,
    float luma_strength,
    float chroma_curve_strength,
    float neutral_protect,
    double skin_protect,
    float max_chroma_gain,
    float max_chroma_boost,
    float max_chroma_absolute,
    uint32_t thread_count);

#endif
