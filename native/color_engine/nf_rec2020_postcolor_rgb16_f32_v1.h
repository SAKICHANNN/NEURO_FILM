#ifndef NF_REC2020_POSTCOLOR_RGB16_F32_V1_H
#define NF_REC2020_POSTCOLOR_RGB16_F32_V1_H

#include <stdint.h>

#if defined(_WIN32)
#define NF_COLOR_API __declspec(dllexport)
#else
#define NF_COLOR_API
#endif

enum {
    NF_POSTCOLOR_OK = 0,
    NF_POSTCOLOR_INVALID_ARGUMENT = 1,
    NF_POSTCOLOR_INVALID_INPUT = 2,
    NF_POSTCOLOR_INVALID_OUTPUT = 3
};

NF_COLOR_API int nf_rec2020_postcolor_rgb16_f32_v1(
    const float *source_rgb,
    const float *candidate_rgb,
    const float *quantization_thresholds,
    const uint32_t *quantization_buckets,
    uint16_t *output_rgb16,
    float *residual_scale,
    uint64_t pixel_count,
    uint32_t threshold_count,
    uint32_t bucket_count,
    double margin,
    uint32_t thread_count);

#endif
