#ifndef NF_REC2020_OKLAB_INTERIOR_F32_V1_H
#define NF_REC2020_OKLAB_INTERIOR_F32_V1_H

#include <stdint.h>

#if defined(_WIN32)
#define NF_COLOR_API __declspec(dllexport)
#else
#define NF_COLOR_API
#endif

enum {
    NF_COLOR_OK = 0,
    NF_COLOR_INVALID_ARGUMENT = 1,
    NF_COLOR_NONFINITE_INPUT = 2,
    NF_COLOR_MAPPING_FAILED = 3
};

NF_COLOR_API int nf_rec2020_oklab_interior_f32_v1(
    const float *input_rgb,
    float *output_rgb,
    float *chroma_scale,
    uint64_t pixel_count,
    double softness,
    double margin,
    uint32_t iterations,
    uint32_t thread_count);

#endif
