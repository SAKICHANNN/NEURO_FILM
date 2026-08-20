#ifndef NF_REC2020_LAB_SOURCE_COMPRESS_F32_V1_H
#define NF_REC2020_LAB_SOURCE_COMPRESS_F32_V1_H

#include <stdint.h>

#if defined(_WIN32)
#define NF_COLOR_API __declspec(dllexport)
#else
#define NF_COLOR_API
#endif

enum {
    NF_LAB_COMPRESS_OK = 0,
    NF_LAB_COMPRESS_INVALID_ARGUMENT = 1,
    NF_LAB_COMPRESS_NONFINITE_INPUT = 2,
    NF_LAB_COMPRESS_SOURCE_OUT_OF_GAMUT = 3,
    NF_LAB_COMPRESS_OUTPUT_OUT_OF_GAMUT = 4
};

NF_COLOR_API int nf_rec2020_lab_source_compress_f32_v1(
    const float *source_lab,
    const float *target_lab,
    float *output_lab,
    float *output_rgb,
    float *scale,
    uint64_t pixel_count,
    uint32_t iterations,
    double tolerance,
    uint32_t thread_count);

#endif
