#ifndef NF_SRGB_EOTF_F32_V1_H
#define NF_SRGB_EOTF_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NF_SRGB_EOTF_F32_LUT_V1_SHA256 "1b8f915b4ddf4dc2b4aa961934b05549d23f4593a7b65c2aacfdb7eaea80a1ca"

const char *nf_srgb_eotf_f32_lut_sha256_v1(void);
int nf_srgb_eotf_f32_apply_v1(
    const void *samples,
    size_t sample_count,
    uint32_t bit_depth,
    float *output,
    size_t output_capacity);

#ifdef __cplusplus
}
#endif

#endif
