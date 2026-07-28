#ifndef NF_SRGB_OETF_QUANTIZE_V1_H
#define NF_SRGB_OETF_QUANTIZE_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NF_SRGB_OETF_QUANTIZE_THRESHOLDS_V1_SHA256 "fae645ef1aad04fcd1233631a32f820cf7696e3ca65d31939acf60d7f123674c"
#define NF_SRGB_OETF_QUANTIZE_LINEAR_MIN_V1 -0x1.0c6f7a0000000p-19f
#define NF_SRGB_OETF_QUANTIZE_LINEAR_MAX_V1 0x1.0000200000000p+0f

const char *nf_srgb_oetf_quantize_thresholds_sha256_v1(void);
int nf_srgb_oetf_quantize_apply_v1(
    const float *linear_samples,
    size_t sample_count,
    uint32_t bit_depth,
    void *output,
    size_t output_capacity);

#ifdef __cplusplus
}
#endif

#endif
