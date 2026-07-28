#ifndef NF_SRGB_ICC_PROFILE_V1_H
#define NF_SRGB_ICC_PROFILE_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NF_SRGB_ICC_PROFILE_V1_SIZE 588u
#define NF_SRGB_ICC_PROFILE_V1_SHA256 "217fe48ec958c667f8eef725aa27198f465df95d7662593b90d0a1cc30114356"

size_t nf_srgb_icc_profile_size_v1(void);
const char *nf_srgb_icc_profile_sha256_v1(void);
int nf_srgb_icc_profile_copy_v1(uint8_t *output, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
