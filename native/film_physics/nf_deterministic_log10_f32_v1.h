#ifndef NF_DETERMINISTIC_LOG10_F32_V1_H
#define NF_DETERMINISTIC_LOG10_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#define NF_DETERMINISTIC_LOG10_F32_ABI_VERSION_V1 1u

#if defined(_WIN32) && defined(NF_DETERMINISTIC_LOG10_F32_BUILD)
#define NF_DETERMINISTIC_LOG10_F32_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_DETERMINISTIC_LOG10_F32_API __declspec(dllimport)
#else
#define NF_DETERMINISTIC_LOG10_F32_API __attribute__((visibility("default")))
#endif

typedef enum nf_deterministic_log10_f32_status_v1 {
    NF_DETERMINISTIC_LOG10_F32_OK_V1 = 0,
    NF_DETERMINISTIC_LOG10_F32_INVALID_ARGUMENT_V1 = 1,
    NF_DETERMINISTIC_LOG10_F32_DOMAIN_ERROR_V1 = 2
} nf_deterministic_log10_f32_status_v1;

NF_DETERMINISTIC_LOG10_F32_API uint32_t
nf_deterministic_log10_f32_abi_version_v1(void);

NF_DETERMINISTIC_LOG10_F32_API nf_deterministic_log10_f32_status_v1
nf_deterministic_neg_log10_f32_apply_v1(
    const float* input,
    size_t count,
    float* output);

#endif
