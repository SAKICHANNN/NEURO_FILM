#ifndef NF_CLOUD_ATTENUATION_F32_V1_H
#define NF_CLOUD_ATTENUATION_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#define NF_CLOUD_ATTENUATION_F32_ABI_VERSION_V1 1u

#if defined(_WIN32) && defined(NF_CLOUD_ATTENUATION_F32_BUILD)
#define NF_CLOUD_ATTENUATION_F32_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_CLOUD_ATTENUATION_F32_API __declspec(dllimport)
#else
#define NF_CLOUD_ATTENUATION_F32_API __attribute__((visibility("default")))
#endif

typedef enum nf_cloud_attenuation_f32_status_v1 {
    NF_CLOUD_ATTENUATION_F32_OK_V1 = 0,
    NF_CLOUD_ATTENUATION_F32_INVALID_ARGUMENT_V1 = 1,
    NF_CLOUD_ATTENUATION_F32_DOMAIN_ERROR_V1 = 2
} nf_cloud_attenuation_f32_status_v1;

NF_CLOUD_ATTENUATION_F32_API uint32_t
nf_cloud_attenuation_f32_abi_version_v1(void);

/* All arrays are interleaved RGB with 3*pixel_count floats. */
NF_CLOUD_ATTENUATION_F32_API nf_cloud_attenuation_f32_status_v1
nf_cloud_attenuation_f32_apply_v1(
    const float* expected_transmittance_rgb,
    const float* base_transmittance_rgb,
    size_t pixel_count,
    const float channel_gain_rgb[3],
    float* output_density_rgb,
    float* output_transmittance_rgb);

#endif
