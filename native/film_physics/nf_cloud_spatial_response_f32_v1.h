#ifndef NF_CLOUD_SPATIAL_RESPONSE_F32_V1_H
#define NF_CLOUD_SPATIAL_RESPONSE_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#define NF_CLOUD_SPATIAL_RESPONSE_F32_ABI_VERSION_V1 1u
#define NF_CLOUD_SPATIAL_RESPONSE_F32_MAX_RADIUS_V1 32u

#if defined(_WIN32) && defined(NF_CLOUD_SPATIAL_RESPONSE_F32_BUILD)
#define NF_CLOUD_SPATIAL_RESPONSE_F32_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_CLOUD_SPATIAL_RESPONSE_F32_API __declspec(dllimport)
#else
#define NF_CLOUD_SPATIAL_RESPONSE_F32_API __attribute__((visibility("default")))
#endif

typedef struct nf_cloud_spatial_response_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    double sigma_pixels_cmy[3];
    double mark_optical_density_cmy[3];
    double truncate;
} nf_cloud_spatial_response_f32_profile_v1;

typedef enum nf_cloud_spatial_response_f32_status_v1 {
    NF_CLOUD_SPATIAL_RESPONSE_F32_OK_V1 = 0,
    NF_CLOUD_SPATIAL_RESPONSE_F32_INVALID_ARGUMENT_V1 = 1,
    NF_CLOUD_SPATIAL_RESPONSE_F32_DOMAIN_ERROR_V1 = 2
} nf_cloud_spatial_response_f32_status_v1;

NF_CLOUD_SPATIAL_RESPONSE_F32_API uint32_t
nf_cloud_spatial_response_f32_abi_version_v1(void);

/* Counts cover core_height + 2*halo rows. Workspace requires 2*extended pixels. */
NF_CLOUD_SPATIAL_RESPONSE_F32_API nf_cloud_spatial_response_f32_status_v1
nf_cloud_spatial_response_f32_apply_v1(
    const nf_cloud_spatial_response_f32_profile_v1* profile,
    const uint16_t* extended_counts_cmy,
    size_t core_height,
    size_t width,
    size_t halo,
    double* workspace,
    size_t workspace_doubles,
    float* output_density_cmy,
    float* output_transmittance_cmy,
    size_t output_values);

#endif
