#ifndef NF_CLOUD_INTERPRETATION_CHAIN_F32_V1_H
#define NF_CLOUD_INTERPRETATION_CHAIN_F32_V1_H

#include "nf_sensitometry_cloud_bridge_f32_v1.h"

#if defined(_WIN32)
# if defined(NF_CLOUD_INTERPRETATION_CHAIN_F32_BUILD)
#  define NF_CLOUD_INTERPRETATION_CHAIN_F32_API __declspec(dllexport)
# else
#  define NF_CLOUD_INTERPRETATION_CHAIN_F32_API __declspec(dllimport)
# endif
#else
# define NF_CLOUD_INTERPRETATION_CHAIN_F32_API
#endif

NF_CLOUD_INTERPRETATION_CHAIN_F32_API uint32_t
nf_cloud_interpretation_chain_f32_abi_version_v1(void);

NF_CLOUD_INTERPRETATION_CHAIN_F32_API int
nf_cloud_interpretation_chain_f32_apply_window_v1(
    const nf_physical_domains_f32_profile_v1* domains_profile,
    const nf_density_conditioned_poisson_u16_profile_v3* count_profile,
    const nf_cloud_spatial_response_f32_profile_v2* spatial_profile,
    size_t full_height, size_t width, size_t first_logical_y,
    size_t core_height, size_t halo, const float* scene_window_rgb,
    size_t scene_values, const double capacity_cmy[3],
    double* developed_scale_workspace, size_t scale_workspace_values,
    const float* expected_transmittance_cmy, size_t expected_values,
    const float channel_gain_cmy[3], uint16_t* count_workspace,
    size_t count_workspace_values, double* convolution_workspace,
    size_t convolution_workspace_doubles, float* spatial_density_workspace,
    float* spatial_transmittance_workspace, size_t spatial_workspace_values,
    float* cloud_density_workspace, float* cloud_transmittance_workspace,
    size_t cloud_workspace_values, float* output_scan_linear_rgb,
    size_t output_values);

#endif
