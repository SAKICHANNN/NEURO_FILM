#ifndef NF_SENSITOMETRY_CLOUD_BRIDGE_F32_V2_H
#define NF_SENSITOMETRY_CLOUD_BRIDGE_F32_V2_H

#include <stddef.h>
#include <stdint.h>

#include "nf_conditioned_cloud_row_chain_f32_v2.h"
#include "nf_physical_domains_f32_v1.h"

#if defined(_WIN32) && defined(NF_SENSITOMETRY_CLOUD_BRIDGE_F32_V2_BUILD)
#define NF_SCB_V2_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_SCB_V2_API __declspec(dllimport)
#else
#define NF_SCB_V2_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

NF_SCB_V2_API uint32_t
nf_sensitometry_cloud_bridge_f32_abi_version_v2(void);

NF_SCB_V2_API int
nf_sensitometry_cloud_bridge_f32_apply_window_v2(
    const nf_physical_domains_f32_profile_v1* domains_profile,
    const nf_density_conditioned_poisson_u16_profile_v3* count_profile,
    const nf_cloud_spatial_response_f32_profile_v2* spatial_profile,
    size_t full_height, size_t width, size_t first_logical_y,
    size_t core_height, size_t halo, const float* scene_window_rgb,
    size_t scene_values, const double capacity_cmy[3],
    double* developed_density_workspace, size_t developed_workspace_values,
    float* expected_transmittance_workspace, size_t expected_workspace_values,
    const float channel_gain_cmy[3], uint16_t* count_workspace,
    size_t count_workspace_values, double* convolution_workspace,
    size_t convolution_workspace_doubles, float* spatial_density_workspace,
    float* spatial_transmittance_workspace, size_t spatial_workspace_values,
    float* output_density_cmy, float* output_transmittance_cmy,
    size_t output_values);

#ifdef __cplusplus
}
#endif

#endif
