#define NF_CLOUD_INTERPRETATION_CHAIN_F32_BUILD
#include "nf_cloud_interpretation_chain_f32_v1.h"

uint32_t nf_cloud_interpretation_chain_f32_abi_version_v1(void) { return 1u; }

int nf_cloud_interpretation_chain_f32_apply_window_v1(
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
    size_t output_values) {
    size_t counts, doubles, core;
    int status;
    if (nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core_height, width, halo, &counts, &doubles, &core) != 0 ||
        cloud_density_workspace == NULL || cloud_transmittance_workspace == NULL ||
        output_scan_linear_rgb == NULL || cloud_workspace_values < core ||
        output_values < core) return 1;
    status = nf_sensitometry_cloud_bridge_f32_apply_window_v1(
        domains_profile, count_profile, spatial_profile, full_height, width,
        first_logical_y, core_height, halo, scene_window_rgb, scene_values,
        capacity_cmy, developed_scale_workspace, scale_workspace_values,
        expected_transmittance_cmy, expected_values, channel_gain_cmy,
        count_workspace, count_workspace_values, convolution_workspace,
        convolution_workspace_doubles, spatial_density_workspace,
        spatial_transmittance_workspace, spatial_workspace_values,
        cloud_density_workspace, cloud_transmittance_workspace,
        cloud_workspace_values);
    if (status != 0) return 10 + status;
    status = (int)nf_physical_interpretation_f32_apply_v1(
        domains_profile, cloud_density_workspace, core / 3u,
        output_scan_linear_rgb);
    return status == 0 ? 0 : 20 + status;
}
