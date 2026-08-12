#define NF_SENSITOMETRY_CLOUD_BRIDGE_F32_V2_BUILD
#include "nf_sensitometry_cloud_bridge_f32_v2.h"

#include <math.h>

uint32_t nf_sensitometry_cloud_bridge_f32_abi_version_v2(void) {
    return 2u;
}

int nf_sensitometry_cloud_bridge_f32_apply_window_v2(
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
    size_t output_values) {
    size_t required_counts, required_doubles, required_core, index;
    size_t core_offset;
    if (nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core_height, width, halo, &required_counts, &required_doubles,
            &required_core) != 0 || scene_window_rgb == NULL ||
        capacity_cmy == NULL || developed_density_workspace == NULL ||
        expected_transmittance_workspace == NULL ||
        scene_values < required_counts ||
        developed_workspace_values < required_counts ||
        expected_workspace_values < required_core ||
        halo > SIZE_MAX / width || halo * width > SIZE_MAX / 3u) {
        return 1;
    }
    (void)required_doubles;
    for (index = 0u; index < 3u; ++index) {
        if (!isfinite(capacity_cmy[index]) || capacity_cmy[index] <= 0.0) {
            return 2;
        }
    }
    if (nf_physical_sensitometry_f64_apply_v2(
            domains_profile, scene_window_rgb, required_counts / 3u,
            developed_density_workspace) != 0) {
        return 3;
    }
    for (index = 0u; index < required_counts; ++index) {
        const double value = developed_density_workspace[index];
        const double scale = value / capacity_cmy[index % 3u];
        if (!isfinite(value) || !isfinite(scale) || scale < 0.0 || scale > 1.0) {
            return 4;
        }
    }
    core_offset = halo * width * 3u;
    for (index = 0u; index < required_core; ++index) {
        const double value = pow(10.0, -developed_density_workspace[core_offset + index]);
        if (!isfinite(value) || value <= 0.0 || value >= 1.0) {
            return 5;
        }
        expected_transmittance_workspace[index] = (float)value;
    }
    for (index = 0u; index < required_counts; ++index) {
        developed_density_workspace[index] /= capacity_cmy[index % 3u];
    }
    return (int)nf_conditioned_cloud_row_chain_f32_apply_window_v3(
        count_profile, spatial_profile, full_height, width, first_logical_y,
        core_height, halo, developed_density_workspace, required_counts,
        expected_transmittance_workspace, required_core, channel_gain_cmy,
        count_workspace, count_workspace_values, convolution_workspace,
        convolution_workspace_doubles, spatial_density_workspace,
        spatial_transmittance_workspace, spatial_workspace_values,
        output_density_cmy, output_transmittance_cmy, output_values);
}
