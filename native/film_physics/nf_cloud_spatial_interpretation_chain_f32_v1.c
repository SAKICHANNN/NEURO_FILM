#define NF_CLOUD_SPATIAL_INTERPRETATION_CHAIN_F32_BUILD
#include "nf_cloud_spatial_interpretation_chain_f32_v1.h"

uint32_t nf_cloud_spatial_interpretation_chain_f32_abi_version_v1(void) {
    return NF_CLOUD_SPATIAL_INTERPRETATION_CHAIN_F32_ABI_VERSION_V1;
}

int nf_cloud_spatial_interpretation_chain_f32_apply_window_v1(
    const nf_physical_domains_f32_profile_v1* domains_profile,
    const nf_density_conditioned_poisson_u16_profile_v3* count_profile,
    const nf_cloud_spatial_response_f32_profile_v2* cloud_profile,
    const nf_gaussian_f32_profile_v1* adjacency_blur_profile,
    const nf_bounded_adjacency_f32_profile_v1* adjacency_profile,
    const nf_gaussian_f32_profile_v1* dye_diffusion_profile,
    const nf_gaussian_f32_profile_v1* scanner_mtf_profile,
    size_t full_height, size_t width, size_t first_logical_y,
    size_t core_height, size_t cloud_halo, const float* scene_window_rgb,
    size_t scene_values, const double capacity_cmy[3],
    double* developed_scale_workspace, size_t scale_workspace_values,
    const float* expected_transmittance_cmy, size_t expected_values,
    const float channel_gain_cmy[3], uint16_t* count_workspace,
    size_t count_workspace_values, double* convolution_workspace,
    size_t convolution_workspace_doubles, float* cloud_density_workspace,
    float* cloud_transmittance_workspace, float* cloud_spatial_density_workspace,
    float* cloud_spatial_transmittance_workspace,
    size_t cloud_workspace_values, float* gaussian_workspace,
    float* blurred_density_workspace, float* adjacent_density_workspace,
    float* diffused_density_workspace, float* interpreted_workspace,
    size_t post_cloud_workspace_values, float* output_scan_linear_rgb,
    size_t output_values) {
    size_t required_counts = 0u, required_doubles = 0u, core_values = 0u;
    int status;
    if (nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core_height, width, cloud_halo, &required_counts,
            &required_doubles, &core_values) != 0 ||
        gaussian_workspace == NULL || blurred_density_workspace == NULL ||
        adjacent_density_workspace == NULL || diffused_density_workspace == NULL ||
        interpreted_workspace == NULL || output_scan_linear_rgb == NULL ||
        post_cloud_workspace_values < core_values || output_values < core_values)
        return 1;
    (void)required_counts;
    (void)required_doubles;
    status = nf_sensitometry_cloud_bridge_f32_apply_window_v1(
        domains_profile, count_profile, cloud_profile, full_height, width,
        first_logical_y, core_height, cloud_halo, scene_window_rgb, scene_values,
        capacity_cmy, developed_scale_workspace, scale_workspace_values,
        expected_transmittance_cmy, expected_values, channel_gain_cmy,
        count_workspace, count_workspace_values, convolution_workspace,
        convolution_workspace_doubles, cloud_spatial_density_workspace,
        cloud_spatial_transmittance_workspace, cloud_workspace_values,
        cloud_density_workspace, cloud_transmittance_workspace,
        cloud_workspace_values);
    if (status != 0) return 10 + status;
    status = (int)nf_gaussian_f32_apply_v1(
        adjacency_blur_profile, cloud_density_workspace, core_height, width,
        gaussian_workspace, post_cloud_workspace_values,
        blurred_density_workspace);
    if (status != 0) return 30 + status;
    status = (int)nf_bounded_adjacency_f32_apply_v1(
        adjacency_profile, cloud_density_workspace, blurred_density_workspace,
        core_values / 3u, adjacent_density_workspace);
    if (status != 0) return 40 + status;
    status = (int)nf_gaussian_f32_apply_v1(
        dye_diffusion_profile, adjacent_density_workspace, core_height, width,
        gaussian_workspace, post_cloud_workspace_values,
        diffused_density_workspace);
    if (status != 0) return 50 + status;
    status = (int)nf_physical_interpretation_f32_apply_v1(
        domains_profile, diffused_density_workspace, core_values / 3u,
        interpreted_workspace);
    if (status != 0) return 60 + status;
    status = (int)nf_gaussian_f32_apply_v1(
        scanner_mtf_profile, interpreted_workspace, core_height, width,
        gaussian_workspace, post_cloud_workspace_values,
        output_scan_linear_rgb);
    return status == 0 ? 0 : 70 + status;
}
