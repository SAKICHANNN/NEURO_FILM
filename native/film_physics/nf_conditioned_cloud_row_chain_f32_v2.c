#define NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_BUILD
#include "nf_conditioned_cloud_row_chain_f32_v2.h"

#include "nf_cloud_attenuation_f32_v1.h"
#include "nf_deterministic_log10_f32_v1.h"

#include <math.h>

uint32_t nf_conditioned_cloud_row_chain_f32_abi_version_v2(void) {
    return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_ABI_VERSION_V2;
}

nf_conditioned_cloud_row_chain_f32_status_v1
nf_conditioned_cloud_row_chain_f32_apply_v2(
    const nf_density_conditioned_poisson_u16_profile_v3* count_profile,
    const nf_cloud_spatial_response_f32_profile_v2* spatial_profile,
    size_t full_height, size_t width, size_t origin_y, size_t core_height,
    size_t halo, const double* full_scale_cmy, size_t scale_values,
    const float* expected_transmittance_cmy, size_t expected_values,
    const float channel_gain_cmy[3], uint16_t* count_workspace,
    size_t count_workspace_values, double* convolution_workspace,
    size_t convolution_workspace_doubles, float* spatial_density_workspace,
    float* spatial_transmittance_workspace, size_t spatial_workspace_values,
    float* output_density_cmy, float* output_transmittance_cmy,
    size_t output_values) {
    size_t required_counts, required_doubles, required_core;
    nf_conditioned_cloud_row_chain_f32_status_v1 status;
    if (nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core_height, width, halo, &required_counts, &required_doubles,
            &required_core) != NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1 ||
        output_values < required_core) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_INVALID_ARGUMENT_V1;
    }
    status = nf_conditioned_cloud_row_chain_f32_apply_v1(
        count_profile, spatial_profile, full_height, width, origin_y,
        core_height, halo, full_scale_cmy, scale_values,
        expected_transmittance_cmy, expected_values, channel_gain_cmy,
        count_workspace, count_workspace_values, convolution_workspace,
        convolution_workspace_doubles, spatial_density_workspace,
        spatial_transmittance_workspace, spatial_workspace_values,
        output_density_cmy, output_transmittance_cmy, output_values);
    if (status != NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1) return status;
    if (nf_deterministic_neg_log10_f32_apply_v1(
            output_transmittance_cmy, required_core, output_density_cmy) !=
        NF_DETERMINISTIC_LOG10_F32_OK_V1) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
    }
    return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1;
}

nf_conditioned_cloud_row_chain_f32_status_v1
nf_conditioned_cloud_row_chain_f32_apply_window_v3(
    const nf_density_conditioned_poisson_u16_profile_v3* count_profile,
    const nf_cloud_spatial_response_f32_profile_v2* spatial_profile,
    size_t full_height, size_t width, size_t first_logical_y,
    size_t core_height, size_t halo, const double* scale_window_cmy,
    size_t scale_values, const float* expected_transmittance_cmy,
    size_t expected_values, const float channel_gain_cmy[3],
    uint16_t* count_workspace, size_t count_workspace_values,
    double* convolution_workspace, size_t convolution_workspace_doubles,
    float* spatial_density_workspace, float* spatial_transmittance_workspace,
    size_t spatial_workspace_values, float* output_density_cmy,
    float* output_transmittance_cmy, size_t output_values) {
    size_t required_counts, required_doubles, required_core;
    size_t extended_height, cursor = 0u;
    if (nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core_height, width, halo, &required_counts, &required_doubles,
            &required_core) != NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1 ||
        full_height == 0u || first_logical_y >= full_height ||
        scale_window_cmy == NULL || scale_values < required_counts ||
        count_workspace_values < required_counts ||
        convolution_workspace_doubles < required_doubles ||
        spatial_workspace_values < required_core || expected_values < required_core ||
        output_values < required_core) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_INVALID_ARGUMENT_V1;
    }
    extended_height = core_height + 2u * halo;
    while (cursor < extended_height) {
        const size_t offset = cursor % full_height;
        const size_t logical = offset < full_height - first_logical_y
            ? first_logical_y + offset : offset - (full_height - first_logical_y);
        size_t run = extended_height - cursor;
        nf_density_conditioned_poisson_u16_status_v3 status;
        if (run > full_height - logical) run = full_height - logical;
        status = nf_density_conditioned_poisson_u16_sample_region_v3(
            count_profile, full_height, width, logical, 0u, run, width,
            scale_window_cmy + cursor * width * 3u, run * width * 3u,
            count_workspace + cursor * width * 3u, run * width * 3u);
        if (status != NF_DENSITY_CONDITIONED_POISSON_U16_OK_V3)
            return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
        cursor += run;
    }
    if (nf_cloud_spatial_response_f32_apply_v2(
            spatial_profile, count_workspace, core_height, width, halo,
            convolution_workspace, convolution_workspace_doubles,
            spatial_density_workspace, spatial_transmittance_workspace,
            spatial_workspace_values) != NF_CLOUD_SPATIAL_RESPONSE_F32_OK_V2)
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
    for (size_t index=0u; index<required_core; ++index)
        spatial_transmittance_workspace[index]=(float)pow(10.0,-(double)spatial_density_workspace[index]);
    if (nf_cloud_attenuation_f32_apply_v1(
            expected_transmittance_cmy, spatial_transmittance_workspace,
            required_core/3u, channel_gain_cmy, output_density_cmy,
            output_transmittance_cmy) != NF_CLOUD_ATTENUATION_F32_OK_V1)
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
    if (nf_deterministic_neg_log10_f32_apply_v1(
            output_transmittance_cmy, required_core, output_density_cmy) !=
        NF_DETERMINISTIC_LOG10_F32_OK_V1)
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
    return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1;
}
