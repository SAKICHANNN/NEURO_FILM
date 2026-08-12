#define NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_BUILD
#include "nf_conditioned_cloud_row_chain_f32_v1.h"

#include "nf_cloud_attenuation_f32_v1.h"

#include <math.h>

static int nf_mul_size(size_t first, size_t second, size_t* result) {
    if (first != 0u && second > SIZE_MAX / first) return 0;
    *result = first * second;
    return 1;
}

uint32_t nf_conditioned_cloud_row_chain_f32_abi_version_v1(void) {
    return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_ABI_VERSION_V1;
}

nf_conditioned_cloud_row_chain_f32_status_v1
nf_conditioned_cloud_row_chain_f32_workspace_v1(
    size_t core_height, size_t width, size_t halo,
    size_t* count_values, size_t* convolution_doubles,
    size_t* core_float_values) {
    size_t extended_height;
    size_t extended_pixels;
    size_t core_pixels;
    if (count_values == NULL || convolution_doubles == NULL ||
        core_float_values == NULL || core_height == 0u || width == 0u ||
        halo > SIZE_MAX / 2u || core_height > SIZE_MAX - 2u * halo) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_INVALID_ARGUMENT_V1;
    }
    extended_height = core_height + 2u * halo;
    if (!nf_mul_size(extended_height, width, &extended_pixels) ||
        !nf_mul_size(core_height, width, &core_pixels) ||
        !nf_mul_size(extended_pixels, 3u, count_values) ||
        !nf_mul_size(core_pixels, 3u, core_float_values)) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_INVALID_ARGUMENT_V1;
    }
    *convolution_doubles = extended_pixels;
    return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1;
}

nf_conditioned_cloud_row_chain_f32_status_v1
nf_conditioned_cloud_row_chain_f32_apply_v1(
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
    size_t required_counts;
    size_t required_doubles;
    size_t required_core;
    size_t full_pixels;
    size_t cursor = 0u;
    size_t first_logical;
    const size_t extended_height = core_height + 2u * halo;
    if (nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core_height, width, halo, &required_counts, &required_doubles,
            &required_core) != NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1 ||
        count_profile == NULL || spatial_profile == NULL ||
        full_scale_cmy == NULL || expected_transmittance_cmy == NULL ||
        channel_gain_cmy == NULL || count_workspace == NULL ||
        convolution_workspace == NULL || spatial_density_workspace == NULL ||
        spatial_transmittance_workspace == NULL || output_density_cmy == NULL ||
        output_transmittance_cmy == NULL || full_height == 0u ||
        origin_y >= full_height || core_height > full_height ||
        !nf_mul_size(full_height, width, &full_pixels) ||
        full_pixels > SIZE_MAX / 3u || scale_values < 3u * full_pixels ||
        expected_values < required_core || output_values < required_core ||
        count_workspace_values < required_counts ||
        convolution_workspace_doubles < required_doubles ||
        spatial_workspace_values < required_core) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_INVALID_ARGUMENT_V1;
    }
    first_logical = halo % full_height;
    first_logical = origin_y >= first_logical
        ? origin_y - first_logical
        : full_height - (first_logical - origin_y);
    while (cursor < extended_height) {
        const size_t offset = cursor % full_height;
        const size_t logical = offset < full_height - first_logical
            ? first_logical + offset
            : offset - (full_height - first_logical);
        size_t run = extended_height - cursor;
        nf_density_conditioned_poisson_u16_status_v3 status;
        if (run > full_height - logical) run = full_height - logical;
        status = nf_density_conditioned_poisson_u16_sample_region_v3(
            count_profile, full_height, width, logical, 0u, run, width,
            full_scale_cmy + logical * width * 3u,
            run * width * 3u,
            count_workspace + cursor * width * 3u,
            run * width * 3u);
        if (status != NF_DENSITY_CONDITIONED_POISSON_U16_OK_V3) {
            return status == NF_DENSITY_CONDITIONED_POISSON_U16_DOMAIN_ERROR_V3
                ? NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1
                : NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_INVALID_ARGUMENT_V1;
        }
        cursor += run;
    }
    if (nf_cloud_spatial_response_f32_apply_v2(
            spatial_profile, count_workspace, core_height, width, halo,
            convolution_workspace, convolution_workspace_doubles,
            spatial_density_workspace, spatial_transmittance_workspace,
            spatial_workspace_values) != NF_CLOUD_SPATIAL_RESPONSE_F32_OK_V2) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
    }
    for (size_t index = 0u; index < required_core; ++index) {
        spatial_transmittance_workspace[index] = (float)pow(
            10.0, -(double)spatial_density_workspace[index]);
    }
    if (nf_cloud_attenuation_f32_apply_v1(
            expected_transmittance_cmy, spatial_transmittance_workspace,
            required_core / 3u, channel_gain_cmy, output_density_cmy,
            output_transmittance_cmy) != NF_CLOUD_ATTENUATION_F32_OK_V1) {
        return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_DOMAIN_ERROR_V1;
    }
    return NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_OK_V1;
}
