#define NF_THOMAS_DENSITY_F32_BUILD
#include "nf_thomas_density_f32_v1.h"

#include <math.h>
#include <stdint.h>

static int nf_ranges_overlap(
    const float* first, size_t first_count,
    const float* second, size_t second_count) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const size_t first_bytes = first_count * sizeof(float);
    const size_t second_bytes = second_count * sizeof(float);
    const uintptr_t first_end = first_start + first_bytes;
    const uintptr_t second_end = second_start + second_bytes;
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

uint32_t nf_thomas_density_f32_abi_version_v1(void) {
    return NF_THOMAS_DENSITY_F32_ABI_VERSION_V1;
}

nf_thomas_density_f32_status_v1 nf_thomas_density_f32_workspace_floats_v1(
    size_t height,
    size_t width,
    size_t* workspace_floats) {
    if (workspace_floats == NULL || height == 0u || width == 0u ||
        height > SIZE_MAX / width || height * width > SIZE_MAX / 4u) {
        return NF_THOMAS_DENSITY_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_floats = 4u * height * width;
    return NF_THOMAS_DENSITY_F32_OK_V1;
}

nf_thomas_density_f32_status_v1 nf_thomas_density_f32_apply_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t height,
    size_t width,
    const float* base_density,
    size_t base_density_floats,
    const float* point_density_sigma,
    size_t point_density_sigma_floats,
    float* workspace,
    size_t workspace_floats,
    float* output_transmittance,
    size_t output_floats,
    double* raw_field_mean) {
    size_t count;
    size_t required;
    size_t index;
    float* candidate;
    double local_mean = 0.0;
    nf_thomas_field_f32_status_v1 field_status;

    if (nf_thomas_field_f32_validate_profile_v1(profile) !=
            NF_THOMAS_FIELD_F32_OK_V1) {
        return NF_THOMAS_DENSITY_F32_INVALID_PROFILE_V1;
    }
    if (nf_thomas_density_f32_workspace_floats_v1(
            height, width, &required) != NF_THOMAS_DENSITY_F32_OK_V1 ||
        base_density == NULL || point_density_sigma == NULL ||
        workspace == NULL || output_transmittance == NULL ||
        raw_field_mean == NULL) {
        return NF_THOMAS_DENSITY_F32_INVALID_ARGUMENT_V1;
    }
    count = height * width;
    if (base_density_floats < count || point_density_sigma_floats < count ||
        workspace_floats < required || output_floats < count ||
        nf_ranges_overlap(base_density, count, point_density_sigma, count) ||
        nf_ranges_overlap(base_density, count, workspace, required) ||
        nf_ranges_overlap(base_density, count, output_transmittance, count) ||
        nf_ranges_overlap(point_density_sigma, count, workspace, required) ||
        nf_ranges_overlap(point_density_sigma, count, output_transmittance, count) ||
        nf_ranges_overlap(workspace, required, output_transmittance, count)) {
        return NF_THOMAS_DENSITY_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < count; ++index) {
        if (!isfinite((double)base_density[index]) || base_density[index] < 0.0f ||
            !isfinite((double)point_density_sigma[index]) ||
            point_density_sigma[index] < 0.0f) {
            return NF_THOMAS_DENSITY_F32_DOMAIN_ERROR_V1;
        }
    }

    candidate = workspace + 3u * count;
    field_status = nf_thomas_field_f32_apply_v1(
        profile, height, width, workspace, 3u * count,
        candidate, count, &local_mean);
    if (field_status != NF_THOMAS_FIELD_F32_OK_V1) {
        return field_status == NF_THOMAS_FIELD_F32_INVALID_PROFILE_V1 ?
            NF_THOMAS_DENSITY_F32_INVALID_PROFILE_V1 :
            NF_THOMAS_DENSITY_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < count; ++index) {
        const double density = (double)base_density[index] +
            (double)point_density_sigma[index] * (double)candidate[index];
        const double transmittance = pow(10.0, -density);
        if (!isfinite(density) || density < 0.0 ||
            !isfinite(transmittance) || transmittance <= 0.0 ||
            transmittance > 1.0) {
            return NF_THOMAS_DENSITY_F32_DOMAIN_ERROR_V1;
        }
        candidate[index] = (float)transmittance;
    }
    for (index = 0; index < count; ++index) {
        output_transmittance[index] = candidate[index];
    }
    *raw_field_mean = local_mean;
    return NF_THOMAS_DENSITY_F32_OK_V1;
}
