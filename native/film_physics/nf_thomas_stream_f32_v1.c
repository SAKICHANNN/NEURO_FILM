#define NF_THOMAS_STREAM_F32_BUILD
#define NF_THOMAS_ROWS_F32_BUILD
#define NF_THOMAS_FIELD_F32_BUILD
#define NF_NEUMAIER_F32_BUILD
#include "nf_thomas_stream_f32_v1.h"
#include "nf_neumaier_f32_v1.h"

#include <math.h>
#include <stdint.h>

static int nf_ranges_overlap(
    const float* first,
    size_t first_count,
    const float* second,
    size_t second_count) {
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

uint32_t nf_thomas_stream_f32_abi_version_v1(void) {
    return NF_THOMAS_STREAM_F32_ABI_VERSION_V1;
}

nf_thomas_stream_f32_status_v1 nf_thomas_stream_f32_workspace_floats_v1(
    size_t width,
    size_t row_partition,
    size_t* workspace_floats) {
    size_t row_workspace;
    size_t output_floats;
    if (workspace_floats == NULL || width == 0u || row_partition == 0u ||
        row_partition > SIZE_MAX / width ||
        nf_thomas_rows_f32_workspace_floats_v1(
            width, row_partition, &row_workspace) != NF_THOMAS_ROWS_F32_OK_V1) {
        return NF_THOMAS_STREAM_F32_INVALID_ARGUMENT_V1;
    }
    output_floats = row_partition * width;
    if (row_workspace > SIZE_MAX - output_floats) {
        return NF_THOMAS_STREAM_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_floats = row_workspace + output_floats;
    return NF_THOMAS_STREAM_F32_OK_V1;
}

nf_thomas_stream_f32_status_v1 nf_thomas_stream_f32_density_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    const float* base_density,
    size_t base_density_floats,
    const float* point_density_sigma,
    size_t point_density_sigma_floats,
    float* workspace,
    size_t workspace_floats,
    nf_thomas_stream_f32_sink_v1 sink,
    void* sink_context,
    double* raw_field_mean) {
    size_t full_count;
    size_t required;
    size_t row_workspace;
    size_t row_start;
    size_t index;
    double total = 0.0;
    double compensation = 0.0;
    double mean;
    float* output;
    if (nf_thomas_field_f32_validate_profile_v1(profile) !=
            NF_THOMAS_FIELD_F32_OK_V1) {
        return NF_THOMAS_STREAM_F32_INVALID_PROFILE_V1;
    }
    if (full_height == 0u || width == 0u || row_partition == 0u ||
        full_height > SIZE_MAX / width || base_density == NULL ||
        point_density_sigma == NULL || workspace == NULL || sink == NULL ||
        raw_field_mean == NULL ||
        nf_thomas_stream_f32_workspace_floats_v1(
            width, row_partition, &required) != NF_THOMAS_STREAM_F32_OK_V1 ||
        nf_thomas_rows_f32_workspace_floats_v1(
            width, row_partition, &row_workspace) != NF_THOMAS_ROWS_F32_OK_V1) {
        return NF_THOMAS_STREAM_F32_INVALID_ARGUMENT_V1;
    }
    full_count = full_height * width;
    if (base_density_floats < full_count ||
        point_density_sigma_floats < full_count || workspace_floats < required ||
        nf_ranges_overlap(base_density, full_count, point_density_sigma, full_count) ||
        nf_ranges_overlap(base_density, full_count, workspace, required) ||
        nf_ranges_overlap(point_density_sigma, full_count, workspace, required)) {
        return NF_THOMAS_STREAM_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0u; index < full_count; ++index) {
        if (!isfinite((double)base_density[index]) || base_density[index] < 0.0f ||
            !isfinite((double)point_density_sigma[index]) ||
            point_density_sigma[index] < 0.0f) {
            return NF_THOMAS_STREAM_F32_DOMAIN_ERROR_V1;
        }
    }
    output = workspace + row_workspace;
    for (row_start = 0u; row_start < full_height; row_start += row_partition) {
        const size_t row_count = row_partition < full_height - row_start ?
            row_partition : full_height - row_start;
        const size_t count = row_count * width;
        const nf_thomas_rows_f32_status_v1 status = nf_thomas_rows_f32_field_v1(
            profile, full_height, width, row_start, row_count,
            workspace, row_workspace, output, row_partition * width);
        if (status != NF_THOMAS_ROWS_F32_OK_V1 ||
            nf_neumaier_f32_accumulate_v1(
                output, count, &total, &compensation) != 1) {
            return status == NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1 ?
                NF_THOMAS_STREAM_F32_INVALID_PROFILE_V1 :
                NF_THOMAS_STREAM_F32_DOMAIN_ERROR_V1;
        }
    }
    mean = (total + compensation) / (double)full_count;
    if (!isfinite(mean)) {
        return NF_THOMAS_STREAM_F32_DOMAIN_ERROR_V1;
    }
    for (row_start = 0u; row_start < full_height; row_start += row_partition) {
        const size_t row_count = row_partition < full_height - row_start ?
            row_partition : full_height - row_start;
        const size_t offset = row_start * width;
        const size_t count = row_count * width;
        const nf_thomas_rows_f32_status_v1 status = nf_thomas_rows_f32_density_v1(
            profile, full_height, width, row_start, row_count,
            base_density + offset, count, point_density_sigma + offset, count,
            mean, workspace, row_workspace, output, row_partition * width);
        if (status != NF_THOMAS_ROWS_F32_OK_V1) {
            return status == NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1 ?
                NF_THOMAS_STREAM_F32_INVALID_PROFILE_V1 :
                status == NF_THOMAS_ROWS_F32_DOMAIN_ERROR_V1 ?
                    NF_THOMAS_STREAM_F32_DOMAIN_ERROR_V1 :
                    NF_THOMAS_STREAM_F32_INVALID_ARGUMENT_V1;
        }
        if (sink(sink_context, row_start, row_count, output, count) == 0) {
            return NF_THOMAS_STREAM_F32_CALLBACK_FAILED_V1;
        }
    }
    *raw_field_mean = mean;
    return NF_THOMAS_STREAM_F32_OK_V1;
}
