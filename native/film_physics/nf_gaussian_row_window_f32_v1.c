#define NF_GAUSSIAN_ROW_WINDOW_F32_BUILD
#include "nf_gaussian_row_window_f32_v1.h"

uint32_t nf_gaussian_row_window_f32_abi_version_v1(void) {
    return NF_GAUSSIAN_ROW_WINDOW_F32_ABI_VERSION_V1;
}

nf_gaussian_f32_status_v1 nf_gaussian_row_window_f32_apply_v1(
    const nf_gaussian_f32_profile_v1* profile,
    size_t full_height,
    size_t full_width,
    size_t input_logical_start,
    size_t input_height,
    const float* input_rgb,
    size_t core_logical_start,
    size_t core_height,
    float* workspace_rgb,
    size_t workspace_floats,
    float* window_output_rgb,
    size_t window_output_floats,
    float* core_output_rgb,
    size_t core_output_floats) {
    uint32_t halo = 0u;
    size_t input_values;
    size_t core_values;
    size_t core_offset;
    size_t index;
    nf_gaussian_f32_status_v1 status;
    if (
        profile == NULL || input_rgb == NULL || workspace_rgb == NULL ||
        window_output_rgb == NULL || core_output_rgb == NULL ||
        full_height == 0u || full_width == 0u || input_height == 0u ||
        core_height == 0u || input_logical_start > full_height ||
        input_height > full_height - input_logical_start ||
        core_logical_start < input_logical_start ||
        core_logical_start > full_height ||
        core_height > full_height - core_logical_start ||
        core_height > input_height - (core_logical_start - input_logical_start) ||
        input_height > SIZE_MAX / full_width ||
        input_height * full_width > SIZE_MAX / 3u ||
        core_height > SIZE_MAX / full_width ||
        core_height * full_width > SIZE_MAX / 3u
    ) {
        return NF_GAUSSIAN_F32_INVALID_ARGUMENT_V1;
    }
    status = nf_gaussian_f32_required_halo_v1(profile, &halo);
    if (status != NF_GAUSSIAN_F32_OK_V1) {
        return status;
    }
    core_offset = core_logical_start - input_logical_start;
    if (
        core_offset != (core_logical_start < (size_t)halo
            ? core_logical_start : (size_t)halo) ||
        input_height - (core_offset + core_height) !=
            (full_height - (core_logical_start + core_height) < (size_t)halo
                ? full_height - (core_logical_start + core_height)
                : (size_t)halo)
    ) {
        return NF_GAUSSIAN_F32_INVALID_ARGUMENT_V1;
    }
    input_values = input_height * full_width * 3u;
    core_values = core_height * full_width * 3u;
    if (
        workspace_floats < input_values ||
        window_output_floats < input_values ||
        core_output_floats < core_values
    ) {
        return NF_GAUSSIAN_F32_INVALID_ARGUMENT_V1;
    }
    status = nf_gaussian_f32_apply_v1(
        profile, input_rgb, input_height, full_width, workspace_rgb,
        workspace_floats, window_output_rgb);
    if (status != NF_GAUSSIAN_F32_OK_V1) {
        return status;
    }
    core_offset *= full_width * 3u;
    for (index = 0u; index < core_values; ++index) {
        core_output_rgb[index] = window_output_rgb[core_offset + index];
    }
    return NF_GAUSSIAN_F32_OK_V1;
}
