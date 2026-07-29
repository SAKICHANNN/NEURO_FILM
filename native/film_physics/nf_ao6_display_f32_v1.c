#define NF_AO6_DISPLAY_F32_BUILD
#include "nf_ao6_display_f32_v1.h"

#include <math.h>

static float nf_encoded_to_linear(float value) {
    return value <= 0.04045f
        ? (float)((double)value / 12.92)
        : (float)pow(
            ((double)value + 0.055) / 1.055,
            2.4
        );
}

static float nf_linear_to_encoded(float value) {
    return value <= 0.0031308f
        ? (float)(12.92 * (double)value)
        : (float)(
            1.055 * pow(fmax((double)value, 0.0), 1.0 / 2.4) -
            0.055
        );
}

uint32_t nf_ao6_display_f32_abi_version_v1(void) {
    return NF_AO6_DISPLAY_F32_ABI_VERSION_V1;
}

nf_ao6_display_f32_status_v1 nf_ao6_display_f32_apply_v1(
    const nf_ao6_base_f32_profile_v1* base_profile,
    const nf_ao6_base_f32_context_v1* context,
    const nf_ao6_residual_f32_profile_v1* residual_profile,
    const float* input_encoded_srgb,
    size_t rgb_count,
    float* scratch_rgb,
    float* output_encoded_srgb) {
    size_t index;
    nf_ao6_base_f32_status_v1 base_status;
    nf_ao6_residual_f32_status_v1 residual_status;
    if (
        base_profile == NULL ||
        context == NULL ||
        residual_profile == NULL ||
        input_encoded_srgb == NULL ||
        scratch_rgb == NULL ||
        output_encoded_srgb == NULL ||
        scratch_rgb == input_encoded_srgb ||
        scratch_rgb == output_encoded_srgb ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float))
    ) {
        return NF_AO6_DISPLAY_F32_INVALID_ARGUMENT_V1;
    }
    base_status = nf_ao6_base_f32_apply_v1(
        base_profile,
        context,
        input_encoded_srgb,
        rgb_count,
        scratch_rgb
    );
    if (base_status != NF_AO6_BASE_F32_OK_V1) {
        return NF_AO6_DISPLAY_F32_BASE_FAILURE_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        scratch_rgb[index] = nf_encoded_to_linear(scratch_rgb[index]);
        if (
            !isfinite((double)scratch_rgb[index]) ||
            scratch_rgb[index] < 0.0f ||
            scratch_rgb[index] > 1.0f
        ) {
            return NF_AO6_DISPLAY_F32_NUMERIC_FAILURE_V1;
        }
    }
    residual_status = nf_ao6_residual_f32_apply_v1(
        residual_profile,
        scratch_rgb,
        rgb_count,
        scratch_rgb,
        NULL,
        NULL
    );
    if (residual_status != NF_AO6_RESIDUAL_F32_OK_V1) {
        return NF_AO6_DISPLAY_F32_RESIDUAL_FAILURE_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        const float value = nf_linear_to_encoded(scratch_rgb[index]);
        if (!isfinite((double)value) || value < 0.0f || value > 1.0f) {
            return NF_AO6_DISPLAY_F32_NUMERIC_FAILURE_V1;
        }
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        output_encoded_srgb[index] =
            nf_linear_to_encoded(scratch_rgb[index]);
    }
    return NF_AO6_DISPLAY_F32_OK_V1;
}
