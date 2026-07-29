#define NF_AO6_RESIDUAL_F32_BUILD
#define NF_AO6_RESIDUAL_F32_V2_BUILD
#include "nf_ao6_residual_f32_v2.h"

/* Keep the frozen v1 implementation and ABI available in this DLL. */
#include "nf_ao6_residual_f32_v1.c"

nf_ao6_residual_f32_status_v1
nf_ao6_residual_f32_apply_scratch_v2(
    const nf_ao6_residual_f32_profile_v1* profile,
    const float* base_linear_rgb,
    size_t rgb_count,
    float* output_linear_rgb,
    float* tone_scale,
    float* chroma_scale) {
    size_t index;
    const nf_ao6_residual_f32_status_v1 profile_status =
        nf_ao6_residual_f32_validate_profile_v1(profile);
    if (profile_status != NF_AO6_RESIDUAL_F32_OK_V1) {
        return profile_status;
    }
    if (
        base_linear_rgb == NULL ||
        output_linear_rgb == NULL ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float))
    ) {
        return NF_AO6_RESIDUAL_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        if (
            !isfinite((double)base_linear_rgb[index]) ||
            base_linear_rgb[index] < 0.0f ||
            base_linear_rgb[index] > 1.0f
        ) {
            return NF_AO6_RESIDUAL_F32_INVALID_INPUT_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        if (!nf_apply_pixel(
            profile,
            &base_linear_rgb[index * 3u],
            &output_linear_rgb[index * 3u],
            tone_scale == NULL ? NULL : &tone_scale[index],
            chroma_scale == NULL ? NULL : &chroma_scale[index]
        )) {
            return NF_AO6_RESIDUAL_F32_NUMERIC_FAILURE_V1;
        }
    }
    return NF_AO6_RESIDUAL_F32_OK_V1;
}
