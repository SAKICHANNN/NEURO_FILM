#define NF_AO6_BASE_F32_BUILD
#define NF_AO6_BASE_F32_V2_BUILD
#define NF_AO6_BASE_F32_V3_BUILD
#include "nf_ao6_base_f32_v3.h"

/* Keep the frozen v1/v2 implementation and ABI available in this DLL. */
#include "nf_ao6_base_f32_v2.c"

nf_ao6_base_f32_status_v1 nf_ao6_base_f32_apply_scratch_v3(
    const nf_ao6_base_f32_profile_v1* profile,
    const nf_ao6_base_f32_context_v1* context,
    const float* encoded_srgb,
    size_t rgb_count,
    float* output_encoded_srgb) {
    size_t index;
    const nf_ao6_base_f32_status_v1 profile_status =
        nf_ao6_base_f32_validate_profile_v1(profile);
    if (profile_status != NF_AO6_BASE_F32_OK_V1) {
        return profile_status;
    }
    if (
        encoded_srgb == NULL ||
        output_encoded_srgb == NULL ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float))
    ) {
        return NF_AO6_BASE_F32_INVALID_ARGUMENT_V1;
    }
    if (!nf_context_valid(context)) {
        return NF_AO6_BASE_F32_INVALID_CONTEXT_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        if (
            !isfinite((double)encoded_srgb[index]) ||
            encoded_srgb[index] < 0.0f ||
            encoded_srgb[index] > 1.0f
        ) {
            return NF_AO6_BASE_F32_INVALID_INPUT_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        if (!nf_apply_pixel_v2(
            profile,
            context,
            &encoded_srgb[index * 3u],
            &output_encoded_srgb[index * 3u]
        )) {
            return NF_AO6_BASE_F32_NUMERIC_FAILURE_V1;
        }
    }
    return NF_AO6_BASE_F32_OK_V1;
}
