#define NF_AO6_CONTEXT_F32_BUILD
#define NF_AO6_CONTEXT_F32_V2_BUILD
#include "nf_ao6_context_f32_v2.h"

/* Keep the frozen v1 implementation and ABI available in this DLL. */
#include "nf_ao6_context_f32_v1.c"

nf_ao6_context_f32_status_v1 nf_ao6_context_f32_update_v2(
    const nf_ao6_base_f32_profile_v1* profile,
    nf_ao6_context_f32_state_v1* state,
    const float* encoded_srgb,
    size_t rgb_count,
    float* scratch_lab) {
    size_t index;
    size_t channel;
    if (!nf_profile_valid(profile)) {
        return NF_AO6_CONTEXT_F32_INVALID_PROFILE_V1;
    }
    if (
        encoded_srgb == NULL ||
        scratch_lab == NULL ||
        scratch_lab == encoded_srgb ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float)) ||
        !nf_state_valid(state) ||
        state->pixel_count > UINT64_MAX - (uint64_t)rgb_count
    ) {
        return NF_AO6_CONTEXT_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        if (
            !isfinite((double)encoded_srgb[index]) ||
            encoded_srgb[index] < 0.0f ||
            encoded_srgb[index] > 1.0f
        ) {
            return NF_AO6_CONTEXT_F32_INVALID_INPUT_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        if (!nf_lab_for_pixel(
            profile,
            &encoded_srgb[index * 3u],
            &scratch_lab[index * 3u]
        )) {
            return NF_AO6_CONTEXT_F32_NUMERIC_FAILURE_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        const uint64_t next_count = state->pixel_count + 1u;
        for (channel = 0; channel < 3; ++channel) {
            const double sample =
                (double)scratch_lab[index * 3u + channel];
            const double delta = sample - state->mean[channel];
            state->mean[channel] += delta / (double)next_count;
            state->m2[channel] += delta * (
                sample - state->mean[channel]
            );
        }
        state->pixel_count = next_count;
    }
    return NF_AO6_CONTEXT_F32_OK_V1;
}
