#define NF_AO6_BASE_F32_BUILD
#define NF_AO6_BASE_F32_V2_BUILD
#include "nf_ao6_base_f32_v2.h"

/* Keep the frozen v1 implementation and ABI available in this DLL. */
#include "nf_ao6_base_f32_v1.c"

static int nf_apply_pixel_v2(
    const nf_ao6_base_f32_profile_v1* profile,
    const nf_ao6_base_f32_context_v1* context,
    const float input[3],
    float output[3]) {
    float density_rgb[3];
    float source_lab[3];
    float target_lab[3];
    float candidate[3];
    float linear[3];
    float low = 0.0f;
    float high = 1.0f;
    size_t channel;
    uint32_t iteration;
    if (!nf_density_encoded(profile, input, density_rgb)) {
        return 0;
    }
    nf_rgb_to_lab(density_rgb, source_lab);
    for (channel = 0; channel < 3; ++channel) {
        const float transferred = (
            (source_lab[channel] - context->source_mean[channel]) /
            context->source_std[channel]
        ) * profile->destination_std[channel] +
            profile->destination_mean[channel];
        const float strength = channel == 0
            ? profile->style_strength * profile->luma_strength
            : profile->style_strength;
        target_lab[channel] = source_lab[channel] +
            strength * (transferred - source_lab[channel]);
    }
    if (nf_lab_in_gamut(target_lab)) {
        /*
         * The legacy loop never returns 1.0 for an always-valid target.
         * Resolve its exact float32 low endpoint without 14 Lab conversions.
         */
        for (iteration = 0;
             iteration < profile->gamut_iterations;
             ++iteration) {
            low = (low + high) * 0.5f;
        }
    } else {
        for (iteration = 0;
             iteration < profile->gamut_iterations;
             ++iteration) {
            const float mid = (low + high) * 0.5f;
            candidate[0] = target_lab[0];
            candidate[1] = target_lab[1] * mid;
            candidate[2] = target_lab[2] * mid;
            if (nf_lab_in_gamut(candidate)) {
                low = mid;
            } else {
                high = mid;
            }
        }
    }
    candidate[0] = target_lab[0];
    candidate[1] = target_lab[1] * low;
    candidate[2] = target_lab[2] * low;
    nf_lab_to_linear(candidate, linear);
    for (channel = 0; channel < 3; ++channel) {
        float value;
        const float margin =
            (float)profile->output_margin_8bit / 255.0f;
        if (!isfinite((double)linear[channel])) {
            return 0;
        }
        value = nf_linear_to_encoded(
            fminf(1.0f, fmaxf(0.0f, linear[channel]))
        );
        if (value < margin) {
            value = margin;
        } else if (value > 1.0f - margin) {
            value = 1.0f - margin;
        }
        if (!isfinite((double)value)) {
            return 0;
        }
        if (output != NULL) {
            output[channel] = value;
        }
    }
    return 1;
}

nf_ao6_base_f32_status_v1 nf_ao6_base_f32_apply_v2(
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
            profile, context, &encoded_srgb[index * 3u], NULL
        )) {
            return NF_AO6_BASE_F32_NUMERIC_FAILURE_V1;
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
