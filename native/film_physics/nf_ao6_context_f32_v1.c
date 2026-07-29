#define NF_AO6_CONTEXT_F32_BUILD
#include "nf_ao6_context_f32_v1.h"

#include <float.h>
#include <math.h>

static int nf_profile_valid(const nf_ao6_base_f32_profile_v1* profile) {
    size_t channel;
    if (
        profile == NULL ||
        profile->struct_size != sizeof(nf_ao6_base_f32_profile_v1) ||
        profile->abi_version != NF_AO6_BASE_F32_ABI_VERSION_V1 ||
        profile->exposure_floor != (1.0 / 65536.0) ||
        profile->density_strength != 0.5
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite(profile->black_endpoint[channel]) ||
            !isfinite(profile->white_endpoint[channel]) ||
            profile->white_endpoint[channel] <=
                profile->black_endpoint[channel] ||
            !isfinite(profile->negative_midpoints[channel]) ||
            !isfinite(profile->negative_slopes[channel]) ||
            !isfinite(profile->negative_maximum_densities[channel]) ||
            !isfinite(profile->paper_midpoints[channel]) ||
            !isfinite(profile->paper_slopes[channel]) ||
            !isfinite(profile->paper_maximum_densities[channel])
        ) {
            return 0;
        }
    }
    return 1;
}

static int nf_state_valid(
    const nf_ao6_context_f32_state_v1* state) {
    size_t channel;
    if (
        state == NULL ||
        state->struct_size != sizeof(nf_ao6_context_f32_state_v1) ||
        state->abi_version != NF_AO6_CONTEXT_F32_ABI_VERSION_V1
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite(state->mean[channel]) ||
            !isfinite(state->m2[channel]) ||
            state->m2[channel] < 0.0
        ) {
            return 0;
        }
    }
    return 1;
}

static double nf_encoded_to_linear(double value) {
    return value <= 0.04045
        ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

static float nf_linear_to_encoded(float value) {
    return value <= 0.0031308f
        ? 12.92f * value
        : 1.055f * powf(fmaxf(value, 0.0f), 1.0f / 2.4f) - 0.055f;
}

static void nf_raw_reflectance(
    const nf_ao6_base_f32_profile_v1* profile,
    const double linear[3],
    double output[3]) {
    double negative_density[3];
    double transmission[3];
    size_t row;
    size_t column;
    for (row = 0; row < 3; ++row) {
        double exposure = 0.0;
        double argument;
        for (column = 0; column < 3; ++column) {
            exposure += profile->capture_matrix[row][column] *
                linear[column];
        }
        argument = profile->negative_slopes[row] * (
            log2(exposure + profile->exposure_floor) -
            profile->negative_midpoints[row]
        );
        negative_density[row] =
            profile->negative_maximum_densities[row] /
            (1.0 + exp(-argument));
    }
    for (row = 0; row < 3; ++row) {
        double absorption = 0.0;
        for (column = 0; column < 3; ++column) {
            absorption +=
                profile->dye_absorption_matrix[row][column] *
                negative_density[column];
        }
        transmission[row] = pow(10.0, -absorption);
    }
    for (row = 0; row < 3; ++row) {
        double exposure = 0.0;
        double argument;
        double density;
        for (column = 0; column < 3; ++column) {
            exposure += profile->print_matrix[row][column] *
                transmission[column];
        }
        argument = profile->paper_slopes[row] * (
            log2(exposure + profile->exposure_floor) -
            profile->paper_midpoints[row]
        );
        density = profile->paper_maximum_densities[row] /
            (1.0 + exp(-argument));
        output[row] = pow(10.0, -density);
    }
}

static int nf_density_encoded(
    const nf_ao6_base_f32_profile_v1* profile,
    const float input[3],
    float output[3]) {
    double linear[3];
    double raw[3];
    size_t channel;
    for (channel = 0; channel < 3; ++channel) {
        linear[channel] = nf_encoded_to_linear((double)input[channel]);
    }
    nf_raw_reflectance(profile, linear, raw);
    for (channel = 0; channel < 3; ++channel) {
        const double full = (
            raw[channel] - profile->black_endpoint[channel]
        ) / (
            profile->white_endpoint[channel] -
            profile->black_endpoint[channel]
        );
        const double value = linear[channel] +
            profile->density_strength * (full - linear[channel]);
        if (!isfinite(value) || full < -3.0e-12 || full > 1.0 + 3.0e-12) {
            return 0;
        }
        output[channel] = nf_linear_to_encoded((float)value);
    }
    return 1;
}

static float nf_lab_f(float value) {
    return value > 0.008856f
        ? cbrtf(value)
        : 7.787f * value + 16.0f / 116.0f;
}

static void nf_rgb_to_lab(const float rgb[3], float lab[3]) {
    const float r = (float)nf_encoded_to_linear((double)rgb[0]);
    const float g = (float)nf_encoded_to_linear((double)rgb[1]);
    const float b = (float)nf_encoded_to_linear((double)rgb[2]);
    const float x = (
        0.412453f * r + 0.357580f * g + 0.180423f * b
    ) / 0.95047f;
    const float y =
        0.212671f * r + 0.715160f * g + 0.072169f * b;
    const float z = (
        0.019334f * r + 0.119193f * g + 0.950227f * b
    ) / 1.08883f;
    const float fx = nf_lab_f(x);
    const float fy = nf_lab_f(y);
    const float fz = nf_lab_f(z);
    lab[0] = 116.0f * fy - 16.0f;
    lab[1] = 500.0f * (fx - fy);
    lab[2] = 200.0f * (fy - fz);
}

static int nf_lab_for_pixel(
    const nf_ao6_base_f32_profile_v1* profile,
    const float input[3],
    float lab[3]) {
    float density_rgb[3];
    size_t channel;
    if (!nf_density_encoded(profile, input, density_rgb)) {
        return 0;
    }
    nf_rgb_to_lab(density_rgb, lab);
    for (channel = 0; channel < 3; ++channel) {
        if (!isfinite((double)lab[channel])) {
            return 0;
        }
    }
    return 1;
}

uint32_t nf_ao6_context_f32_abi_version_v1(void) {
    return NF_AO6_CONTEXT_F32_ABI_VERSION_V1;
}

nf_ao6_context_f32_status_v1 nf_ao6_context_f32_init_v1(
    nf_ao6_context_f32_state_v1* state) {
    size_t channel;
    if (state == NULL) {
        return NF_AO6_CONTEXT_F32_INVALID_ARGUMENT_V1;
    }
    state->struct_size = sizeof(nf_ao6_context_f32_state_v1);
    state->abi_version = NF_AO6_CONTEXT_F32_ABI_VERSION_V1;
    state->pixel_count = 0u;
    for (channel = 0; channel < 3; ++channel) {
        state->mean[channel] = 0.0;
        state->m2[channel] = 0.0;
    }
    return NF_AO6_CONTEXT_F32_OK_V1;
}

nf_ao6_context_f32_status_v1 nf_ao6_context_f32_update_v1(
    const nf_ao6_base_f32_profile_v1* profile,
    nf_ao6_context_f32_state_v1* state,
    const float* encoded_srgb,
    size_t rgb_count) {
    size_t index;
    size_t channel;
    if (!nf_profile_valid(profile)) {
        return NF_AO6_CONTEXT_F32_INVALID_PROFILE_V1;
    }
    if (
        encoded_srgb == NULL ||
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
        float lab[3];
        if (!nf_lab_for_pixel(
            profile, &encoded_srgb[index * 3u], lab
        )) {
            return NF_AO6_CONTEXT_F32_NUMERIC_FAILURE_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        float lab[3];
        const uint64_t next_count = state->pixel_count + 1u;
        nf_lab_for_pixel(profile, &encoded_srgb[index * 3u], lab);
        for (channel = 0; channel < 3; ++channel) {
            const double delta =
                (double)lab[channel] - state->mean[channel];
            state->mean[channel] += delta / (double)next_count;
            state->m2[channel] += delta * (
                (double)lab[channel] - state->mean[channel]
            );
        }
        state->pixel_count = next_count;
    }
    return NF_AO6_CONTEXT_F32_OK_V1;
}

nf_ao6_context_f32_status_v1 nf_ao6_context_f32_finalize_v1(
    const nf_ao6_context_f32_state_v1* state,
    nf_ao6_base_f32_context_v1* context) {
    size_t channel;
    nf_ao6_base_f32_context_v1 candidate;
    if (context == NULL) {
        return NF_AO6_CONTEXT_F32_INVALID_ARGUMENT_V1;
    }
    if (!nf_state_valid(state) || state->pixel_count == 0u) {
        return NF_AO6_CONTEXT_F32_INVALID_STATE_V1;
    }
    candidate.struct_size = sizeof(nf_ao6_base_f32_context_v1);
    candidate.abi_version = NF_AO6_BASE_F32_ABI_VERSION_V1;
    candidate.pixel_count = state->pixel_count;
    for (channel = 0; channel < 3; ++channel) {
        const double variance =
            state->m2[channel] / (double)state->pixel_count;
        const double standard_deviation = sqrt(fmax(variance, 0.0));
        candidate.source_mean[channel] = (float)state->mean[channel];
        candidate.source_std[channel] = (float)fmax(
            standard_deviation, 1.0e-3
        );
        if (
            !isfinite((double)candidate.source_mean[channel]) ||
            !isfinite((double)candidate.source_std[channel])
        ) {
            return NF_AO6_CONTEXT_F32_NUMERIC_FAILURE_V1;
        }
    }
    *context = candidate;
    return NF_AO6_CONTEXT_F32_OK_V1;
}
