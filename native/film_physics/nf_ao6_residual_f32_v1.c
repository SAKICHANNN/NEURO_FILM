#define NF_AO6_RESIDUAL_F32_BUILD
#include "nf_ao6_residual_f32_v1.h"

#include <float.h>
#include <math.h>

#define NF_AO6_F32_TOLERANCE_V1 2.0e-6

static int nf_is_hex_sha256(const char value[65]) {
    size_t index;
    if (value[64] != '\0') {
        return 0;
    }
    for (index = 0; index < 64; ++index) {
        const char c = value[index];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

static double nf_det3(const double matrix[3][3]) {
    return
        matrix[0][0] * (
            matrix[1][1] * matrix[2][2] -
            matrix[1][2] * matrix[2][1]) -
        matrix[0][1] * (
            matrix[1][0] * matrix[2][2] -
            matrix[1][2] * matrix[2][0]) +
        matrix[0][2] * (
            matrix[1][0] * matrix[2][1] -
            matrix[1][1] * matrix[2][0]);
}

static int nf_matrix_valid(
    const double matrix[3][3],
    double determinant_floor) {
    size_t row;
    size_t column;
    for (row = 0; row < 3; ++row) {
        double sum = 0.0;
        for (column = 0; column < 3; ++column) {
            const double value = matrix[row][column];
            if (!isfinite(value) || value < 0.0) {
                return 0;
            }
            sum += value;
        }
        if (fabs(sum - 1.0) > 1.0e-12) {
            return 0;
        }
    }
    return nf_det3(matrix) >= determinant_floor;
}

static int nf_profile_valid(
    const nf_ao6_residual_f32_profile_v1* profile) {
    size_t channel;
    double luma_sum = 0.0;
    if (
        profile == NULL ||
        profile->struct_size !=
            sizeof(nf_ao6_residual_f32_profile_v1) ||
        profile->abi_version != NF_AO6_RESIDUAL_F32_ABI_VERSION_V1 ||
        !nf_is_hex_sha256(profile->source_component_sha256) ||
        profile->exposure_floor != (1.0 / 65536.0) ||
        !isfinite(profile->matrix_minimum_determinant) ||
        profile->matrix_minimum_determinant <= 0.0 ||
        profile->matrix_minimum_determinant >= 1.0 ||
        !isfinite(profile->minimum_endpoint_span) ||
        profile->minimum_endpoint_span <= 0.0 ||
        !nf_matrix_valid(
            profile->capture_matrix,
            profile->matrix_minimum_determinant) ||
        !nf_matrix_valid(
            profile->scan_matrix,
            profile->matrix_minimum_determinant) ||
        profile->tone_strength != 0.15 ||
        profile->chroma_strength != 0.35 ||
        !isfinite(profile->hard_low_linear) ||
        !isfinite(profile->hard_high_linear) ||
        !isfinite(profile->guard_low_linear) ||
        !isfinite(profile->guard_high_linear) ||
        profile->hard_low_linear < 0.0 ||
        profile->hard_high_linear > 1.0 ||
        profile->guard_low_linear <= profile->hard_low_linear ||
        profile->guard_high_linear >= profile->hard_high_linear ||
        profile->guard_low_linear >= profile->guard_high_linear
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite(profile->response_midpoints[channel]) ||
            profile->response_midpoints[channel] < -12.0 ||
            profile->response_midpoints[channel] > 2.0 ||
            !isfinite(profile->response_slopes[channel]) ||
            profile->response_slopes[channel] < 0.2 ||
            profile->response_slopes[channel] > 4.0 ||
            !isfinite(profile->maximum_responses[channel]) ||
            profile->maximum_responses[channel] < 0.2 ||
            profile->maximum_responses[channel] > 4.0 ||
            !isfinite(profile->black_endpoint[channel]) ||
            !isfinite(profile->white_endpoint[channel]) ||
            profile->white_endpoint[channel] -
                profile->black_endpoint[channel] <
                profile->minimum_endpoint_span ||
            !isfinite(profile->luma_weights[channel]) ||
            profile->luma_weights[channel] <= 0.0
        ) {
            return 0;
        }
        luma_sum += profile->luma_weights[channel];
    }
    return fabs(luma_sum - 1.0) <= 1.0e-12;
}

static double nf_safe_scale(
    const double base[3],
    const double delta[3],
    const double lower[3],
    const double upper[3]) {
    double scale = 1.0;
    size_t channel;
    for (channel = 0; channel < 3; ++channel) {
        double limit = INFINITY;
        if (delta[channel] > 0.0) {
            limit = (upper[channel] - base[channel]) /
                fmax(delta[channel], DBL_MIN);
        } else if (delta[channel] < 0.0) {
            limit = (lower[channel] - base[channel]) /
                fmin(delta[channel], -DBL_MIN);
        }
        if (limit < scale) {
            scale = limit;
        }
    }
    if (scale < 0.0) {
        return 0.0;
    }
    if (scale > 1.0) {
        return 1.0;
    }
    return scale;
}

static void nf_raw_response(
    const nf_ao6_residual_f32_profile_v1* profile,
    const double source[3],
    double output[3]) {
    double layer_response[3];
    size_t row;
    size_t column;
    for (row = 0; row < 3; ++row) {
        double exposure = 0.0;
        double log_exposure;
        double argument;
        for (column = 0; column < 3; ++column) {
            exposure += profile->capture_matrix[row][column] *
                source[column];
        }
        log_exposure = log2(exposure + profile->exposure_floor);
        argument = profile->response_slopes[row] * (
            log_exposure - profile->response_midpoints[row]
        );
        layer_response[row] =
            profile->maximum_responses[row] /
            (1.0 + exp(-argument));
    }
    for (row = 0; row < 3; ++row) {
        output[row] = 0.0;
        for (column = 0; column < 3; ++column) {
            output[row] += profile->scan_matrix[row][column] *
                layer_response[column];
        }
    }
}

static int nf_apply_pixel(
    const nf_ao6_residual_f32_profile_v1* profile,
    const float* input,
    float* output,
    float* tone_scale_output,
    float* chroma_scale_output) {
    double source[3];
    double raw[3];
    double residual[3];
    double lower[3];
    double upper[3];
    double tone_delta[3];
    double tone_output[3];
    double chroma_delta[3];
    double final_output[3];
    double luma_delta = 0.0;
    double tone_scale;
    double chroma_scale;
    size_t channel;
    for (channel = 0; channel < 3; ++channel) {
        source[channel] = (double)input[channel];
    }
    nf_raw_response(profile, source, raw);
    for (channel = 0; channel < 3; ++channel) {
        const double full = (
            raw[channel] - profile->black_endpoint[channel]
        ) / (
            profile->white_endpoint[channel] -
            profile->black_endpoint[channel]
        );
        if (
            !isfinite(full) ||
            full < -NF_AO6_F32_TOLERANCE_V1 ||
            full > 1.0 + NF_AO6_F32_TOLERANCE_V1
        ) {
            return 0;
        }
        residual[channel] = full - source[channel];
        luma_delta += residual[channel] *
            profile->luma_weights[channel];
        lower[channel] = source[channel] <= profile->hard_low_linear
            ? 0.0
            : fmin(source[channel], profile->guard_low_linear);
        upper[channel] = source[channel] >= profile->hard_high_linear
            ? 1.0
            : fmax(source[channel], profile->guard_high_linear);
    }
    for (channel = 0; channel < 3; ++channel) {
        tone_delta[channel] = profile->tone_strength * luma_delta;
    }
    tone_scale = nf_safe_scale(source, tone_delta, lower, upper);
    for (channel = 0; channel < 3; ++channel) {
        tone_output[channel] =
            source[channel] + tone_scale * tone_delta[channel];
        chroma_delta[channel] = profile->chroma_strength * (
            residual[channel] - luma_delta
        );
    }
    chroma_scale = nf_safe_scale(
        tone_output, chroma_delta, lower, upper);
    for (channel = 0; channel < 3; ++channel) {
        final_output[channel] =
            tone_output[channel] +
            chroma_scale * chroma_delta[channel];
        if (
            !isfinite(final_output[channel]) ||
            final_output[channel] <
                lower[channel] - NF_AO6_F32_TOLERANCE_V1 ||
            final_output[channel] >
                upper[channel] + NF_AO6_F32_TOLERANCE_V1
        ) {
            return 0;
        }
    }
    if (output != NULL) {
        for (channel = 0; channel < 3; ++channel) {
            output[channel] = (float)final_output[channel];
        }
        if (tone_scale_output != NULL) {
            *tone_scale_output = (float)tone_scale;
        }
        if (chroma_scale_output != NULL) {
            *chroma_scale_output = (float)chroma_scale;
        }
    }
    return 1;
}

uint32_t nf_ao6_residual_f32_abi_version_v1(void) {
    return NF_AO6_RESIDUAL_F32_ABI_VERSION_V1;
}

nf_ao6_residual_f32_status_v1
nf_ao6_residual_f32_validate_profile_v1(
    const nf_ao6_residual_f32_profile_v1* profile) {
    if (profile == NULL) {
        return NF_AO6_RESIDUAL_F32_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile)
        ? NF_AO6_RESIDUAL_F32_OK_V1
        : NF_AO6_RESIDUAL_F32_INVALID_PROFILE_V1;
}

nf_ao6_residual_f32_status_v1 nf_ao6_residual_f32_apply_v1(
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
            NULL,
            NULL,
            NULL
        )) {
            return NF_AO6_RESIDUAL_F32_NUMERIC_FAILURE_V1;
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
