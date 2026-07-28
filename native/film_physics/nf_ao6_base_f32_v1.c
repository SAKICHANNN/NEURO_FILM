#define NF_AO6_BASE_F32_BUILD
#include "nf_ao6_base_f32_v1.h"

#include <math.h>

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
            if (!isfinite(matrix[row][column]) ||
                matrix[row][column] < 0.0) {
                return 0;
            }
            sum += matrix[row][column];
        }
        if (fabs(sum - 1.0) > 1.0e-12) {
            return 0;
        }
    }
    return nf_det3(matrix) >= determinant_floor;
}

static int nf_profile_valid(const nf_ao6_base_f32_profile_v1* profile) {
    size_t channel;
    if (
        profile == NULL ||
        profile->struct_size != sizeof(nf_ao6_base_f32_profile_v1) ||
        profile->abi_version != NF_AO6_BASE_F32_ABI_VERSION_V1 ||
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
            profile->dye_absorption_matrix,
            profile->matrix_minimum_determinant) ||
        !nf_matrix_valid(
            profile->print_matrix,
            profile->matrix_minimum_determinant) ||
        profile->density_strength != 0.5 ||
        profile->style_strength != 0.58f ||
        profile->luma_strength != 0.35f ||
        profile->gamut_iterations != 14u ||
        profile->output_margin_8bit != 4u
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite(profile->negative_midpoints[channel]) ||
            profile->negative_midpoints[channel] < -12.0 ||
            profile->negative_midpoints[channel] > 2.0 ||
            !isfinite(profile->negative_slopes[channel]) ||
            profile->negative_slopes[channel] < 0.2 ||
            profile->negative_slopes[channel] > 4.0 ||
            !isfinite(profile->negative_maximum_densities[channel]) ||
            profile->negative_maximum_densities[channel] < 0.2 ||
            profile->negative_maximum_densities[channel] > 4.0 ||
            !isfinite(profile->paper_midpoints[channel]) ||
            profile->paper_midpoints[channel] < -12.0 ||
            profile->paper_midpoints[channel] > 2.0 ||
            !isfinite(profile->paper_slopes[channel]) ||
            profile->paper_slopes[channel] < 0.2 ||
            profile->paper_slopes[channel] > 4.0 ||
            !isfinite(profile->paper_maximum_densities[channel]) ||
            profile->paper_maximum_densities[channel] < 0.2 ||
            profile->paper_maximum_densities[channel] > 4.0 ||
            !isfinite(profile->black_endpoint[channel]) ||
            !isfinite(profile->white_endpoint[channel]) ||
            profile->white_endpoint[channel] -
                profile->black_endpoint[channel] <
                profile->minimum_endpoint_span ||
            !isfinite((double)profile->destination_mean[channel]) ||
            !isfinite((double)profile->destination_std[channel]) ||
            profile->destination_std[channel] <= 0.0f
        ) {
            return 0;
        }
    }
    return 1;
}

static int nf_context_valid(
    const nf_ao6_base_f32_context_v1* context) {
    size_t channel;
    if (
        context == NULL ||
        context->struct_size != sizeof(nf_ao6_base_f32_context_v1) ||
        context->abi_version != NF_AO6_BASE_F32_ABI_VERSION_V1 ||
        context->pixel_count == 0u
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite((double)context->source_mean[channel]) ||
            !isfinite((double)context->source_std[channel]) ||
            context->source_std[channel] < 1.0e-3f
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
        double full = (raw[channel] - profile->black_endpoint[channel]) /
            (profile->white_endpoint[channel] -
             profile->black_endpoint[channel]);
        double value;
        if (!isfinite(full) || full < -3.0e-12 || full > 1.0 + 3.0e-12) {
            return 0;
        }
        value = linear[channel] + profile->density_strength *
            (full - linear[channel]);
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
    const float y = (
        0.212671f * r + 0.715160f * g + 0.072169f * b
    );
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

static float nf_lab_inv_f(float value) {
    return value > (6.0f / 29.0f)
        ? value * value * value
        : 3.0f * (6.0f / 29.0f) * (6.0f / 29.0f) *
            (value - 4.0f / 29.0f);
}

static void nf_lab_to_linear(const float lab[3], float rgb[3]) {
    const float fy = (lab[0] + 16.0f) / 116.0f;
    const float fx = fy + lab[1] / 500.0f;
    const float fz = fy - lab[2] / 200.0f;
    const float x = 0.95047f * nf_lab_inv_f(fx);
    const float y = nf_lab_inv_f(fy);
    const float z = 1.08883f * nf_lab_inv_f(fz);
    rgb[0] = 3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    rgb[1] = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    rgb[2] = 0.0556434f * x - 0.2040259f * y + 1.0572252f * z;
}

static int nf_lab_in_gamut(const float lab[3]) {
    float rgb[3];
    size_t channel;
    nf_lab_to_linear(lab, rgb);
    for (channel = 0; channel < 3; ++channel) {
        if (!isfinite((double)rgb[channel]) ||
            rgb[channel] < 0.0f || rgb[channel] > 1.0f) {
            return 0;
        }
    }
    return 1;
}

static int nf_apply_pixel(
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
    for (iteration = 0; iteration < profile->gamut_iterations; ++iteration) {
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
    candidate[0] = target_lab[0];
    candidate[1] = target_lab[1] * low;
    candidate[2] = target_lab[2] * low;
    nf_lab_to_linear(candidate, linear);
    for (channel = 0; channel < 3; ++channel) {
        float value;
        const float margin = (float)profile->output_margin_8bit / 255.0f;
        if (!isfinite((double)linear[channel])) {
            return 0;
        }
        value = nf_linear_to_encoded(fminf(1.0f, fmaxf(0.0f, linear[channel])));
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

uint32_t nf_ao6_base_f32_abi_version_v1(void) {
    return NF_AO6_BASE_F32_ABI_VERSION_V1;
}

nf_ao6_base_f32_status_v1 nf_ao6_base_f32_validate_profile_v1(
    const nf_ao6_base_f32_profile_v1* profile) {
    if (profile == NULL) {
        return NF_AO6_BASE_F32_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile)
        ? NF_AO6_BASE_F32_OK_V1
        : NF_AO6_BASE_F32_INVALID_PROFILE_V1;
}

nf_ao6_base_f32_status_v1 nf_ao6_base_f32_apply_v1(
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
        if (!nf_apply_pixel(
            profile, context, &encoded_srgb[index * 3u], NULL
        )) {
            return NF_AO6_BASE_F32_NUMERIC_FAILURE_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        if (!nf_apply_pixel(
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
