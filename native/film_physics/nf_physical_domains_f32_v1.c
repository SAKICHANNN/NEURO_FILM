#define NF_PHYSICAL_DOMAINS_F32_BUILD
#include "nf_physical_domains_f32_v1.h"

#include <math.h>

#define NF_F32_DENSITY_TOLERANCE_V1 2.0e-6

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

static double nf_spline_apply(
    const nf_physical_domains_f32_profile_v1* profile,
    size_t channel,
    double input) {
    const uint32_t count = profile->knot_count[channel];
    const double* x = profile->x_knots[channel];
    const double* y = profile->y_knots[channel];
    const double* derivative = profile->derivatives[channel];
    uint32_t bin = 0;
    double width;
    double height;
    double slope;
    double theta;
    double one_minus;
    double denominator;
    double numerator;
    if (input < x[0]) {
        return y[0] + derivative[0] * (input - x[0]);
    }
    if (input > x[count - 1]) {
        return y[count - 1] +
            derivative[count - 1] * (input - x[count - 1]);
    }
    while (bin + 1 < count && input >= x[bin + 1]) {
        ++bin;
    }
    if (bin >= count - 1) {
        bin = count - 2;
    }
    width = x[bin + 1] - x[bin];
    height = y[bin + 1] - y[bin];
    slope = height / width;
    theta = (input - x[bin]) / width;
    one_minus = theta * (1.0 - theta);
    denominator = slope + (
        derivative[bin + 1] + derivative[bin] - 2.0 * slope
    ) * one_minus;
    numerator = height * (
        slope * theta * theta + derivative[bin] * one_minus
    );
    return y[bin] + numerator / denominator;
}

static void nf_raw_reflectance(
    const nf_physical_domains_f32_profile_v1* profile,
    const double density[3],
    double output[3]) {
    double transmission[3];
    size_t row;
    size_t column;
    for (row = 0; row < 3; ++row) {
        double absorption = 0.0;
        for (column = 0; column < 3; ++column) {
            absorption +=
                profile->dye_absorption_matrix[row][column] *
                density[column];
        }
        transmission[row] = pow(10.0, -absorption);
    }
    for (row = 0; row < 3; ++row) {
        const double paper_log_exposure = log2(
            profile->print_matrix[row][0] * transmission[0] +
            profile->print_matrix[row][1] * transmission[1] +
            profile->print_matrix[row][2] * transmission[2] +
            profile->exposure_floor
        );
        const double argument = profile->paper_slopes[row] * (
            paper_log_exposure - profile->paper_midpoints[row]
        );
        const double paper_density =
            profile->paper_maximum_densities[row] /
            (1.0 + exp(-argument));
        output[row] = pow(10.0, -paper_density);
    }
}

static int nf_profile_numeric_valid(
    const nf_physical_domains_f32_profile_v1* profile) {
    size_t channel;
    uint32_t index;
    if (
        !isfinite(profile->reference_linear) ||
        !isfinite(profile->black_offset) ||
        profile->reference_linear <= 0.0 ||
        profile->black_offset <= 0.0 ||
        profile->exposure_floor != (1.0 / 65536.0) ||
        !isfinite(profile->matrix_minimum_determinant) ||
        profile->matrix_minimum_determinant <= 0.0 ||
        profile->matrix_minimum_determinant >= 1.0 ||
        !nf_matrix_valid(
            profile->dye_absorption_matrix,
            profile->matrix_minimum_determinant) ||
        !nf_matrix_valid(
            profile->print_matrix,
            profile->matrix_minimum_determinant)
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        const uint32_t count = profile->knot_count[channel];
        if (
            count < 2 ||
            count > NF_PHYSICAL_DOMAINS_F32_MAX_KNOTS_V1
        ) {
            return 0;
        }
        for (index = 0; index < count; ++index) {
            if (
                !isfinite(profile->x_knots[channel][index]) ||
                !isfinite(profile->y_knots[channel][index]) ||
                !isfinite(profile->derivatives[channel][index]) ||
                profile->derivatives[channel][index] <= 0.0 ||
                (
                    index > 0 &&
                    (
                        profile->x_knots[channel][index] <=
                            profile->x_knots[channel][index - 1] ||
                        profile->y_knots[channel][index] <=
                            profile->y_knots[channel][index - 1]
                    )
                )
            ) {
                return 0;
            }
        }
        if (
            !isfinite(profile->paper_midpoints[channel]) ||
            !isfinite(profile->paper_slopes[channel]) ||
            !isfinite(profile->paper_maximum_densities[channel]) ||
            !isfinite(profile->black_reference_density[channel]) ||
            !isfinite(profile->white_reference_density[channel]) ||
            profile->paper_slopes[channel] < 0.2 ||
            profile->paper_slopes[channel] > 4.0 ||
            profile->paper_maximum_densities[channel] < 0.2 ||
            profile->paper_maximum_densities[channel] > 4.0 ||
            profile->white_reference_density[channel] <=
                profile->black_reference_density[channel]
        ) {
            return 0;
        }
    }
    return 1;
}

uint32_t nf_physical_domains_f32_abi_version_v1(void) {
    return NF_PHYSICAL_DOMAINS_F32_ABI_VERSION_V1;
}

nf_physical_domains_f32_status_v1
nf_physical_domains_f32_validate_profile_v1(
    const nf_physical_domains_f32_profile_v1* profile) {
    double raw_black[3];
    double raw_white[3];
    size_t channel;
    if (profile == NULL) {
        return NF_PHYSICAL_DOMAINS_F32_INVALID_ARGUMENT_V1;
    }
    if (
        profile->struct_size !=
            sizeof(nf_physical_domains_f32_profile_v1) ||
        profile->abi_version != NF_PHYSICAL_DOMAINS_F32_ABI_VERSION_V1 ||
        !nf_is_hex_sha256(profile->source_component_sha256) ||
        !nf_profile_numeric_valid(profile)
    ) {
        return NF_PHYSICAL_DOMAINS_F32_INVALID_PROFILE_V1;
    }
    nf_raw_reflectance(
        profile, profile->black_reference_density, raw_black);
    nf_raw_reflectance(
        profile, profile->white_reference_density, raw_white);
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite(raw_black[channel]) ||
            !isfinite(raw_white[channel]) ||
            raw_white[channel] <= raw_black[channel]
        ) {
            return NF_PHYSICAL_DOMAINS_F32_INVALID_PROFILE_V1;
        }
    }
    return NF_PHYSICAL_DOMAINS_F32_OK_V1;
}

nf_physical_domains_f32_status_v1
nf_physical_sensitometry_f32_apply_v1(
    const nf_physical_domains_f32_profile_v1* profile,
    const float* scene_linear_rgb,
    size_t rgb_count,
    float* developed_density_rgb) {
    size_t index;
    size_t channel;
    const nf_physical_domains_f32_status_v1 status =
        nf_physical_domains_f32_validate_profile_v1(profile);
    if (status != NF_PHYSICAL_DOMAINS_F32_OK_V1) {
        return status;
    }
    if (
        scene_linear_rgb == NULL ||
        developed_density_rgb == NULL ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float))
    ) {
        return NF_PHYSICAL_DOMAINS_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        if (
            !isfinite((double)scene_linear_rgb[index]) ||
            scene_linear_rgb[index] < 0.0f ||
            scene_linear_rgb[index] > 1.0f
        ) {
            return NF_PHYSICAL_DOMAINS_F32_INVALID_INPUT_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        for (channel = 0; channel < 3; ++channel) {
            const double sample =
                (double)scene_linear_rgb[index * 3u + channel];
            const double exposure = log10(
                (sample + profile->black_offset) /
                (profile->reference_linear + profile->black_offset)
            );
            developed_density_rgb[index * 3u + channel] = (float)
                nf_spline_apply(profile, channel, exposure);
        }
    }
    return NF_PHYSICAL_DOMAINS_F32_OK_V1;
}

nf_physical_domains_f32_status_v1
nf_physical_interpretation_f32_apply_v1(
    const nf_physical_domains_f32_profile_v1* profile,
    const float* developed_density_rgb,
    size_t rgb_count,
    float* scan_linear_rgb) {
    size_t index;
    size_t channel;
    double raw_black[3];
    double raw_white[3];
    const nf_physical_domains_f32_status_v1 status =
        nf_physical_domains_f32_validate_profile_v1(profile);
    if (status != NF_PHYSICAL_DOMAINS_F32_OK_V1) {
        return status;
    }
    if (
        developed_density_rgb == NULL ||
        scan_linear_rgb == NULL ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float))
    ) {
        return NF_PHYSICAL_DOMAINS_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < rgb_count; ++index) {
        for (channel = 0; channel < 3; ++channel) {
            const double value =
                (double)developed_density_rgb[index * 3u + channel];
            if (
                !isfinite(value) ||
                value <
                    profile->black_reference_density[channel] -
                    NF_F32_DENSITY_TOLERANCE_V1 ||
                value >
                    profile->white_reference_density[channel] +
                    NF_F32_DENSITY_TOLERANCE_V1
            ) {
                return NF_PHYSICAL_DOMAINS_F32_INVALID_INPUT_V1;
            }
        }
    }
    nf_raw_reflectance(
        profile, profile->black_reference_density, raw_black);
    nf_raw_reflectance(
        profile, profile->white_reference_density, raw_white);
    for (index = 0; index < rgb_count; ++index) {
        double density[3];
        double raw[3];
        for (channel = 0; channel < 3; ++channel) {
            density[channel] =
                (double)developed_density_rgb[index * 3u + channel];
        }
        nf_raw_reflectance(profile, density, raw);
        for (channel = 0; channel < 3; ++channel) {
            double value = (raw[channel] - raw_black[channel]) /
                (raw_white[channel] - raw_black[channel]);
            if (
                !isfinite(value) ||
                value < -NF_F32_DENSITY_TOLERANCE_V1 ||
                value > 1.0 + NF_F32_DENSITY_TOLERANCE_V1
            ) {
                return NF_PHYSICAL_DOMAINS_F32_NUMERIC_FAILURE_V1;
            }
            if (value < 0.0) {
                value = 0.0;
            } else if (value > 1.0) {
                value = 1.0;
            }
            scan_linear_rgb[index * 3u + channel] = (float)value;
        }
    }
    return NF_PHYSICAL_DOMAINS_F32_OK_V1;
}
