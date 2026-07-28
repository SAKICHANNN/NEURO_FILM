#define NF_NEUTRAL_GAUGE_F32_BUILD
#include "nf_neutral_gauge_f32_v1.h"

#include <math.h>

#define NF_NEUTRAL_GAUGE_F32_TOLERANCE_V1 2.0e-6

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

static int nf_profile_valid(
    const nf_neutral_gauge_f32_profile_v1* profile) {
    size_t channel;
    uint32_t index;
    if (
        profile == NULL ||
        profile->struct_size !=
            sizeof(nf_neutral_gauge_f32_profile_v1) ||
        profile->abi_version != NF_NEUTRAL_GAUGE_F32_ABI_VERSION_V1 ||
        !nf_is_hex_sha256(profile->source_component_sha256)
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        const uint32_t count = profile->knot_count[channel];
        if (
            count < 2u ||
            count > NF_NEUTRAL_GAUGE_F32_MAX_KNOTS_V1
        ) {
            return 0;
        }
        for (index = 0; index < count; ++index) {
            const double x = profile->x_knots[channel][index];
            const double y = profile->y_knots[channel][index];
            const double derivative =
                profile->derivatives[channel][index];
            if (
                !isfinite(x) ||
                !isfinite(y) ||
                !isfinite(derivative) ||
                derivative <= 0.0 ||
                (
                    index > 0u &&
                    (
                        x <= profile->x_knots[channel][index - 1u] ||
                        y <= profile->y_knots[channel][index - 1u]
                    )
                )
            ) {
                return 0;
            }
        }
        if (
            profile->x_knots[channel][0] != 0.0 ||
            profile->y_knots[channel][0] != 0.0 ||
            profile->x_knots[channel][count - 1u] != 1.0 ||
            profile->y_knots[channel][count - 1u] != 1.0
        ) {
            return 0;
        }
    }
    return 1;
}

static uint32_t nf_find_bin(
    const double* knots,
    uint32_t count,
    double input) {
    uint32_t low = 0u;
    uint32_t high = count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2u;
        if (input < knots[middle]) {
            high = middle;
        } else {
            low = middle + 1u;
        }
    }
    if (low == 0u) {
        return 0u;
    }
    if (low >= count) {
        return count - 2u;
    }
    return low - 1u;
}

static double nf_spline_apply(
    const nf_neutral_gauge_f32_profile_v1* profile,
    size_t channel,
    double input) {
    const uint32_t count = profile->knot_count[channel];
    const double* x = profile->x_knots[channel];
    const double* y = profile->y_knots[channel];
    const double* derivative = profile->derivatives[channel];
    uint32_t bin;
    double width;
    double height;
    double slope;
    double theta;
    double theta_one_minus;
    double denominator;
    double numerator;
    if (input < x[0]) {
        return y[0] + derivative[0] * (input - x[0]);
    }
    if (input > x[count - 1u]) {
        return y[count - 1u] +
            derivative[count - 1u] * (input - x[count - 1u]);
    }
    bin = nf_find_bin(x, count, input);
    width = x[bin + 1u] - x[bin];
    height = y[bin + 1u] - y[bin];
    slope = height / width;
    theta = (input - x[bin]) / width;
    theta_one_minus = theta * (1.0 - theta);
    denominator = slope + (
        derivative[bin + 1u] + derivative[bin] - 2.0 * slope
    ) * theta_one_minus;
    numerator = height * (
        slope * theta * theta +
        derivative[bin] * theta_one_minus
    );
    return y[bin] + numerator / denominator;
}

uint32_t nf_neutral_gauge_f32_abi_version_v1(void) {
    return NF_NEUTRAL_GAUGE_F32_ABI_VERSION_V1;
}

nf_neutral_gauge_f32_status_v1
nf_neutral_gauge_f32_validate_profile_v1(
    const nf_neutral_gauge_f32_profile_v1* profile) {
    if (profile == NULL) {
        return NF_NEUTRAL_GAUGE_F32_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile)
        ? NF_NEUTRAL_GAUGE_F32_OK_V1
        : NF_NEUTRAL_GAUGE_F32_INVALID_PROFILE_V1;
}

nf_neutral_gauge_f32_status_v1 nf_neutral_gauge_f32_apply_v1(
    const nf_neutral_gauge_f32_profile_v1* profile,
    const float* scan_linear_rgb,
    size_t rgb_count,
    float* gauged_linear_rgb) {
    size_t index;
    size_t channel;
    const nf_neutral_gauge_f32_status_v1 profile_status =
        nf_neutral_gauge_f32_validate_profile_v1(profile);
    if (profile_status != NF_NEUTRAL_GAUGE_F32_OK_V1) {
        return profile_status;
    }
    if (
        scan_linear_rgb == NULL ||
        gauged_linear_rgb == NULL ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(float))
    ) {
        return NF_NEUTRAL_GAUGE_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < rgb_count * 3u; ++index) {
        if (
            !isfinite((double)scan_linear_rgb[index]) ||
            scan_linear_rgb[index] < 0.0f ||
            scan_linear_rgb[index] > 1.0f
        ) {
            return NF_NEUTRAL_GAUGE_F32_INVALID_INPUT_V1;
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        for (channel = 0; channel < 3; ++channel) {
            double output = nf_spline_apply(
                profile,
                channel,
                (double)scan_linear_rgb[index * 3u + channel]
            );
            if (
                !isfinite(output) ||
                output < -NF_NEUTRAL_GAUGE_F32_TOLERANCE_V1 ||
                output > 1.0 + NF_NEUTRAL_GAUGE_F32_TOLERANCE_V1
            ) {
                return NF_NEUTRAL_GAUGE_F32_NUMERIC_FAILURE_V1;
            }
            if (output < 0.0) {
                output = 0.0;
            } else if (output > 1.0) {
                output = 1.0;
            }
            gauged_linear_rgb[index * 3u + channel] = (float)output;
        }
    }
    return NF_NEUTRAL_GAUGE_F32_OK_V1;
}
