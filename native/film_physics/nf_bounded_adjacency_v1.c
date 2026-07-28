#define NF_BOUNDED_ADJACENCY_BUILD
#include "nf_bounded_adjacency_v1.h"

#include <float.h>
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

static double nf_min2(double left, double right) {
    return left < right ? left : right;
}

static double nf_min3(double first, double second, double third) {
    return nf_min2(nf_min2(first, second), third);
}

static int nf_profile_valid(
    const nf_bounded_adjacency_profile_v1* profile) {
    size_t channel;
    if (
        profile == NULL ||
        profile->struct_size != sizeof(nf_bounded_adjacency_profile_v1) ||
        profile->abi_version != NF_BOUNDED_ADJACENCY_ABI_VERSION_V1 ||
        !nf_is_hex_sha256(profile->source_component_sha256) ||
        !isfinite(profile->maximum_absolute_transmittance_delta) ||
        profile->maximum_absolute_transmittance_delta <= 0.0 ||
        profile->maximum_absolute_transmittance_delta >= 1.0 ||
        !isfinite(profile->maximum_absolute_density_delta) ||
        profile->maximum_absolute_density_delta <= 0.0
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (
            !isfinite(profile->gain_rgb[channel]) ||
            profile->gain_rgb[channel] < 0.0 ||
            !isfinite(profile->black_reference_density[channel]) ||
            !isfinite(profile->white_reference_density[channel]) ||
            profile->black_reference_density[channel] < 0.0 ||
            profile->white_reference_density[channel] <=
                profile->black_reference_density[channel]
        ) {
            return 0;
        }
    }
    return 1;
}

uint32_t nf_bounded_adjacency_abi_version_v1(void) {
    return NF_BOUNDED_ADJACENCY_ABI_VERSION_V1;
}

nf_bounded_adjacency_status_v1
nf_bounded_adjacency_validate_profile_v1(
    const nf_bounded_adjacency_profile_v1* profile) {
    if (profile == NULL) {
        return NF_BOUNDED_ADJACENCY_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile)
        ? NF_BOUNDED_ADJACENCY_OK_V1
        : NF_BOUNDED_ADJACENCY_INVALID_PROFILE_V1;
}

nf_bounded_adjacency_status_v1 nf_bounded_adjacency_apply_v1(
    const nf_bounded_adjacency_profile_v1* profile,
    const double* developed_density_rgb,
    const double* blurred_density_rgb,
    size_t rgb_count,
    double* output_density_rgb) {
    size_t index;
    size_t channel;
    const nf_bounded_adjacency_status_v1 profile_status =
        nf_bounded_adjacency_validate_profile_v1(profile);
    if (profile_status != NF_BOUNDED_ADJACENCY_OK_V1) {
        return profile_status;
    }
    if (
        developed_density_rgb == NULL ||
        blurred_density_rgb == NULL ||
        output_density_rgb == NULL ||
        rgb_count == 0 ||
        rgb_count > SIZE_MAX / (3u * sizeof(double))
    ) {
        return NF_BOUNDED_ADJACENCY_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < rgb_count; ++index) {
        for (channel = 0; channel < 3; ++channel) {
            const double density =
                developed_density_rgb[index * 3u + channel];
            const double blurred =
                blurred_density_rgb[index * 3u + channel];
            if (
                !isfinite(density) ||
                !isfinite(blurred) ||
                density <
                    profile->black_reference_density[channel] - 1.0e-12 ||
                density >
                    profile->white_reference_density[channel] + 1.0e-12 ||
                blurred <
                    profile->black_reference_density[channel] - 1.0e-12 ||
                blurred >
                    profile->white_reference_density[channel] + 1.0e-12
            ) {
                return NF_BOUNDED_ADJACENCY_INVALID_INPUT_V1;
            }
        }
    }
    for (index = 0; index < rgb_count; ++index) {
        for (channel = 0; channel < 3; ++channel) {
            const double density =
                developed_density_rgb[index * 3u + channel];
            const double blurred =
                blurred_density_rgb[index * 3u + channel];
            const double raw = profile->gain_rgb[channel] * (
                density - blurred
            );
            const double transmittance = pow(10.0, -density);
            const double lower_transmittance = fmax(
                transmittance -
                    profile->maximum_absolute_transmittance_delta,
                DBL_MIN
            );
            const double upper_transmittance = fmin(
                transmittance +
                    profile->maximum_absolute_transmittance_delta,
                1.0
            );
            const double positive_limit = nf_min3(
                -log10(lower_transmittance) - density,
                profile->maximum_absolute_density_delta,
                profile->white_reference_density[channel] - density
            );
            const double negative_limit = nf_min3(
                density + log10(upper_transmittance),
                profile->maximum_absolute_density_delta,
                density - profile->black_reference_density[channel]
            );
            const double limit = raw >= 0.0
                ? positive_limit
                : negative_limit;
            double correction = 0.0;
            double output;
            if (limit > 0.0) {
                correction = copysign(
                    limit * tanh(fabs(raw) / limit),
                    raw
                );
            }
            output = density + correction;
            if (
                !isfinite(output) ||
                output <
                    profile->black_reference_density[channel] - 1.0e-12 ||
                output >
                    profile->white_reference_density[channel] + 1.0e-12 ||
                fabs(
                    pow(10.0, -output) - transmittance
                ) >
                    profile->maximum_absolute_transmittance_delta + 1.0e-12
            ) {
                return NF_BOUNDED_ADJACENCY_NUMERIC_FAILURE_V1;
            }
            output_density_rgb[index * 3u + channel] = output;
        }
    }
    return NF_BOUNDED_ADJACENCY_OK_V1;
}
