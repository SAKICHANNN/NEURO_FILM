#define NF_GRANULARITY_AMPLITUDE_F32_BUILD
#include "nf_granularity_amplitude_f32_v1.h"

#include <math.h>
#include <stdint.h>

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

static int nf_ranges_overlap(
    const float* first, size_t first_count,
    const float* second, size_t second_count) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const size_t first_bytes = first_count * sizeof(float);
    const size_t second_bytes = second_count * sizeof(float);
    const uintptr_t first_end = first_start + first_bytes;
    const uintptr_t second_end = second_start + second_bytes;
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

static int nf_profile_valid(
    const nf_granularity_amplitude_f32_profile_v1* profile) {
    size_t channel;
    uint32_t index;
    if (profile == NULL ||
        profile->struct_size != sizeof(nf_granularity_amplitude_f32_profile_v1) ||
        profile->abi_version != NF_GRANULARITY_AMPLITUDE_F32_ABI_VERSION_V1 ||
        !nf_is_hex_sha256(profile->source_profile_sha256) ||
        !isfinite(profile->shared_amplitude) ||
        profile->shared_amplitude <= 0.0 ||
        !isfinite(profile->measurement_energy) ||
        profile->measurement_energy <= 0.0 || profile->measurement_energy > 1.0) {
        return 0;
    }
    for (channel = 0; channel < 3u; ++channel) {
        const uint32_t count = profile->knot_count[channel];
        if (count < 3u || count > NF_GRANULARITY_AMPLITUDE_F32_MAX_KNOTS_V1 ||
            !isfinite(profile->channel_floor_variance[channel]) ||
            profile->channel_floor_variance[channel] <= 0.0) {
            return 0;
        }
        for (index = 0; index < count; ++index) {
            if (!isfinite(profile->log_exposure_knots[channel][index]) ||
                !isfinite(profile->density_knots[channel][index]) ||
                (index > 0u &&
                    (profile->log_exposure_knots[channel][index] <=
                        profile->log_exposure_knots[channel][index - 1u] ||
                     profile->density_knots[channel][index] <
                        profile->density_knots[channel][index - 1u]))) {
                return 0;
            }
        }
    }
    return 1;
}

static uint32_t nf_bin(
    const nf_granularity_amplitude_f32_profile_v1* profile,
    size_t channel,
    double exposure) {
    const uint32_t count = profile->knot_count[channel];
    uint32_t bin = 0u;
    while (bin + 1u < count &&
           exposure >= profile->log_exposure_knots[channel][bin + 1u]) {
        ++bin;
    }
    return bin >= count - 1u ? count - 2u : bin;
}

static void nf_evaluate(
    const nf_granularity_amplitude_f32_profile_v1* profile,
    size_t channel,
    double exposure,
    double* density,
    double* point_sigma) {
    const uint32_t bin = nf_bin(profile, channel, exposure);
    const double x0 = profile->log_exposure_knots[channel][bin];
    const double x1 = profile->log_exposure_knots[channel][bin + 1u];
    const double y0 = profile->density_knots[channel][bin];
    const double y1 = profile->density_knots[channel][bin + 1u];
    const double slope = (y1 - y0) / (x1 - x0);
    const double fraction = (exposure - x0) / (x1 - x0);
    const double variance = profile->channel_floor_variance[channel] +
        profile->shared_amplitude * slope * slope / pow(10.0, exposure);
    *density = y0 + fraction * (y1 - y0);
    *point_sigma = sqrt(variance / profile->measurement_energy);
}

uint32_t nf_granularity_amplitude_f32_abi_version_v1(void) {
    return NF_GRANULARITY_AMPLITUDE_F32_ABI_VERSION_V1;
}

nf_granularity_amplitude_f32_status_v1
nf_granularity_amplitude_f32_validate_profile_v1(
    const nf_granularity_amplitude_f32_profile_v1* profile) {
    if (profile == NULL) {
        return NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile) ? NF_GRANULARITY_AMPLITUDE_F32_OK_V1 :
        NF_GRANULARITY_AMPLITUDE_F32_INVALID_PROFILE_V1;
}

nf_granularity_amplitude_f32_status_v1
nf_granularity_amplitude_f32_apply_v1(
    const nf_granularity_amplitude_f32_profile_v1* profile,
    const float* relative_log_exposure_chw,
    size_t sample_count,
    float* developed_density_chw,
    float* point_density_sigma_chw) {
    size_t total;
    size_t channel;
    size_t index;
    if (!nf_profile_valid(profile)) {
        return profile == NULL ? NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1 :
            NF_GRANULARITY_AMPLITUDE_F32_INVALID_PROFILE_V1;
    }
    if (relative_log_exposure_chw == NULL || developed_density_chw == NULL ||
        point_density_sigma_chw == NULL || sample_count == 0u ||
        sample_count > SIZE_MAX / (3u * sizeof(float))) {
        return NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1;
    }
    total = 3u * sample_count;
    if (nf_ranges_overlap(relative_log_exposure_chw, total,
            developed_density_chw, total) ||
        nf_ranges_overlap(relative_log_exposure_chw, total,
            point_density_sigma_chw, total) ||
        nf_ranges_overlap(developed_density_chw, total,
            point_density_sigma_chw, total)) {
        return NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1;
    }
    /* Full preflight keeps both outputs unchanged on every rejected request. */
    for (channel = 0; channel < 3u; ++channel) {
        const uint32_t count = profile->knot_count[channel];
        const double lower = profile->log_exposure_knots[channel][0];
        const double upper = profile->log_exposure_knots[channel][count - 1u];
        for (index = 0; index < sample_count; ++index) {
            double density;
            double point_sigma;
            const double exposure =
                (double)relative_log_exposure_chw[channel * sample_count + index];
            if (!isfinite(exposure) || exposure < lower || exposure > upper) {
                return NF_GRANULARITY_AMPLITUDE_F32_DOMAIN_ERROR_V1;
            }
            nf_evaluate(profile, channel, exposure, &density, &point_sigma);
            if (!isfinite(density) || density < 0.0 ||
                !isfinite(point_sigma) || point_sigma <= 0.0) {
                return NF_GRANULARITY_AMPLITUDE_F32_DOMAIN_ERROR_V1;
            }
        }
    }
    for (channel = 0; channel < 3u; ++channel) {
        for (index = 0; index < sample_count; ++index) {
            double density;
            double point_sigma;
            const size_t offset = channel * sample_count + index;
            nf_evaluate(profile, channel,
                (double)relative_log_exposure_chw[offset],
                &density, &point_sigma);
            developed_density_chw[offset] = (float)density;
            point_density_sigma_chw[offset] = (float)point_sigma;
        }
    }
    return NF_GRANULARITY_AMPLITUDE_F32_OK_V1;
}

nf_granularity_amplitude_f32_status_v1
nf_granularity_amplitude_f32_apply_layer_v1(
    const nf_granularity_amplitude_f32_profile_v1* profile,
    uint32_t channel,
    const float* relative_log_exposure,
    size_t sample_count,
    float* developed_density,
    float* point_density_sigma) {
    size_t index;
    uint32_t count;
    double lower;
    double upper;
    if (!nf_profile_valid(profile)) {
        return profile == NULL ? NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1 :
            NF_GRANULARITY_AMPLITUDE_F32_INVALID_PROFILE_V1;
    }
    if (channel >= 3u || relative_log_exposure == NULL ||
        developed_density == NULL || point_density_sigma == NULL ||
        sample_count == 0u || sample_count > SIZE_MAX / sizeof(float) ||
        nf_ranges_overlap(relative_log_exposure, sample_count,
            developed_density, sample_count) ||
        nf_ranges_overlap(relative_log_exposure, sample_count,
            point_density_sigma, sample_count) ||
        nf_ranges_overlap(developed_density, sample_count,
            point_density_sigma, sample_count)) {
        return NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1;
    }
    count = profile->knot_count[channel];
    lower = profile->log_exposure_knots[channel][0];
    upper = profile->log_exposure_knots[channel][count - 1u];
    for (index = 0; index < sample_count; ++index) {
        double density;
        double point_sigma;
        const double exposure = (double)relative_log_exposure[index];
        if (!isfinite(exposure) || exposure < lower || exposure > upper) {
            return NF_GRANULARITY_AMPLITUDE_F32_DOMAIN_ERROR_V1;
        }
        nf_evaluate(profile, channel, exposure, &density, &point_sigma);
        if (!isfinite(density) || density < 0.0 ||
            !isfinite(point_sigma) || point_sigma <= 0.0) {
            return NF_GRANULARITY_AMPLITUDE_F32_DOMAIN_ERROR_V1;
        }
    }
    for (index = 0; index < sample_count; ++index) {
        double density;
        double point_sigma;
        nf_evaluate(profile, channel, (double)relative_log_exposure[index],
            &density, &point_sigma);
        developed_density[index] = (float)density;
        point_density_sigma[index] = (float)point_sigma;
    }
    return NF_GRANULARITY_AMPLITUDE_F32_OK_V1;
}
