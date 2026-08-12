#define NF_CLOUD_ATTENUATION_F32_BUILD
#include "nf_cloud_attenuation_f32_v1.h"

#include <stdint.h>

extern float log10f(float value);

static int nf_isfinite_f32(float value) {
    return value == value && value <= 3.402823466e+38F &&
        value >= -3.402823466e+38F;
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

uint32_t nf_cloud_attenuation_f32_abi_version_v1(void) {
    return NF_CLOUD_ATTENUATION_F32_ABI_VERSION_V1;
}

nf_cloud_attenuation_f32_status_v1 nf_cloud_attenuation_f32_apply_v1(
    const float* expected_transmittance_rgb,
    const float* base_transmittance_rgb,
    size_t pixel_count,
    const float channel_gain_rgb[3],
    float* output_density_rgb,
    float* output_transmittance_rgb) {
    size_t total;
    size_t index;
    if (expected_transmittance_rgb == NULL || base_transmittance_rgb == NULL ||
        channel_gain_rgb == NULL || output_density_rgb == NULL ||
        output_transmittance_rgb == NULL || pixel_count == 0u ||
        pixel_count > SIZE_MAX / (3u * sizeof(float))) {
        return NF_CLOUD_ATTENUATION_F32_INVALID_ARGUMENT_V1;
    }
    total = 3u * pixel_count;
    if (nf_ranges_overlap(expected_transmittance_rgb, total,
            output_density_rgb, total) ||
        nf_ranges_overlap(expected_transmittance_rgb, total,
            output_transmittance_rgb, total) ||
        nf_ranges_overlap(base_transmittance_rgb, total,
            output_density_rgb, total) ||
        nf_ranges_overlap(base_transmittance_rgb, total,
            output_transmittance_rgb, total) ||
        nf_ranges_overlap(output_density_rgb, total,
            output_transmittance_rgb, total)) {
        return NF_CLOUD_ATTENUATION_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0u; index < 3u; ++index) {
        if (!nf_isfinite_f32(channel_gain_rgb[index]) ||
            channel_gain_rgb[index] <= 0.0f || channel_gain_rgb[index] > 1.0f) {
            return NF_CLOUD_ATTENUATION_F32_DOMAIN_ERROR_V1;
        }
    }
    /* Full preflight keeps both outputs unchanged on all failures. */
    for (index = 0u; index < total; ++index) {
        const float expected = expected_transmittance_rgb[index];
        const float base = base_transmittance_rgb[index];
        const float gain = channel_gain_rgb[index % 3u];
        const float candidate = expected + gain * (base - expected);
        if (!nf_isfinite_f32(expected) || !nf_isfinite_f32(base) ||
            expected <= 0.0f || expected >= 1.0f ||
            base <= 0.0f || base >= 1.0f ||
            !nf_isfinite_f32(candidate) || candidate <= 0.0f || candidate >= 1.0f) {
            return NF_CLOUD_ATTENUATION_F32_DOMAIN_ERROR_V1;
        }
    }
    for (index = 0u; index < total; ++index) {
        const float expected = expected_transmittance_rgb[index];
        const float base = base_transmittance_rgb[index];
        const float candidate =
            expected + channel_gain_rgb[index % 3u] * (base - expected);
        output_density_rgb[index] = -log10f(candidate);
        output_transmittance_rgb[index] = candidate;
    }
    return NF_CLOUD_ATTENUATION_F32_OK_V1;
}
