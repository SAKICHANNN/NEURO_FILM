#define NF_THOMAS_RGB16_F32_BUILD
#define NF_GRANULARITY_AMPLITUDE_F32_BUILD
#define NF_NEUTRAL_GAUGE_F32_BUILD
#define NF_THOMAS_ROWS_F32_BUILD
#define NF_THOMAS_FIELD_F32_BUILD
#define NF_NEUMAIER_F32_BUILD
#include "nf_thomas_rgb16_f32_v1.h"
#include "nf_neumaier_f32_v1.h"
#include "reference_srgb_oetf_quantize_v1.h"

#include <math.h>
#include <stdint.h>

static int nf_ranges_overlap_bytes(
    const void* first,
    size_t first_bytes,
    const void* second,
    size_t second_bytes) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const uintptr_t first_end = first_start + first_bytes;
    const uintptr_t second_end = second_start + second_bytes;
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

uint32_t nf_thomas_rgb16_f32_abi_version_v1(void) {
    return NF_THOMAS_RGB16_F32_ABI_VERSION_V1;
}

nf_thomas_rgb16_f32_status_v1 nf_thomas_rgb16_f32_workspace_bytes_v1(
    size_t width,
    size_t row_partition,
    size_t* workspace_bytes) {
    size_t row_workspace;
    size_t tile_count;
    size_t float_count;
    size_t float_bytes;
    size_t quantized_bytes;
    if (workspace_bytes == NULL || width == 0u || row_partition == 0u ||
        row_partition > SIZE_MAX / width ||
        nf_thomas_rows_f32_workspace_floats_v1(
            width, row_partition, &row_workspace) != NF_THOMAS_ROWS_F32_OK_V1) {
        return NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1;
    }
    tile_count = row_partition * width;
    if (tile_count > (SIZE_MAX - row_workspace) / 8u) {
        return NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1;
    }
    float_count = row_workspace + 8u * tile_count;
    if (float_count > SIZE_MAX / sizeof(float) ||
        tile_count > SIZE_MAX / (3u * sizeof(uint16_t))) {
        return NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1;
    }
    float_bytes = float_count * sizeof(float);
    quantized_bytes = 3u * tile_count * sizeof(uint16_t);
    if (float_bytes > SIZE_MAX - quantized_bytes) {
        return NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_bytes = float_bytes + quantized_bytes;
    return NF_THOMAS_RGB16_F32_OK_V1;
}

nf_thomas_rgb16_f32_status_v1 nf_thomas_rgb16_f32_apply_v1(
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile,
    const nf_thomas_field_f32_profile_v1 field_profiles[3],
    const nf_neutral_gauge_f32_profile_v1* gauge_profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    const float* relative_log_exposure_chw,
    size_t exposure_floats,
    void* workspace,
    size_t workspace_bytes,
    nf_thomas_rgb16_f32_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]) {
    size_t required_bytes;
    size_t row_workspace;
    size_t full_count;
    size_t tile_count;
    size_t row_start;
    size_t channel;
    size_t index;
    double means[3];
    float* floats;
    float* density;
    float* sigma;
    float* transmittance;
    float* interleaved;
    uint16_t* quantized;
    if (nf_granularity_amplitude_f32_validate_profile_v1(amplitude_profile) !=
            NF_GRANULARITY_AMPLITUDE_F32_OK_V1 || field_profiles == NULL ||
        nf_neutral_gauge_f32_validate_profile_v1(gauge_profile) !=
            NF_NEUTRAL_GAUGE_F32_OK_V1) {
        return NF_THOMAS_RGB16_F32_INVALID_PROFILE_V1;
    }
    for (channel = 0u; channel < 3u; ++channel) {
        if (nf_thomas_field_f32_validate_profile_v1(&field_profiles[channel]) !=
                NF_THOMAS_FIELD_F32_OK_V1) {
            return NF_THOMAS_RGB16_F32_INVALID_PROFILE_V1;
        }
    }
    if (full_height == 0u || width == 0u || row_partition == 0u ||
        row_partition > full_height || full_height > SIZE_MAX / width ||
        relative_log_exposure_chw == NULL || workspace == NULL || sink == NULL ||
        raw_field_means == NULL || (uintptr_t)workspace % _Alignof(float) != 0u ||
        nf_thomas_rgb16_f32_workspace_bytes_v1(
            width, row_partition, &required_bytes) != NF_THOMAS_RGB16_F32_OK_V1 ||
        nf_thomas_rows_f32_workspace_floats_v1(
            width, row_partition, &row_workspace) != NF_THOMAS_ROWS_F32_OK_V1) {
        return NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1;
    }
    full_count = full_height * width;
    if (full_count > SIZE_MAX / (3u * sizeof(float)) ||
        exposure_floats < 3u * full_count ||
        workspace_bytes < required_bytes ||
        nf_ranges_overlap_bytes(
            relative_log_exposure_chw,
            3u * full_count * sizeof(float),
            workspace,
            required_bytes)) {
        return NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1;
    }
    for (channel = 0u; channel < 3u; ++channel) {
        const uint32_t knot_count = amplitude_profile->knot_count[channel];
        const double lower = amplitude_profile->log_exposure_knots[channel][0];
        const double upper =
            amplitude_profile->log_exposure_knots[channel][knot_count - 1u];
        for (index = 0u; index < full_count; ++index) {
            const double value =
                (double)relative_log_exposure_chw[channel * full_count + index];
            if (!isfinite(value) || value < lower || value > upper) {
                return NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1;
            }
        }
    }
    tile_count = row_partition * width;
    floats = (float*)workspace;
    density = floats + row_workspace;
    sigma = density + tile_count;
    transmittance = sigma + tile_count;
    interleaved = transmittance + 3u * tile_count;
    quantized = (uint16_t*)(interleaved + 3u * tile_count);
    for (channel = 0u; channel < 3u; ++channel) {
        double total = 0.0;
        double compensation = 0.0;
        for (row_start = 0u; row_start < full_height; row_start += row_partition) {
            const size_t row_count = row_partition < full_height - row_start ?
                row_partition : full_height - row_start;
            const size_t count = row_count * width;
            const nf_thomas_rows_f32_status_v1 status = nf_thomas_rows_f32_field_v1(
                &field_profiles[channel], full_height, width, row_start, row_count,
                floats, row_workspace, density, tile_count);
            if (status != NF_THOMAS_ROWS_F32_OK_V1 ||
                nf_neumaier_f32_accumulate_v1(
                    density, count, &total, &compensation) != 1) {
                return NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1;
            }
        }
        means[channel] = (total + compensation) / (double)full_count;
        if (!isfinite(means[channel])) {
            return NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1;
        }
    }
    for (row_start = 0u; row_start < full_height; row_start += row_partition) {
        const size_t row_count = row_partition < full_height - row_start ?
            row_partition : full_height - row_start;
        const size_t count = row_count * width;
        const size_t offset = row_start * width;
        for (channel = 0u; channel < 3u; ++channel) {
            nf_granularity_amplitude_f32_status_v1 amplitude_status;
            nf_thomas_rows_f32_status_v1 row_status;
            amplitude_status = nf_granularity_amplitude_f32_apply_layer_v1(
                amplitude_profile,
                (uint32_t)channel,
                relative_log_exposure_chw + channel * full_count + offset,
                count,
                density,
                sigma);
            if (amplitude_status != NF_GRANULARITY_AMPLITUDE_F32_OK_V1) {
                return NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1;
            }
            row_status = nf_thomas_rows_f32_density_v1(
                &field_profiles[channel], full_height, width, row_start, row_count,
                density, count, sigma, count, means[channel],
                floats, row_workspace, transmittance + channel * tile_count,
                tile_count);
            if (row_status != NF_THOMAS_ROWS_F32_OK_V1) {
                return row_status == NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1 ?
                    NF_THOMAS_RGB16_F32_INVALID_PROFILE_V1 :
                    NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1;
            }
        }
        for (index = 0u; index < count; ++index) {
            for (channel = 0u; channel < 3u; ++channel) {
                interleaved[3u * index + channel] =
                    transmittance[channel * tile_count + index];
            }
        }
        if (nf_neutral_gauge_f32_apply_v1(
                gauge_profile, interleaved, count, interleaved) !=
                NF_NEUTRAL_GAUGE_F32_OK_V1 ||
            nf_srgb_oetf_quantize_apply_v1(
                interleaved, 3u * count, 16u, quantized, 3u * count) != 1) {
            return NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1;
        }
        if (sink(
                sink_context, row_start, row_count, quantized, 3u * count) == 0) {
            return NF_THOMAS_RGB16_F32_CALLBACK_FAILED_V1;
        }
    }
    for (channel = 0u; channel < 3u; ++channel) {
        raw_field_means[channel] = means[channel];
    }
    return NF_THOMAS_RGB16_F32_OK_V1;
}
