#define NF_THOMAS_RGB16_CACHED_F32_BUILD
#if defined(_WIN32)
#define NF_THOMAS_RGB16_F32_BUILD
#endif
#include "nf_thomas_rgb16_cached_f32_v1.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct nf_raw_sink_v1 {
    FILE* file;
    size_t calls;
    size_t values;
} nf_raw_sink_v1;

static int nf_raw_sink(
    void* context,
    size_t row_start,
    size_t row_count,
    const uint16_t* values,
    size_t value_count) {
    nf_raw_sink_v1* sink = (nf_raw_sink_v1*)context;
    (void)row_start;
    (void)row_count;
    if (sink->file != NULL &&
        fwrite(values, sizeof(uint16_t), value_count, sink->file) != value_count) {
        return 0;
    }
    sink->calls += 1u;
    sink->values += value_count;
    return 1;
}

static void nf_hex_identity(char output[65], char digit) {
    size_t index;
    for (index = 0u; index < 64u; ++index) {
        output[index] = digit;
    }
    output[64] = '\0';
}

static void nf_build_profiles(
    nf_granularity_amplitude_f32_profile_v1* amplitude,
    nf_thomas_field_f32_profile_v1 fields[3],
    nf_neutral_gauge_f32_profile_v1* gauge) {
    static const uint64_t component_seeds[3][2] = {
        {UINT64_C(17), UINT64_C(29)},
        {UINT64_C(37), UINT64_C(43)},
        {UINT64_C(53), UINT64_C(61)},
    };
    static const uint64_t realization_seeds[3] = {
        UINT64_C(101), UINT64_C(103), UINT64_C(107)};
    size_t channel;
    memset(amplitude, 0, sizeof(*amplitude));
    memset(fields, 0, 3u * sizeof(*fields));
    memset(gauge, 0, sizeof(*gauge));
    amplitude->struct_size = (uint32_t)sizeof(*amplitude);
    amplitude->abi_version = NF_GRANULARITY_AMPLITUDE_F32_ABI_VERSION_V1;
    amplitude->shared_amplitude = 0.000001;
    amplitude->measurement_energy = 0.75;
    nf_hex_identity(amplitude->source_profile_sha256, '0');
    gauge->struct_size = (uint32_t)sizeof(*gauge);
    gauge->abi_version = NF_NEUTRAL_GAUGE_F32_ABI_VERSION_V1;
    nf_hex_identity(gauge->source_component_sha256, '1');
    for (channel = 0u; channel < 3u; ++channel) {
        amplitude->knot_count[channel] = 3u;
        amplitude->log_exposure_knots[channel][0] = -2.0;
        amplitude->log_exposure_knots[channel][1] = 0.0;
        amplitude->log_exposure_knots[channel][2] = 2.0;
        amplitude->density_knots[channel][0] = 0.3;
        amplitude->density_knots[channel][1] = 0.6;
        amplitude->density_knots[channel][2] = 0.9;
        amplitude->channel_floor_variance[channel] =
            0.00000001 + 0.000000002 * channel;
        fields[channel].struct_size = (uint32_t)sizeof(fields[channel]);
        fields[channel].abi_version = NF_THOMAS_FIELD_F32_ABI_VERSION_V1;
        fields[channel].particle_sigma_pixels = 0.8;
        fields[channel].cluster_sigma_pixels = 1.7;
        fields[channel].mean_offspring = 3.0;
        fields[channel].truncate = 3.0;
        fields[channel].component_seeds[0] = component_seeds[channel][0];
        fields[channel].component_seeds[1] = component_seeds[channel][1];
        fields[channel].realization_seed = realization_seeds[channel];
        gauge->knot_count[channel] = 2u;
        gauge->x_knots[channel][0] = 0.0;
        gauge->x_knots[channel][1] = 1.0;
        gauge->y_knots[channel][0] = 0.0;
        gauge->y_knots[channel][1] = 1.0;
        gauge->derivatives[channel][0] = 1.0;
        gauge->derivatives[channel][1] = 1.0;
    }
}

static void nf_build_exposure(float* exposure, size_t sample_count) {
    size_t channel;
    size_t index;
    for (channel = 0u; channel < 3u; ++channel) {
        for (index = 0u; index < sample_count; ++index) {
            const int32_t code =
                (int32_t)((index * 17u + channel * 13u) % 193u) - 96;
            exposure[channel * sample_count + index] = (float)code / 64.0f;
        }
    }
}

static int nf_parse_size(const char* text, size_t minimum, size_t* output) {
    char* end = NULL;
    unsigned long long value;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < minimum ||
        value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *output = (size_t)value;
    return 1;
}

int main(int argc, char** argv) {
    const int cached = argc == 7 && strcmp(argv[1], "cached") == 0;
    const int legacy = argc == 7 && strcmp(argv[1], "legacy") == 0;
    size_t height;
    size_t width;
    size_t row_partition;
    size_t parallel_layers;
    size_t sample_count;
    size_t workspace_bytes = 0u;
    nf_granularity_amplitude_f32_profile_v1 amplitude;
    nf_thomas_field_f32_profile_v1 fields[3];
    nf_neutral_gauge_f32_profile_v1 gauge;
    float* exposure = NULL;
    void* workspace = NULL;
    FILE* output = NULL;
    nf_raw_sink_v1 sink = {0};
    nf_raw_sink_v1 invalid_sink = {0};
    double means[3] = {-13.0, -13.0, -13.0};
    double invalid_means[3] = {-13.0, -13.0, -13.0};
    int status;
    int invalid_status;
    int result = 1;
    if ((!cached && !legacy) || !nf_parse_size(argv[3], 1u, &height) ||
        !nf_parse_size(argv[4], 1u, &width) ||
        !nf_parse_size(argv[5], 1u, &row_partition) ||
        !nf_parse_size(argv[6], 1u, &parallel_layers) ||
        row_partition > height || (parallel_layers != 1u && parallel_layers != 3u) ||
        height > SIZE_MAX / width) {
        return 2;
    }
    sample_count = height * width;
    if (sample_count > SIZE_MAX / 3u ||
        3u * sample_count > SIZE_MAX / sizeof(float)) {
        return 2;
    }
    nf_build_profiles(&amplitude, fields, &gauge);
    exposure = (float*)malloc(3u * sample_count * sizeof(float));
    if (exposure == NULL) {
        goto cleanup;
    }
    nf_build_exposure(exposure, sample_count);
    if ((cached && nf_thomas_rgb16_cached_f32_workspace_bytes_v1(
            height, width, row_partition, (uint32_t)parallel_layers,
            &workspace_bytes) != NF_THOMAS_RGB16_CACHED_F32_OK_V1) ||
        (legacy && nf_thomas_rgb16_f32_workspace_bytes_v1(
            width, row_partition, &workspace_bytes) != NF_THOMAS_RGB16_F32_OK_V1)) {
        goto cleanup;
    }
    workspace = malloc(workspace_bytes);
    output = fopen(argv[2], "wb");
    if (workspace == NULL || output == NULL) {
        goto cleanup;
    }
    sink.file = output;
    if (cached) {
        status = (int)nf_thomas_rgb16_cached_f32_apply_v1(
            &amplitude, fields, &gauge, height, width, row_partition,
            (uint32_t)parallel_layers, exposure, 3u * sample_count,
            workspace, workspace_bytes, nf_raw_sink, &sink, means);
    } else {
        status = (int)nf_thomas_rgb16_f32_apply_v1(
            &amplitude, fields, &gauge, height, width, row_partition,
            exposure, 3u * sample_count, workspace, workspace_bytes,
            nf_raw_sink, &sink, means);
    }
    if (fclose(output) != 0) {
        output = NULL;
        goto cleanup;
    }
    output = NULL;
    if (status != 0) {
        goto cleanup;
    }
    exposure[3u * sample_count - 1u] = NAN;
    if (cached) {
        invalid_status = (int)nf_thomas_rgb16_cached_f32_apply_v1(
            &amplitude, fields, &gauge, height, width, row_partition,
            (uint32_t)parallel_layers, exposure, 3u * sample_count,
            workspace, workspace_bytes, nf_raw_sink, &invalid_sink,
            invalid_means);
    } else {
        invalid_status = (int)nf_thomas_rgb16_f32_apply_v1(
            &amplitude, fields, &gauge, height, width, row_partition,
            exposure, 3u * sample_count, workspace, workspace_bytes,
            nf_raw_sink, &invalid_sink, invalid_means);
    }
    printf(
        "mode=%s status=%d height=%zu width=%zu rows=%zu parallel=%zu "
        "values=%zu calls=%zu workspace=%zu means=%a,%a,%a "
        "invalid_status=%d invalid_calls=%zu invalid_means=%a,%a,%a\n",
        cached ? "cached" : "legacy", status, height, width, row_partition,
        parallel_layers, sink.values, sink.calls, workspace_bytes,
        means[0], means[1], means[2], invalid_status, invalid_sink.calls,
        invalid_means[0], invalid_means[1], invalid_means[2]);
    result = 0;

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    free(workspace);
    free(exposure);
    return result;
}
