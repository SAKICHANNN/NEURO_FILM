#if defined(_WIN32)
#define NF_THOMAS_RGB16_PNG_F32_BUILD
#endif
#include "nf_thomas_rgb16_png_f32_v1.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct nf_scale_sink_v1 {
    FILE* file;
    size_t calls;
    size_t bytes;
} nf_scale_sink_v1;

static int nf_scale_sink(
    void* context,
    const uint8_t* bytes,
    size_t byte_count) {
    nf_scale_sink_v1* sink = (nf_scale_sink_v1*)context;
    if (fwrite(bytes, 1u, byte_count, sink->file) != byte_count) {
        return 0;
    }
    sink->calls += 1u;
    sink->bytes += byte_count;
    return 1;
}

static void nf_hex_identity(char output[65], char digit) {
    size_t index;
    for (index = 0u; index < 64u; ++index) {
        output[index] = digit;
    }
    output[64] = '\0';
}

static void nf_build_amplitude(
    nf_granularity_amplitude_f32_profile_v1* profile) {
    size_t channel;
    memset(profile, 0, sizeof(*profile));
    profile->struct_size = (uint32_t)sizeof(*profile);
    profile->abi_version = NF_GRANULARITY_AMPLITUDE_F32_ABI_VERSION_V1;
    nf_hex_identity(profile->source_profile_sha256, '0');
    for (channel = 0u; channel < 3u; ++channel) {
        profile->knot_count[channel] = 3u;
        profile->log_exposure_knots[channel][0] = -2.0;
        profile->log_exposure_knots[channel][1] = 0.0;
        profile->log_exposure_knots[channel][2] = 2.0;
        profile->density_knots[channel][0] = 0.3;
        profile->density_knots[channel][1] = 0.6;
        profile->density_knots[channel][2] = 0.9;
        profile->channel_floor_variance[channel] =
            0.00000001 + 0.000000002 * channel;
    }
    profile->shared_amplitude = 0.000001;
    profile->measurement_energy = 0.75;
}

static void nf_build_fields(nf_thomas_field_f32_profile_v1 profiles[3]) {
    static const uint64_t component_seeds[3][2] = {
        {UINT64_C(17), UINT64_C(29)},
        {UINT64_C(37), UINT64_C(43)},
        {UINT64_C(53), UINT64_C(61)},
    };
    static const uint64_t realization_seeds[3] = {
        UINT64_C(101), UINT64_C(103), UINT64_C(107),
    };
    size_t channel;
    memset(profiles, 0, 3u * sizeof(*profiles));
    for (channel = 0u; channel < 3u; ++channel) {
        profiles[channel].struct_size = (uint32_t)sizeof(profiles[channel]);
        profiles[channel].abi_version = NF_THOMAS_FIELD_F32_ABI_VERSION_V1;
        profiles[channel].particle_sigma_pixels = 0.8;
        profiles[channel].cluster_sigma_pixels = 1.7;
        profiles[channel].mean_offspring = 3.0;
        profiles[channel].truncate = 3.0;
        profiles[channel].component_seeds[0] = component_seeds[channel][0];
        profiles[channel].component_seeds[1] = component_seeds[channel][1];
        profiles[channel].realization_seed = realization_seeds[channel];
    }
}

static void nf_build_gauge(nf_neutral_gauge_f32_profile_v1* profile) {
    size_t channel;
    memset(profile, 0, sizeof(*profile));
    profile->struct_size = (uint32_t)sizeof(*profile);
    profile->abi_version = NF_NEUTRAL_GAUGE_F32_ABI_VERSION_V1;
    nf_hex_identity(profile->source_component_sha256, '1');
    for (channel = 0u; channel < 3u; ++channel) {
        profile->knot_count[channel] = 2u;
        profile->x_knots[channel][0] = 0.0;
        profile->x_knots[channel][1] = 1.0;
        profile->y_knots[channel][0] = 0.0;
        profile->y_knots[channel][1] = 1.0;
        profile->derivatives[channel][0] = 1.0;
        profile->derivatives[channel][1] = 1.0;
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
    size_t height;
    size_t width;
    size_t row_partition;
    size_t sample_count;
    size_t exposure_count;
    size_t workspace_bytes = 0u;
    nf_granularity_amplitude_f32_profile_v1 amplitude;
    nf_thomas_field_f32_profile_v1 fields[3];
    nf_neutral_gauge_f32_profile_v1 gauge;
    float* exposure = NULL;
    void* workspace = NULL;
    FILE* output = NULL;
    nf_scale_sink_v1 sink = {0};
    double means[3] = {-13.0, -13.0, -13.0};
    nf_thomas_rgb16_png_f32_status_v1 status;
    int result = 1;

    if (argc != 5 || !nf_parse_size(argv[2], 1u, &height) ||
        !nf_parse_size(argv[3], 1u, &width) ||
        !nf_parse_size(argv[4], 1u, &row_partition) ||
        row_partition > height || height > SIZE_MAX / width) {
        return 2;
    }
    sample_count = height * width;
    if (sample_count > SIZE_MAX / 3u ||
        3u * sample_count > SIZE_MAX / sizeof(float)) {
        return 2;
    }
    exposure_count = 3u * sample_count;
    nf_build_amplitude(&amplitude);
    nf_build_fields(fields);
    nf_build_gauge(&gauge);
    exposure = (float*)malloc(exposure_count * sizeof(float));
    if (exposure == NULL) {
        goto cleanup;
    }
    nf_build_exposure(exposure, sample_count);
    if (nf_thomas_rgb16_png_f32_workspace_bytes_v1(
            width, row_partition, &workspace_bytes) !=
        NF_THOMAS_RGB16_PNG_F32_OK_V1) {
        goto cleanup;
    }
    workspace = malloc(workspace_bytes);
    output = fopen(argv[1], "wb");
    if (workspace == NULL || output == NULL) {
        goto cleanup;
    }
    sink.file = output;
    status = nf_thomas_rgb16_png_f32_apply_v1(
        &amplitude, fields, &gauge, height, width, row_partition,
        exposure, exposure_count, workspace, workspace_bytes,
        nf_scale_sink, &sink, means);
    if (fclose(output) != 0) {
        output = NULL;
        goto cleanup;
    }
    output = NULL;
    if (status != NF_THOMAS_RGB16_PNG_F32_OK_V1) {
        goto cleanup;
    }
    printf(
        "status=%d height=%zu width=%zu rows=%zu bytes=%zu calls=%zu "
        "workspace=%zu means=%a,%a,%a\n",
        (int)status, height, width, row_partition, sink.bytes, sink.calls,
        workspace_bytes, means[0], means[1], means[2]);
    result = 0;

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    free(workspace);
    free(exposure);
    return result;
}
