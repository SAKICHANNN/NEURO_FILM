#define NF_HISTOGRAM_COPULA_F32_BUILD
#define NF_THOMAS_FIELD_F32_BUILD

#include "nf_gamma_density_fast_f64_v1.h"
#include "nf_histogram_copula_f32_v1.h"
#include "nf_thomas_field_f32_v1.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEIGHT 47u
#define WIDTH 53u
#define SAMPLES (HEIGHT * WIDTH)
#define VALUES (SAMPLES * 3u)

static uint64_t hash_bytes(const void *data, size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t value = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0u; index < size; ++index) {
        value ^= bytes[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

int main(void) {
    const uint64_t realization[3] = {
        UINT64_C(2608024301), UINT64_C(11400714821016565496),
        UINT64_C(15111065704576690158)};
    const double correlation[9] = {
        1.0, 0.2721655269759087, 0.22360679774997896,
        0.2721655269759087, 1.0, 0.21300321680756462,
        0.22360679774997896, 0.21300321680756462, 1.0};
    float *field_workspace = NULL;
    float *channel_field = NULL;
    float *fields = NULL;
    float *uniforms_f32 = NULL;
    void *copula_workspace = NULL;
    double *uniforms = NULL;
    double *shapes = NULL;
    double *scales = NULL;
    double *density = NULL;
    size_t field_workspace_floats = 0u;
    size_t copula_workspace_bytes = 0u;
    size_t sample;
    size_t channel;
    int ok = 0;
    nf_histogram_copula_f32_diagnostics_v1 copula_diag;
    nf_gamma_density_fast_diagnostics_v1 gamma_diag;

    if (nf_thomas_field_f32_abi_version_v1() != 1u ||
        nf_histogram_copula_f32_abi_version_v1() != 1u ||
        nf_gamma_density_fast_f64_abi_version_v1() != 1u ||
        nf_thomas_field_f32_workspace_floats_v1(
            HEIGHT, WIDTH, &field_workspace_floats) != 0 ||
        nf_histogram_copula_f32_workspace_bytes_v1(
            SAMPLES, 2048u, &copula_workspace_bytes) != 0) {
        return 2;
    }
    field_workspace = (float *)malloc(field_workspace_floats * sizeof(float));
    channel_field = (float *)malloc(SAMPLES * sizeof(float));
    fields = (float *)malloc(VALUES * sizeof(float));
    uniforms_f32 = (float *)malloc(VALUES * sizeof(float));
    copula_workspace = malloc(copula_workspace_bytes);
    uniforms = (double *)malloc(VALUES * sizeof(double));
    shapes = (double *)malloc(VALUES * sizeof(double));
    scales = (double *)malloc(VALUES * sizeof(double));
    density = (double *)malloc(VALUES * sizeof(double));
    if (field_workspace == NULL || channel_field == NULL || fields == NULL ||
        uniforms_f32 == NULL || copula_workspace == NULL || uniforms == NULL ||
        shapes == NULL || scales == NULL || density == NULL) {
        goto cleanup;
    }
    for (channel = 0u; channel < 3u; ++channel) {
        nf_thomas_field_f32_profile_v1 profile;
        double raw_mean = 0.0;
        memset(&profile, 0, sizeof(profile));
        profile.struct_size = (uint32_t)sizeof(profile);
        profile.abi_version = 1u;
        profile.particle_sigma_pixels = 1.3253699417769098;
        profile.cluster_sigma_pixels = 1.1268619873805532;
        profile.mean_offspring = 27.765942352935905;
        profile.truncate = 4.0;
        profile.component_seeds[0] = UINT64_C(2611923443488327891);
        profile.component_seeds[1] = UINT64_C(11400714819323198485);
        profile.realization_seed = realization[channel];
        if (nf_thomas_field_f32_apply_v1(
                &profile, HEIGHT, WIDTH, field_workspace,
                field_workspace_floats, channel_field, SAMPLES,
                &raw_mean) != 0) {
            goto cleanup;
        }
        for (sample = 0u; sample < SAMPLES; ++sample) {
            fields[3u * sample + channel] = channel_field[sample];
        }
    }
    memset(&copula_diag, 0, sizeof(copula_diag));
    copula_diag.struct_size = sizeof(copula_diag);
    copula_diag.abi_version = 1u;
    if (nf_histogram_copula_f32_apply_v1(
            fields, SAMPLES, correlation, 2048u, copula_workspace,
            copula_workspace_bytes, uniforms_f32, VALUES, &copula_diag) != 0) {
        goto cleanup;
    }
    for (sample = 0u; sample < VALUES; ++sample) {
        uniforms[sample] = (double)uniforms_f32[sample];
        shapes[sample] = sample % 3u == 0u ? 20.0 :
                         (sample % 3u == 1u ? 100.0 : 100000.0);
        scales[sample] = 0.000001 + (double)((sample * 17u) % 1000u) * 0.000001;
    }
    memset(&gamma_diag, 0, sizeof(gamma_diag));
    gamma_diag.struct_size = sizeof(gamma_diag);
    gamma_diag.abi_version = 1u;
    if (nf_gamma_density_fast_f64_apply_v1(
            uniforms, shapes, scales, VALUES, 80u, 6u, 50.0, 10000.0,
            density, VALUES, &gamma_diag) != 0) {
        goto cleanup;
    }
    {
        double bad_uniform = 1.0;
        double bad_shape = 100.0;
        double bad_scale = 0.01;
        double sentinel = -19.0;
        nf_gamma_density_fast_diagnostics_v1 sentinel_diag;
        unsigned char before[sizeof(sentinel_diag)];
        memset(&sentinel_diag, 0x5a, sizeof(sentinel_diag));
        memcpy(before, &sentinel_diag, sizeof(before));
        if (nf_gamma_density_fast_f64_apply_v1(
                &bad_uniform, &bad_shape, &bad_scale, 1u, 80u, 6u, 50.0,
                10000.0, &sentinel, 1u, &sentinel_diag) == 0 ||
            sentinel != -19.0 || memcmp(before, &sentinel_diag, sizeof(before)) != 0) {
            goto cleanup;
        }
    }
    printf(
        "field=%016llx uniforms=%016llx density=%016llx "
        "branches=%llu,%llu,%llu atomic=1\n",
        (unsigned long long)hash_bytes(fields, VALUES * sizeof(float)),
        (unsigned long long)hash_bytes(uniforms_f32, VALUES * sizeof(float)),
        (unsigned long long)hash_bytes(density, VALUES * sizeof(double)),
        (unsigned long long)gamma_diag.direct_branch_count,
        (unsigned long long)gamma_diag.newton_branch_count,
        (unsigned long long)gamma_diag.asymptotic_branch_count);
    ok = 1;

cleanup:
    free(density);
    free(scales);
    free(shapes);
    free(uniforms);
    free(copula_workspace);
    free(uniforms_f32);
    free(fields);
    free(channel_field);
    free(field_workspace);
    return ok ? 0 : 3;
}
