#include "nf_conditioned_cloud_row_chain_f32_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double scale_value(size_t y, size_t x, size_t channel) {
    const size_t base = (x * 37u + y * 101u + 11u) % 1001u;
    const size_t offset = channel == 0u ? 0u : (channel == 1u ? 211u : 419u);
    return (double)((base + offset) % 1001u) / 1000.0;
}

int main(int argc, char** argv) {
    const size_t full = 73u, core = 31u, width = 47u, origin = 61u, halo = 9u;
    const size_t full_n = full * width * 3u;
    const size_t output_n = core * width * 3u;
    const size_t first = (origin + full - halo) % full;
    size_t count_n = 0u, conv_n = 0u, scratch_n = 0u, i, y, x, c;
    nf_density_conditioned_poisson_u16_profile_v3 cp = {
        sizeof(cp), 3u, {192., 288., 240.}, 32., {32., 16., 24.}, 78277u, 1009u};
    nf_cloud_spatial_response_f32_profile_v2 sp = {
        sizeof(sp), 2u, {1.3, 1.7, 2.1}, {.00125, .001125, .001375}, 4.};
    const float gain[3] = {.3f, .35f, .25f};
    double *full_scale, *window_scale, *conv;
    float *expected, *sd, *st, *full_density, *full_trans, *window_density,
        *window_trans;
    uint16_t* counts;
    int full_status, window_status, bad;
    FILE *a, *b;
    if (argc != 3 || nf_conditioned_cloud_row_chain_f32_workspace_v1(
                         core, width, halo, &count_n, &conv_n, &scratch_n) != 0)
        return 2;
    full_scale = malloc(full_n * sizeof(double));
    window_scale = malloc(count_n * sizeof(double));
    expected = malloc(output_n * sizeof(float));
    counts = malloc(count_n * sizeof(uint16_t));
    conv = malloc(conv_n * sizeof(double));
    sd = malloc(scratch_n * sizeof(float));
    st = malloc(scratch_n * sizeof(float));
    full_density = malloc(output_n * sizeof(float));
    full_trans = malloc(output_n * sizeof(float));
    window_density = malloc(output_n * sizeof(float));
    window_trans = malloc(output_n * sizeof(float));
    if (!full_scale || !window_scale || !expected || !counts || !conv || !sd ||
        !st || !full_density || !full_trans || !window_density || !window_trans)
        return 3;
    for (y = 0u; y < full; ++y)
        for (x = 0u; x < width; ++x)
            for (c = 0u; c < 3u; ++c)
                full_scale[(y * width + x) * 3u + c] = scale_value(y, x, c);
    for (y = 0u; y < core + 2u * halo; ++y) {
        const size_t logical = (first + y) % full;
        for (x = 0u; x < width; ++x)
            for (c = 0u; c < 3u; ++c)
                window_scale[(y * width + x) * 3u + c] =
                    scale_value(logical, x, c);
    }
    for (i = 0u; i < output_n; ++i)
        expected[i] = .2f + (float)((i * 19u + 7u) % 601u) / 1000.f;
    full_status = nf_conditioned_cloud_row_chain_f32_apply_v2(
        &cp, &sp, full, width, origin, core, halo, full_scale, full_n, expected,
        output_n, gain, counts, count_n, conv, conv_n, sd, st, scratch_n,
        full_density, full_trans, output_n);
    window_status = nf_conditioned_cloud_row_chain_f32_apply_window_v3(
        &cp, &sp, full, width, first, core, halo, window_scale, count_n,
        expected, output_n, gain, counts, count_n, conv, conv_n, sd, st,
        scratch_n, window_density, window_trans, output_n);
    if (full_status != 0 || window_status != 0 ||
        memcmp(full_density, window_density, output_n * sizeof(float)) != 0 ||
        memcmp(full_trans, window_trans, output_n * sizeof(float)) != 0)
        return 4;
    for (i = 0u; i < output_n; ++i) {
        window_density[i] = -77.f;
        window_trans[i] = -77.f;
    }
    window_scale[0] = 0.0 / 0.0;
    bad = nf_conditioned_cloud_row_chain_f32_apply_window_v3(
        &cp, &sp, full, width, first, core, halo, window_scale, count_n,
        expected, output_n, gain, counts, count_n, conv, conv_n, sd, st,
        scratch_n, window_density, window_trans, output_n);
    for (i = 0u; i < output_n; ++i)
        if (window_density[i] != -77.f || window_trans[i] != -77.f) return 5;
    a = fopen(argv[1], "wb");
    b = fopen(argv[2], "wb");
    if (!a || !b || fwrite(full_density, sizeof(float), output_n, a) != output_n ||
        fwrite(full_trans, sizeof(float), output_n, b) != output_n)
        return 6;
    fclose(a);
    fclose(b);
    printf("status=%d window=%d invalid=%d full_window=1 counts=%zu convolution=%zu core=%zu\n",
           full_status, window_status, bad, count_n, conv_n, scratch_n);
    free(full_scale); free(window_scale); free(expected); free(counts); free(conv);
    free(sd); free(st); free(full_density); free(full_trans); free(window_density);
    free(window_trans);
    return bad != 0 ? 0 : 7;
}
