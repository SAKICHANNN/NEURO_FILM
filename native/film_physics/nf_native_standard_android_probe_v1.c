#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nf_ao6_context_f32_v2.h"
#include "nf_ao6_display_f32_v4.h"
#include "nf_bounded_adjacency_f32_v1.h"
#include "nf_gaussian_rgb_f32_v1.h"
#include "nf_neutral_gauge_f32_v1.h"
#include "nf_physical_domains_f32_v1.h"

enum { NF_HEIGHT = 64, NF_WIDTH = 96, NF_PIXELS = NF_HEIGHT * NF_WIDTH };

static int read_exact(const char* path, void* output, size_t bytes) {
    FILE* stream = fopen(path, "rb");
    size_t read_bytes;
    if (stream == NULL) return 0;
    read_bytes = fread(output, 1u, bytes, stream);
    if (read_bytes != bytes || fgetc(stream) != EOF || fclose(stream) != 0) return 0;
    return 1;
}

static int write_exact(const char* path, const void* values, size_t bytes) {
    FILE* stream = fopen(path, "wb");
    if (stream == NULL) return 0;
    if (fwrite(values, 1u, bytes, stream) != bytes || fclose(stream) != 0) return 0;
    return 1;
}

static int load_blob(const char* root, const char* name, void* output, size_t bytes) {
    char path[512];
    if (snprintf(path, sizeof(path), "%s/%s", root, name) <= 0) return 0;
    return read_exact(path, output, bytes);
}

static float srgb_encode(float input) {
    const double value = (double)input;
    if (value <= 0.0031308) return (float)(12.92 * value);
    return (float)(1.055 * pow(value, 1.0 / 2.4) - 0.055);
}

static int blur(
    const nf_gaussian_f32_profile_v1* profile,
    const float* input,
    float* workspace,
    float* output
) {
    return nf_gaussian_f32_apply_v1(
        profile, input, NF_HEIGHT, NF_WIDTH, workspace, NF_PIXELS * 3u, output
    ) == NF_GAUSSIAN_F32_OK_V1;
}

int main(int argc, char** argv) {
    const size_t samples = (size_t)NF_PIXELS * 3u;
    const size_t bytes = samples * sizeof(float);
    const char* root;
    const char* output_path;
    float *source, *source_copy, *a, *b, *c, *workspace, *encoded, *scratch, *output;
    nf_physical_domains_f32_profile_v1 domains;
    nf_gaussian_f32_profile_v1 forward, adjacency_blur, diffusion, scanner;
    nf_bounded_adjacency_f32_profile_v1 adjacency;
    nf_neutral_gauge_f32_profile_v1 gauge;
    nf_ao6_base_f32_profile_v1 base;
    nf_ao6_residual_f32_profile_v1 residual;
    nf_ao6_context_f32_state_v1 state;
    nf_ao6_base_f32_context_v1 context;
    int source_unchanged = 0;
    int invalid_atomic = 0;
    size_t index;

    if (argc != 3) return 2;
    root = argv[1];
    output_path = argv[2];
    source = (float*)malloc(bytes);
    source_copy = (float*)malloc(bytes);
    a = (float*)malloc(bytes);
    b = (float*)malloc(bytes);
    c = (float*)malloc(bytes);
    workspace = (float*)malloc(bytes);
    encoded = (float*)malloc(bytes);
    scratch = (float*)malloc(bytes);
    output = (float*)malloc(bytes);
    if (!source || !source_copy || !a || !b || !c || !workspace || !encoded || !scratch || !output) return 3;
    if (
        !load_blob(root, "source.f32", source, bytes) ||
        !load_blob(root, "domains.bin", &domains, sizeof(domains)) ||
        !load_blob(root, "gaussian_forward.bin", &forward, sizeof(forward)) ||
        !load_blob(root, "gaussian_adjacency.bin", &adjacency_blur, sizeof(adjacency_blur)) ||
        !load_blob(root, "gaussian_diffusion.bin", &diffusion, sizeof(diffusion)) ||
        !load_blob(root, "gaussian_scanner.bin", &scanner, sizeof(scanner)) ||
        !load_blob(root, "adjacency.bin", &adjacency, sizeof(adjacency)) ||
        !load_blob(root, "gauge.bin", &gauge, sizeof(gauge)) ||
        !load_blob(root, "base.bin", &base, sizeof(base)) ||
        !load_blob(root, "residual.bin", &residual, sizeof(residual))
    ) return 4;
    memcpy(source_copy, source, bytes);

    for (index = 0; index < samples; ++index) encoded[index] = srgb_encode(source[index]);
    if (nf_ao6_context_f32_init_v1(&state) != NF_AO6_CONTEXT_F32_OK_V1) return 5;
    if (nf_ao6_context_f32_update_v2(&base, &state, encoded, NF_PIXELS, scratch) != NF_AO6_CONTEXT_F32_OK_V1) return 6;
    if (nf_ao6_context_f32_finalize_v1(&state, &context) != NF_AO6_CONTEXT_F32_OK_V1) return 7;

    if (!blur(&forward, source, workspace, a)) return 8;
    if (nf_physical_sensitometry_f32_apply_v1(&domains, a, NF_PIXELS, b) != NF_PHYSICAL_DOMAINS_F32_OK_V1) return 9;
    if (!blur(&adjacency_blur, b, workspace, c)) return 10;
    if (nf_bounded_adjacency_f32_apply_v1(&adjacency, b, c, NF_PIXELS, a) != NF_BOUNDED_ADJACENCY_F32_OK_V1) return 11;
    if (!blur(&diffusion, a, workspace, b)) return 12;
    if (nf_physical_interpretation_f32_apply_v1(&domains, b, NF_PIXELS, c) != NF_PHYSICAL_DOMAINS_F32_OK_V1) return 13;
    if (!blur(&scanner, c, workspace, a)) return 14;
    if (nf_neutral_gauge_f32_apply_v1(&gauge, a, NF_PIXELS, b) != NF_NEUTRAL_GAUGE_F32_OK_V1) return 15;
    for (index = 0; index < samples; ++index) encoded[index] = srgb_encode(b[index]);
    if (nf_ao6_display_f32_apply_v4(&base, &context, &residual, encoded, NF_PIXELS, scratch, output) != NF_AO6_DISPLAY_F32_OK_V1) return 16;
    source_unchanged = memcmp(source, source_copy, bytes) == 0;

    for (index = 0; index < samples; ++index) c[index] = 0.375f;
    a[0] = NAN;
    invalid_atomic = (
        nf_physical_sensitometry_f32_apply_v1(&domains, a, NF_PIXELS, c) == NF_PHYSICAL_DOMAINS_F32_INVALID_INPUT_V1
    );
    for (index = 0; index < samples; ++index) invalid_atomic = invalid_atomic && c[index] == 0.375f;
    if (!write_exact(output_path, output, bytes)) return 17;
    printf("source_unchanged=%d invalid_atomic=%d samples=%zu\n", source_unchanged, invalid_atomic, samples);
    free(output); free(scratch); free(encoded); free(workspace); free(c); free(b); free(a); free(source_copy); free(source);
    return (source_unchanged && invalid_atomic) ? 0 : 18;
}
