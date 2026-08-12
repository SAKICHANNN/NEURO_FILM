#include "nf_gaussian_row_window_f32_v1.h"

#include <stdio.h>
#include <string.h>

#define HEIGHT 73u
#define WIDTH 47u
#define VALUES (HEIGHT * WIDTH * 3u)

static float source[VALUES];
static float workspace[VALUES];
static float full_output[VALUES];
static float window_output[VALUES];
static float core_output[VALUES];
static float assembled[VALUES];

static int run_window(
    const nf_gaussian_f32_profile_v1* profile,
    size_t start,
    size_t end) {
    uint32_t halo = 0u;
    size_t input_start;
    size_t input_end;
    size_t input_height;
    size_t count = end - start;
    size_t source_offset;
    nf_gaussian_f32_status_v1 status;
    if (nf_gaussian_f32_required_halo_v1(profile, &halo) != 0) {
        return 0;
    }
    input_start = start < halo ? 0u : start - halo;
    input_end = end + halo > HEIGHT ? HEIGHT : end + halo;
    input_height = input_end - input_start;
    source_offset = input_start * WIDTH * 3u;
    status = nf_gaussian_row_window_f32_apply_v1(
        profile, HEIGHT, WIDTH, input_start, input_height,
        source + source_offset, start, count, workspace,
        input_height * WIDTH * 3u, window_output,
        input_height * WIDTH * 3u, core_output, count * WIDTH * 3u);
    if (status != NF_GAUSSIAN_F32_OK_V1) {
        return 0;
    }
    memcpy(
        assembled + start * WIDTH * 3u,
        core_output,
        count * WIDTH * 3u * sizeof(float));
    return 1;
}

int main(int argc, char** argv) {
    nf_gaussian_f32_profile_v1 profile;
    FILE* handle;
    size_t index;
    int invalid;
    if (argc != 2) {
        return 2;
    }
    memset(&profile, 0, sizeof(profile));
    profile.struct_size = (uint32_t)sizeof(profile);
    profile.abi_version = NF_GAUSSIAN_F32_ABI_VERSION_V1;
    memcpy(
        profile.source_component_sha256,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        65u);
    profile.sigma_pixels_rgb[0] = 0.8;
    profile.sigma_pixels_rgb[1] = 1.0;
    profile.sigma_pixels_rgb[2] = 1.2;
    profile.truncate = 3.0;
    for (index = 0u; index < VALUES; ++index) {
        size_t pixel = index / 3u;
        size_t channel = index % 3u;
        size_t y = pixel / WIDTH;
        size_t x = pixel % WIDTH;
        source[index] = (float)((y * 17u + x * 31u + channel * 101u + 3u) % 997u) / 996.0f;
    }
    if (nf_gaussian_f32_apply_v1(
            &profile, source, HEIGHT, WIDTH, workspace, VALUES, full_output) != 0
        || !run_window(&profile, 0u, 11u)
        || !run_window(&profile, 11u, 37u)
        || !run_window(&profile, 37u, HEIGHT)
        || memcmp(full_output, assembled, sizeof(full_output)) != 0) {
        return 3;
    }
    for (index = 0u; index < VALUES; ++index) {
        core_output[index] = -77.0f;
    }
    invalid = nf_gaussian_row_window_f32_apply_v1(
        &profile, HEIGHT, WIDTH, 3u, 20u, source, 11u, 10u,
        workspace, VALUES, window_output, VALUES, core_output, VALUES);
    if (invalid == 0 || core_output[0] != -77.0f) {
        return 4;
    }
    handle = fopen(argv[1], "wb");
    if (handle == NULL || fwrite(assembled, 1u, sizeof(assembled), handle) != sizeof(assembled)) {
        return 5;
    }
    if (fclose(handle) != 0) {
        return 6;
    }
    printf("abi=%u exact=1 invalid=%d\n", nf_gaussian_row_window_f32_abi_version_v1(), invalid);
    return 0;
}
