#define NF_GAUSSIAN_BUILD
#include "nf_gaussian_rgb_f64_v1.h"

#include <math.h>

#define NF_GAUSSIAN_MAX_DIAMETER_V1 \
    (2u * NF_GAUSSIAN_MAX_RADIUS_V1 + 1u)

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

static uint32_t nf_radius(double sigma, double truncate) {
    const double scaled = truncate * sigma;
    if (sigma == 0.0) {
        return 0u;
    }
    if (
        !isfinite(scaled) ||
        scaled + 0.5 >= (double)(NF_GAUSSIAN_MAX_RADIUS_V1 + 1u)
    ) {
        return NF_GAUSSIAN_MAX_RADIUS_V1 + 1u;
    }
    return (uint32_t)(scaled + 0.5);
}

static int nf_profile_valid(const nf_gaussian_profile_v1* profile) {
    size_t channel;
    if (
        profile == NULL ||
        profile->struct_size != sizeof(nf_gaussian_profile_v1) ||
        profile->abi_version != NF_GAUSSIAN_ABI_VERSION_V1 ||
        !nf_is_hex_sha256(profile->source_component_sha256) ||
        !isfinite(profile->truncate) ||
        profile->truncate <= 0.0 ||
        profile->truncate > 8.0
    ) {
        return 0;
    }
    for (channel = 0; channel < 3; ++channel) {
        const double sigma = profile->sigma_pixels_rgb[channel];
        if (
            !isfinite(sigma) ||
            sigma < 0.0 ||
            nf_radius(sigma, profile->truncate) >
                NF_GAUSSIAN_MAX_RADIUS_V1
        ) {
            return 0;
        }
    }
    return 1;
}

static int nf_ranges_overlap(
    const double* first,
    const double* second,
    size_t count) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const size_t bytes = count * sizeof(double);
    const uintptr_t first_end = first_start + bytes;
    const uintptr_t second_end = second_start + bytes;
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

static size_t nf_clamp_index(
    ptrdiff_t value,
    size_t upper_exclusive) {
    if (value < 0) {
        return 0;
    }
    if ((size_t)value >= upper_exclusive) {
        return upper_exclusive - 1u;
    }
    return (size_t)value;
}

static void nf_build_weights(
    double sigma,
    uint32_t radius,
    double weights[NF_GAUSSIAN_MAX_DIAMETER_V1]) {
    uint32_t index;
    double sum = 0.0;
    if (radius == 0u) {
        weights[0] = 1.0;
        return;
    }
    for (index = 0; index <= 2u * radius; ++index) {
        const double offset = (double)((ptrdiff_t)index - radius);
        const double value = exp(-0.5 * (offset / sigma) * (offset / sigma));
        weights[index] = value;
        sum += value;
    }
    for (index = 0; index <= 2u * radius; ++index) {
        weights[index] /= sum;
    }
}

uint32_t nf_gaussian_abi_version_v1(void) {
    return NF_GAUSSIAN_ABI_VERSION_V1;
}

nf_gaussian_status_v1 nf_gaussian_validate_profile_v1(
    const nf_gaussian_profile_v1* profile) {
    if (profile == NULL) {
        return NF_GAUSSIAN_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile)
        ? NF_GAUSSIAN_OK_V1
        : NF_GAUSSIAN_INVALID_PROFILE_V1;
}

nf_gaussian_status_v1 nf_gaussian_required_halo_v1(
    const nf_gaussian_profile_v1* profile,
    uint32_t* halo) {
    uint32_t maximum = 0u;
    size_t channel;
    const nf_gaussian_status_v1 status =
        nf_gaussian_validate_profile_v1(profile);
    if (status != NF_GAUSSIAN_OK_V1) {
        return status;
    }
    if (halo == NULL) {
        return NF_GAUSSIAN_INVALID_ARGUMENT_V1;
    }
    for (channel = 0; channel < 3; ++channel) {
        const uint32_t radius = nf_radius(
            profile->sigma_pixels_rgb[channel],
            profile->truncate);
        if (radius > maximum) {
            maximum = radius;
        }
    }
    *halo = maximum;
    return NF_GAUSSIAN_OK_V1;
}

nf_gaussian_status_v1 nf_gaussian_apply_v1(
    const nf_gaussian_profile_v1* profile,
    const double* input_rgb,
    size_t height,
    size_t width,
    double* workspace_rgb,
    size_t workspace_doubles,
    double* output_rgb) {
    size_t count;
    size_t index;
    size_t channel;
    size_t y;
    size_t x;
    uint32_t radii[3];
    double weights[3][NF_GAUSSIAN_MAX_DIAMETER_V1];
    const nf_gaussian_status_v1 profile_status =
        nf_gaussian_validate_profile_v1(profile);
    if (profile_status != NF_GAUSSIAN_OK_V1) {
        return profile_status;
    }
    if (
        input_rgb == NULL ||
        workspace_rgb == NULL ||
        output_rgb == NULL ||
        height == 0 ||
        width == 0 ||
        height > SIZE_MAX / width ||
        height * width > SIZE_MAX / 3u
    ) {
        return NF_GAUSSIAN_INVALID_ARGUMENT_V1;
    }
    count = height * width * 3u;
    if (
        workspace_doubles < count ||
        count > SIZE_MAX / sizeof(double) ||
        nf_ranges_overlap(input_rgb, workspace_rgb, count) ||
        nf_ranges_overlap(input_rgb, output_rgb, count) ||
        nf_ranges_overlap(workspace_rgb, output_rgb, count)
    ) {
        return NF_GAUSSIAN_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < count; ++index) {
        if (!isfinite(input_rgb[index]) || input_rgb[index] < 0.0) {
            return NF_GAUSSIAN_INVALID_INPUT_V1;
        }
    }
    for (channel = 0; channel < 3; ++channel) {
        radii[channel] = nf_radius(
            profile->sigma_pixels_rgb[channel],
            profile->truncate);
        nf_build_weights(
            profile->sigma_pixels_rgb[channel],
            radii[channel],
            weights[channel]);
    }

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (channel = 0; channel < 3; ++channel) {
                const uint32_t radius = radii[channel];
                double value = 0.0;
                uint32_t kernel;
                for (kernel = 0; kernel <= 2u * radius; ++kernel) {
                    const ptrdiff_t offset =
                        (ptrdiff_t)kernel - (ptrdiff_t)radius;
                    const size_t source_y = nf_clamp_index(
                        (ptrdiff_t)y + offset, height);
                    value += weights[channel][kernel] *
                        input_rgb[(source_y * width + x) * 3u + channel];
                }
                workspace_rgb[(y * width + x) * 3u + channel] = value;
            }
        }
    }
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            for (channel = 0; channel < 3; ++channel) {
                const uint32_t radius = radii[channel];
                double value = 0.0;
                uint32_t kernel;
                for (kernel = 0; kernel <= 2u * radius; ++kernel) {
                    const ptrdiff_t offset =
                        (ptrdiff_t)kernel - (ptrdiff_t)radius;
                    const size_t source_x = nf_clamp_index(
                        (ptrdiff_t)x + offset, width);
                    value += weights[channel][kernel] *
                        workspace_rgb[
                            (y * width + source_x) * 3u + channel
                        ];
                }
                output_rgb[(y * width + x) * 3u + channel] = value;
            }
        }
    }
    return NF_GAUSSIAN_OK_V1;
}
