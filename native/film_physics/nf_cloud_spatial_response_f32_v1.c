#define NF_CLOUD_SPATIAL_RESPONSE_F32_BUILD
#include "nf_cloud_spatial_response_f32_v1.h"

#include <math.h>

static size_t nf_radius(double sigma, double truncate) {
    return (size_t)(truncate * sigma + 0.5);
}

static int nf_profile_valid(
    const nf_cloud_spatial_response_f32_profile_v1* profile) {
    if (profile == NULL || profile->struct_size != sizeof(*profile) ||
        profile->abi_version != NF_CLOUD_SPATIAL_RESPONSE_F32_ABI_VERSION_V1 ||
        !isfinite(profile->truncate) || profile->truncate <= 0.0 ||
        profile->truncate > 8.0) {
        return 0;
    }
    for (size_t channel = 0u; channel < 3u; ++channel) {
        if (!isfinite(profile->sigma_pixels_cmy[channel]) ||
            profile->sigma_pixels_cmy[channel] <= 0.0 ||
            nf_radius(profile->sigma_pixels_cmy[channel], profile->truncate) >
                NF_CLOUD_SPATIAL_RESPONSE_F32_MAX_RADIUS_V1 ||
            !isfinite(profile->mark_optical_density_cmy[channel]) ||
            profile->mark_optical_density_cmy[channel] <= 0.0) {
            return 0;
        }
    }
    return 1;
}

static void nf_kernel(double sigma, size_t radius, double* weights) {
    double total = 0.0;
    for (size_t index = 0u; index <= 2u * radius; ++index) {
        const double offset = (double)((ptrdiff_t)index - (ptrdiff_t)radius);
        weights[index] = exp(-0.5 * (offset / sigma) * (offset / sigma));
        total += weights[index];
    }
    for (size_t index = 0u; index <= 2u * radius; ++index) {
        weights[index] /= total;
    }
}

uint32_t nf_cloud_spatial_response_f32_abi_version_v1(void) {
    return NF_CLOUD_SPATIAL_RESPONSE_F32_ABI_VERSION_V1;
}

nf_cloud_spatial_response_f32_status_v1 nf_cloud_spatial_response_f32_apply_v1(
    const nf_cloud_spatial_response_f32_profile_v1* profile,
    const uint16_t* counts,
    size_t core_height,
    size_t width,
    size_t halo,
    double* workspace,
    size_t workspace_doubles,
    float* output_density,
    float* output_transmittance,
    size_t output_values) {
    size_t extended_height;
    size_t extended_pixels;
    size_t core_pixels;
    double* horizontal;
    double* vertical;
    if (!nf_profile_valid(profile) || counts == NULL || workspace == NULL ||
        output_density == NULL || output_transmittance == NULL ||
        core_height == 0u || width == 0u || halo > SIZE_MAX / 2u ||
        core_height > SIZE_MAX - 2u * halo) {
        return NF_CLOUD_SPATIAL_RESPONSE_F32_INVALID_ARGUMENT_V1;
    }
    extended_height = core_height + 2u * halo;
    if (extended_height > SIZE_MAX / width || core_height > SIZE_MAX / width) {
        return NF_CLOUD_SPATIAL_RESPONSE_F32_INVALID_ARGUMENT_V1;
    }
    extended_pixels = extended_height * width;
    core_pixels = core_height * width;
    if (extended_pixels > SIZE_MAX / 2u || workspace_doubles < 2u * extended_pixels ||
        core_pixels > SIZE_MAX / 3u || output_values < 3u * core_pixels) {
        return NF_CLOUD_SPATIAL_RESPONSE_F32_INVALID_ARGUMENT_V1;
    }
    for (size_t channel = 0u; channel < 3u; ++channel) {
        if (nf_radius(profile->sigma_pixels_cmy[channel], profile->truncate) > halo) {
            return NF_CLOUD_SPATIAL_RESPONSE_F32_DOMAIN_ERROR_V1;
        }
    }
    horizontal = workspace;
    vertical = workspace + extended_pixels;
    for (size_t channel = 0u; channel < 3u; ++channel) {
        double weights[2u * NF_CLOUD_SPATIAL_RESPONSE_F32_MAX_RADIUS_V1 + 1u];
        const size_t radius = nf_radius(
            profile->sigma_pixels_cmy[channel], profile->truncate);
        nf_kernel(profile->sigma_pixels_cmy[channel], radius, weights);
        for (size_t y = 0u; y < extended_height; ++y) {
            for (size_t x = 0u; x < width; ++x) {
                double total = 0.0;
                for (size_t k = 0u; k <= 2u * radius; ++k) {
                    const ptrdiff_t raw = (ptrdiff_t)x + (ptrdiff_t)k -
                        (ptrdiff_t)radius;
                    const size_t wrapped = (size_t)(
                        (raw % (ptrdiff_t)width + (ptrdiff_t)width) %
                        (ptrdiff_t)width);
                    total += weights[k] * (double)counts[
                        (y * width + wrapped) * 3u + channel];
                }
                horizontal[y * width + x] = total;
            }
        }
        for (size_t y = 0u; y < extended_height; ++y) {
            for (size_t x = 0u; x < width; ++x) {
                double total = 0.0;
                for (size_t k = 0u; k <= 2u * radius; ++k) {
                    ptrdiff_t source = (ptrdiff_t)y + (ptrdiff_t)k -
                        (ptrdiff_t)radius;
                    if (source < 0) source = 0;
                    if ((size_t)source >= extended_height) {
                        source = (ptrdiff_t)extended_height - 1;
                    }
                    total += weights[k] * horizontal[(size_t)source * width + x];
                }
                vertical[y * width + x] = total;
            }
        }
        for (size_t y = 0u; y < core_height; ++y) {
            for (size_t x = 0u; x < width; ++x) {
                const size_t output = (y * width + x) * 3u + channel;
                const double density = vertical[(y + halo) * width + x] *
                    profile->mark_optical_density_cmy[channel];
                output_density[output] = (float)density;
                output_transmittance[output] = (float)exp(-density);
            }
        }
    }
    return NF_CLOUD_SPATIAL_RESPONSE_F32_OK_V1;
}
