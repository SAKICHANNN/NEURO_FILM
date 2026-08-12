#define NF_DENSITY_CONDITIONED_POISSON_U16_BUILD
#include "nf_density_conditioned_poisson_u16_v2.h"

#include <math.h>

static uint64_t nf_splitmix64(uint64_t value) {
    uint64_t state = value + UINT64_C(0x9E3779B97F4A7C15);
    state = (state ^ (state >> 30u)) * UINT64_C(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27u)) * UINT64_C(0x94D049BB133111EB);
    return state ^ (state >> 31u);
}

static double nf_uniform(uint64_t counter) {
    return ((double)(nf_splitmix64(counter) >> 11u) + 0.5) /
        9007199254740992.0;
}

static uint16_t nf_poisson(double rate, double uniform) {
    double probability = exp(-rate);
    double cumulative = probability;
    uint16_t order = 0u;
    while (uniform > cumulative) {
        ++order;
        probability *= rate / (double)order;
        cumulative += probability;
    }
    return order;
}

static int nf_profile_valid(
    const nf_density_conditioned_poisson_u16_profile_v2* profile) {
    double independent[3];
    if (profile == NULL || profile->struct_size != sizeof(*profile) ||
        profile->abi_version != NF_DENSITY_CONDITIONED_POISSON_U16_ABI_VERSION_V2 ||
        profile->component_seed_stride == 0u ||
        !isfinite(profile->shared_all_rate) || profile->shared_all_rate <= 0.0 ||
        profile->shared_all_rate > 64.0) {
        return 0;
    }
    for (size_t index = 0u; index < 3u; ++index) {
        if (!isfinite(profile->marginal_rates_cmy[index]) ||
            profile->marginal_rates_cmy[index] <= 0.0 ||
            profile->marginal_rates_cmy[index] > 64.0 ||
            !isfinite(profile->shared_pair_rates_cm_cy_my[index]) ||
            profile->shared_pair_rates_cm_cy_my[index] <= 0.0 ||
            profile->shared_pair_rates_cm_cy_my[index] > 64.0) {
            return 0;
        }
    }
    independent[0] = profile->marginal_rates_cmy[0] - profile->shared_all_rate -
        profile->shared_pair_rates_cm_cy_my[0] -
        profile->shared_pair_rates_cm_cy_my[1];
    independent[1] = profile->marginal_rates_cmy[1] - profile->shared_all_rate -
        profile->shared_pair_rates_cm_cy_my[0] -
        profile->shared_pair_rates_cm_cy_my[2];
    independent[2] = profile->marginal_rates_cmy[2] - profile->shared_all_rate -
        profile->shared_pair_rates_cm_cy_my[1] -
        profile->shared_pair_rates_cm_cy_my[2];
    return independent[0] > 0.0 && independent[1] > 0.0 && independent[2] > 0.0;
}

uint32_t nf_density_conditioned_poisson_u16_abi_version_v2(void) {
    return NF_DENSITY_CONDITIONED_POISSON_U16_ABI_VERSION_V2;
}

nf_density_conditioned_poisson_u16_status_v2
nf_density_conditioned_poisson_u16_sample_region_v2(
    const nf_density_conditioned_poisson_u16_profile_v2* profile,
    size_t full_height,
    size_t full_width,
    size_t origin_y,
    size_t origin_x,
    size_t height,
    size_t width,
    const float* scale_cmy,
    size_t scale_values,
    uint16_t* output_cmy,
    size_t output_values) {
    size_t pixels;
    if (!nf_profile_valid(profile) || scale_cmy == NULL || output_cmy == NULL ||
        full_height == 0u || full_width == 0u || height == 0u || width == 0u ||
        origin_y >= full_height || origin_x >= full_width ||
        height > full_height - origin_y || width > full_width - origin_x ||
        height > SIZE_MAX / width ||
        (uint64_t)full_height > UINT64_MAX / (uint64_t)full_width) {
        return NF_DENSITY_CONDITIONED_POISSON_U16_INVALID_ARGUMENT_V2;
    }
    pixels = height * width;
    if (pixels > SIZE_MAX / 3u || scale_values < pixels * 3u ||
        output_values < pixels * 3u) {
        return NF_DENSITY_CONDITIONED_POISSON_U16_INVALID_ARGUMENT_V2;
    }
    for (size_t index = 0u; index < pixels * 3u; ++index) {
        if (!isfinite(scale_cmy[index]) || scale_cmy[index] < 0.0f ||
            scale_cmy[index] > 1.0f) {
            return NF_DENSITY_CONDITIONED_POISSON_U16_DOMAIN_ERROR_V2;
        }
    }
    for (size_t local_y = 0u; local_y < height; ++local_y) {
        const uint64_t global_y = (uint64_t)(origin_y + local_y);
        for (size_t local_x = 0u; local_x < width; ++local_x) {
            const size_t pixel = local_y * width + local_x;
            const size_t offset = pixel * 3u;
            const double c = (double)scale_cmy[offset];
            const double m = (double)scale_cmy[offset + 1u];
            const double y = (double)scale_cmy[offset + 2u];
            const double shared_all = profile->shared_all_rate * fmin(fmin(c, m), y);
            const double cm = profile->shared_pair_rates_cm_cy_my[0] * fmin(c, m);
            const double cy = profile->shared_pair_rates_cm_cy_my[1] * fmin(c, y);
            const double my = profile->shared_pair_rates_cm_cy_my[2] * fmin(m, y);
            double rates[7];
            uint32_t component[7];
            const uint64_t coordinate = global_y * (uint64_t)full_width +
                (uint64_t)(origin_x + local_x);
            rates[0] = shared_all;
            rates[1] = cm;
            rates[2] = cy;
            rates[3] = my;
            rates[4] = fmax(profile->marginal_rates_cmy[0] * c - shared_all - cm - cy, 0.0);
            rates[5] = fmax(profile->marginal_rates_cmy[1] * m - shared_all - cm - my, 0.0);
            rates[6] = fmax(profile->marginal_rates_cmy[2] * y - shared_all - cy - my, 0.0);
            for (size_t index = 0u; index < 7u; ++index) {
                const uint64_t seed = profile->seed +
                    (uint64_t)index * profile->component_seed_stride;
                component[index] = (uint32_t)nf_poisson(
                    rates[index], nf_uniform(coordinate + seed));
            }
            output_cmy[offset] = (uint16_t)(
                component[0] + component[1] + component[2] + component[4]);
            output_cmy[offset + 1u] = (uint16_t)(
                component[0] + component[1] + component[3] + component[5]);
            output_cmy[offset + 2u] = (uint16_t)(
                component[0] + component[2] + component[3] + component[6]);
        }
    }
    return NF_DENSITY_CONDITIONED_POISSON_U16_OK_V2;
}
