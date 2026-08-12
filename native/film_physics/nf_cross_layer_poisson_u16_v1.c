#define NF_CROSS_LAYER_POISSON_U16_BUILD
#include "nf_cross_layer_poisson_u16_v1.h"

#include <math.h>

static uint64_t nf_splitmix64(uint64_t value) {
    uint64_t state = value + UINT64_C(0x9E3779B97F4A7C15);
    state = (state ^ (state >> 30u)) * UINT64_C(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27u)) * UINT64_C(0x94D049BB133111EB);
    return state ^ (state >> 31u);
}

static double nf_counter_uniform(uint64_t counter) {
    const double inverse_2pow53 = 1.0 / 9007199254740992.0;
    return ((double)(nf_splitmix64(counter) >> 11u) + 0.5) * inverse_2pow53;
}

static int nf_rate_valid(double rate) {
    return isfinite(rate) && rate > 0.0 && rate <= 64.0;
}

static int nf_profile_rates(
    const nf_cross_layer_poisson_u16_profile_v1* profile,
    double rates[7]) {
    double independent_c;
    double independent_m;
    double independent_y;
    if (profile == NULL ||
        profile->struct_size != sizeof(*profile) ||
        profile->abi_version != NF_CROSS_LAYER_POISSON_U16_ABI_VERSION_V1 ||
        profile->component_seed_stride == 0u) {
        return 0;
    }
    rates[0] = profile->shared_all_rate;
    rates[1] = profile->shared_pair_rates_cm_cy_my[0];
    rates[2] = profile->shared_pair_rates_cm_cy_my[1];
    rates[3] = profile->shared_pair_rates_cm_cy_my[2];
    independent_c = profile->marginal_rates_cmy[0] - rates[0] - rates[1] - rates[2];
    independent_m = profile->marginal_rates_cmy[1] - rates[0] - rates[1] - rates[3];
    independent_y = profile->marginal_rates_cmy[2] - rates[0] - rates[2] - rates[3];
    rates[4] = independent_c;
    rates[5] = independent_m;
    rates[6] = independent_y;
    for (size_t index = 0u; index < 7u; ++index) {
        if (!nf_rate_valid(rates[index])) {
            return 0;
        }
    }
    return 1;
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

uint32_t nf_cross_layer_poisson_u16_abi_version_v1(void) {
    return NF_CROSS_LAYER_POISSON_U16_ABI_VERSION_V1;
}

nf_cross_layer_poisson_u16_status_v1
nf_cross_layer_poisson_u16_sample_region_v1(
    const nf_cross_layer_poisson_u16_profile_v1* profile,
    size_t full_height,
    size_t full_width,
    size_t origin_y,
    size_t origin_x,
    size_t height,
    size_t width,
    uint16_t* output_cmy,
    size_t output_values) {
    double rates[7];
    size_t pixels;
    if (!nf_profile_rates(profile, rates) || output_cmy == NULL ||
        full_height == 0u || full_width == 0u || height == 0u || width == 0u ||
        origin_y >= full_height || origin_x >= full_width ||
        height > full_height - origin_y || width > full_width - origin_x ||
        height > SIZE_MAX / width ||
        (uint64_t)full_height > UINT64_MAX / (uint64_t)full_width) {
        return NF_CROSS_LAYER_POISSON_U16_INVALID_ARGUMENT_V1;
    }
    pixels = height * width;
    if (pixels > SIZE_MAX / 3u || output_values < pixels * 3u) {
        return NF_CROSS_LAYER_POISSON_U16_INVALID_ARGUMENT_V1;
    }
    for (size_t local_y = 0u; local_y < height; ++local_y) {
        const uint64_t global_y = (uint64_t)(origin_y + local_y);
        for (size_t local_x = 0u; local_x < width; ++local_x) {
            const uint64_t global_x = (uint64_t)(origin_x + local_x);
            const uint64_t coordinate = global_y * (uint64_t)full_width + global_x;
            uint32_t component[7];
            const size_t output_index = (local_y * width + local_x) * 3u;
            for (size_t index = 0u; index < 7u; ++index) {
                const uint64_t component_seed = profile->seed +
                    (uint64_t)index * profile->component_seed_stride;
                component[index] = (uint32_t)nf_poisson(
                    rates[index], nf_counter_uniform(coordinate + component_seed));
            }
            output_cmy[output_index] = (uint16_t)(
                component[0] + component[1] + component[2] + component[4]);
            output_cmy[output_index + 1u] = (uint16_t)(
                component[0] + component[1] + component[3] + component[5]);
            output_cmy[output_index + 2u] = (uint16_t)(
                component[0] + component[2] + component[3] + component[6]);
        }
    }
    return NF_CROSS_LAYER_POISSON_U16_OK_V1;
}
