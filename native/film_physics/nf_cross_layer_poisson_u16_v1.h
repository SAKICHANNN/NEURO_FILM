#ifndef NF_CROSS_LAYER_POISSON_U16_V1_H
#define NF_CROSS_LAYER_POISSON_U16_V1_H

#include <stddef.h>
#include <stdint.h>

#define NF_CROSS_LAYER_POISSON_U16_ABI_VERSION_V1 1u

#if defined(_WIN32) && defined(NF_CROSS_LAYER_POISSON_U16_BUILD)
#define NF_CROSS_LAYER_POISSON_U16_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_CROSS_LAYER_POISSON_U16_API __declspec(dllimport)
#else
#define NF_CROSS_LAYER_POISSON_U16_API __attribute__((visibility("default")))
#endif

typedef struct nf_cross_layer_poisson_u16_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    double marginal_rates_cmy[3];
    double shared_all_rate;
    double shared_pair_rates_cm_cy_my[3];
    uint64_t seed;
    uint64_t component_seed_stride;
} nf_cross_layer_poisson_u16_profile_v1;

typedef enum nf_cross_layer_poisson_u16_status_v1 {
    NF_CROSS_LAYER_POISSON_U16_OK_V1 = 0,
    NF_CROSS_LAYER_POISSON_U16_INVALID_ARGUMENT_V1 = 1,
    NF_CROSS_LAYER_POISSON_U16_DOMAIN_ERROR_V1 = 2
} nf_cross_layer_poisson_u16_status_v1;

NF_CROSS_LAYER_POISSON_U16_API uint32_t
nf_cross_layer_poisson_u16_abi_version_v1(void);

/* Output is interleaved C/M/Y. Regions are coordinates in the full field. */
NF_CROSS_LAYER_POISSON_U16_API nf_cross_layer_poisson_u16_status_v1
nf_cross_layer_poisson_u16_sample_region_v1(
    const nf_cross_layer_poisson_u16_profile_v1* profile,
    size_t full_height,
    size_t full_width,
    size_t origin_y,
    size_t origin_x,
    size_t height,
    size_t width,
    uint16_t* output_cmy,
    size_t output_values);

#endif
