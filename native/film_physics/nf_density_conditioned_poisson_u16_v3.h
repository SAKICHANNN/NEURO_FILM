#ifndef NF_DENSITY_CONDITIONED_POISSON_U16_V3_H
#define NF_DENSITY_CONDITIONED_POISSON_U16_V3_H

#include <stddef.h>
#include <stdint.h>

#define NF_DENSITY_CONDITIONED_POISSON_U16_ABI_VERSION_V3 3u

#if defined(_WIN32) && defined(NF_DENSITY_CONDITIONED_POISSON_U16_BUILD)
#define NF_DENSITY_CONDITIONED_POISSON_U16_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_DENSITY_CONDITIONED_POISSON_U16_API __declspec(dllimport)
#else
#define NF_DENSITY_CONDITIONED_POISSON_U16_API __attribute__((visibility("default")))
#endif

typedef struct nf_density_conditioned_poisson_u16_profile_v3 {
    uint32_t struct_size;
    uint32_t abi_version;
    double marginal_rates_cmy[3];
    double shared_all_rate;
    double shared_pair_rates_cm_cy_my[3];
    uint64_t seed;
    uint64_t component_seed_stride;
} nf_density_conditioned_poisson_u16_profile_v3;

typedef enum nf_density_conditioned_poisson_u16_status_v3 {
    NF_DENSITY_CONDITIONED_POISSON_U16_OK_V3 = 0,
    NF_DENSITY_CONDITIONED_POISSON_U16_INVALID_ARGUMENT_V3 = 1,
    NF_DENSITY_CONDITIONED_POISSON_U16_DOMAIN_ERROR_V3 = 2
} nf_density_conditioned_poisson_u16_status_v3;

NF_DENSITY_CONDITIONED_POISSON_U16_API uint32_t
nf_density_conditioned_poisson_u16_abi_version_v3(void);

/* Scale and output are interleaved C/M/Y over the requested region. */
NF_DENSITY_CONDITIONED_POISSON_U16_API
nf_density_conditioned_poisson_u16_status_v3
nf_density_conditioned_poisson_u16_sample_region_v3(
    const nf_density_conditioned_poisson_u16_profile_v3* profile,
    size_t full_height,
    size_t full_width,
    size_t origin_y,
    size_t origin_x,
    size_t height,
    size_t width,
    const double* scale_cmy,
    size_t scale_values,
    uint16_t* output_cmy,
    size_t output_values);

#endif
