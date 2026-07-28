#ifndef NF_BOUNDED_ADJACENCY_F32_V1_H
#define NF_BOUNDED_ADJACENCY_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_BOUNDED_ADJACENCY_F32_BUILD)
#    define NF_BOUNDED_ADJACENCY_F32_API __declspec(dllexport)
#  else
#    define NF_BOUNDED_ADJACENCY_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_BOUNDED_ADJACENCY_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_BOUNDED_ADJACENCY_F32_ABI_VERSION_V1 1u

typedef enum nf_bounded_adjacency_f32_status_v1 {
    NF_BOUNDED_ADJACENCY_F32_OK_V1 = 0,
    NF_BOUNDED_ADJACENCY_F32_INVALID_ARGUMENT_V1 = 1,
    NF_BOUNDED_ADJACENCY_F32_INVALID_PROFILE_V1 = 2,
    NF_BOUNDED_ADJACENCY_F32_INVALID_INPUT_V1 = 3,
    NF_BOUNDED_ADJACENCY_F32_NUMERIC_FAILURE_V1 = 4
} nf_bounded_adjacency_f32_status_v1;

typedef struct nf_bounded_adjacency_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    double gain_rgb[3];
    double maximum_absolute_transmittance_delta;
    double maximum_absolute_density_delta;
    double black_reference_density[3];
    double white_reference_density[3];
} nf_bounded_adjacency_f32_profile_v1;

NF_BOUNDED_ADJACENCY_F32_API uint32_t
nf_bounded_adjacency_f32_abi_version_v1(void);

NF_BOUNDED_ADJACENCY_F32_API nf_bounded_adjacency_f32_status_v1
nf_bounded_adjacency_f32_validate_profile_v1(
    const nf_bounded_adjacency_f32_profile_v1* profile);

NF_BOUNDED_ADJACENCY_F32_API nf_bounded_adjacency_f32_status_v1
nf_bounded_adjacency_f32_apply_v1(
    const nf_bounded_adjacency_f32_profile_v1* profile,
    const float* developed_density_rgb,
    const float* blurred_density_rgb,
    size_t rgb_count,
    float* output_density_rgb);

#ifdef __cplusplus
}
#endif

#endif
