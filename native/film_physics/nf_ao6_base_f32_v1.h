#ifndef NF_AO6_BASE_F32_V1_H
#define NF_AO6_BASE_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_AO6_BASE_F32_BUILD)
#    define NF_AO6_BASE_F32_API __declspec(dllexport)
#  else
#    define NF_AO6_BASE_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_BASE_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_BASE_F32_ABI_VERSION_V1 1u

typedef enum nf_ao6_base_f32_status_v1 {
    NF_AO6_BASE_F32_OK_V1 = 0,
    NF_AO6_BASE_F32_INVALID_ARGUMENT_V1 = 1,
    NF_AO6_BASE_F32_INVALID_PROFILE_V1 = 2,
    NF_AO6_BASE_F32_INVALID_CONTEXT_V1 = 3,
    NF_AO6_BASE_F32_INVALID_INPUT_V1 = 4,
    NF_AO6_BASE_F32_NUMERIC_FAILURE_V1 = 5
} nf_ao6_base_f32_status_v1;

typedef struct nf_ao6_base_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    double capture_matrix[3][3];
    double negative_midpoints[3];
    double negative_slopes[3];
    double negative_maximum_densities[3];
    double dye_absorption_matrix[3][3];
    double print_matrix[3][3];
    double paper_midpoints[3];
    double paper_slopes[3];
    double paper_maximum_densities[3];
    double black_endpoint[3];
    double white_endpoint[3];
    float destination_mean[3];
    float destination_std[3];
    double exposure_floor;
    double matrix_minimum_determinant;
    double minimum_endpoint_span;
    double density_strength;
    float style_strength;
    float luma_strength;
    uint32_t gamut_iterations;
    uint32_t output_margin_8bit;
} nf_ao6_base_f32_profile_v1;

typedef struct nf_ao6_base_f32_context_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t pixel_count;
    float source_mean[3];
    float source_std[3];
} nf_ao6_base_f32_context_v1;

NF_AO6_BASE_F32_API uint32_t
nf_ao6_base_f32_abi_version_v1(void);

NF_AO6_BASE_F32_API nf_ao6_base_f32_status_v1
nf_ao6_base_f32_validate_profile_v1(
    const nf_ao6_base_f32_profile_v1* profile);

/*
 * Input and output are encoded relative-display sRGB float32 triplets.
 * Context statistics describe the density-processed full source frame.
 * Output may exactly alias input. Validation completes before any write.
 */
NF_AO6_BASE_F32_API nf_ao6_base_f32_status_v1
nf_ao6_base_f32_apply_v1(
    const nf_ao6_base_f32_profile_v1* profile,
    const nf_ao6_base_f32_context_v1* context,
    const float* encoded_srgb,
    size_t rgb_count,
    float* output_encoded_srgb);

#ifdef __cplusplus
}
#endif

#endif
