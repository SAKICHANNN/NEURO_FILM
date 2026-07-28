#ifndef NF_AO6_RESIDUAL_F32_V1_H
#define NF_AO6_RESIDUAL_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_AO6_RESIDUAL_F32_BUILD)
#    define NF_AO6_RESIDUAL_F32_API __declspec(dllexport)
#  else
#    define NF_AO6_RESIDUAL_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_RESIDUAL_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_RESIDUAL_F32_ABI_VERSION_V1 1u

typedef enum nf_ao6_residual_f32_status_v1 {
    NF_AO6_RESIDUAL_F32_OK_V1 = 0,
    NF_AO6_RESIDUAL_F32_INVALID_ARGUMENT_V1 = 1,
    NF_AO6_RESIDUAL_F32_INVALID_PROFILE_V1 = 2,
    NF_AO6_RESIDUAL_F32_INVALID_INPUT_V1 = 3,
    NF_AO6_RESIDUAL_F32_NUMERIC_FAILURE_V1 = 4
} nf_ao6_residual_f32_status_v1;

typedef struct nf_ao6_residual_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    double capture_matrix[3][3];
    double response_midpoints[3];
    double response_slopes[3];
    double maximum_responses[3];
    double scan_matrix[3][3];
    double black_endpoint[3];
    double white_endpoint[3];
    double luma_weights[3];
    double exposure_floor;
    double matrix_minimum_determinant;
    double minimum_endpoint_span;
    double tone_strength;
    double chroma_strength;
    double hard_low_linear;
    double hard_high_linear;
    double guard_low_linear;
    double guard_high_linear;
} nf_ao6_residual_f32_profile_v1;

NF_AO6_RESIDUAL_F32_API uint32_t
nf_ao6_residual_f32_abi_version_v1(void);

NF_AO6_RESIDUAL_F32_API nf_ao6_residual_f32_status_v1
nf_ao6_residual_f32_validate_profile_v1(
    const nf_ao6_residual_f32_profile_v1* profile);

/*
 * Input is the externally prepared AO6 source-context base in relative
 * linear-sRGB. Output may exactly alias input. Optional tone/chroma scale
 * arrays contain one float per RGB triplet. The implementation performs a
 * validation pass before writing any destination.
 */
NF_AO6_RESIDUAL_F32_API nf_ao6_residual_f32_status_v1
nf_ao6_residual_f32_apply_v1(
    const nf_ao6_residual_f32_profile_v1* profile,
    const float* base_linear_rgb,
    size_t rgb_count,
    float* output_linear_rgb,
    float* tone_scale,
    float* chroma_scale);

#ifdef __cplusplus
}
#endif

#endif
