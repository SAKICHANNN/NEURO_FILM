#ifndef NF_PHYSICAL_DOMAINS_F32_V1_H
#define NF_PHYSICAL_DOMAINS_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_PHYSICAL_DOMAINS_F32_BUILD)
#    define NF_PHYSICAL_DOMAINS_F32_API __declspec(dllexport)
#  else
#    define NF_PHYSICAL_DOMAINS_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_PHYSICAL_DOMAINS_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_PHYSICAL_DOMAINS_F32_ABI_VERSION_V1 1u
#define NF_PHYSICAL_DOMAINS_F32_MAX_KNOTS_V1 16u

typedef enum nf_physical_domains_f32_status_v1 {
    NF_PHYSICAL_DOMAINS_F32_OK_V1 = 0,
    NF_PHYSICAL_DOMAINS_F32_INVALID_ARGUMENT_V1 = 1,
    NF_PHYSICAL_DOMAINS_F32_INVALID_PROFILE_V1 = 2,
    NF_PHYSICAL_DOMAINS_F32_INVALID_INPUT_V1 = 3,
    NF_PHYSICAL_DOMAINS_F32_NUMERIC_FAILURE_V1 = 4
} nf_physical_domains_f32_status_v1;

/*
 * Parameters remain float64 so one compiled profile describes Reference and
 * Standard execution. Sample arrays are float32 and round once per stage.
 */
typedef struct nf_physical_domains_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    double reference_linear;
    double black_offset;
    uint32_t knot_count[3];
    double x_knots[3][NF_PHYSICAL_DOMAINS_F32_MAX_KNOTS_V1];
    double y_knots[3][NF_PHYSICAL_DOMAINS_F32_MAX_KNOTS_V1];
    double derivatives[3][NF_PHYSICAL_DOMAINS_F32_MAX_KNOTS_V1];
    double dye_absorption_matrix[3][3];
    double print_matrix[3][3];
    double paper_midpoints[3];
    double paper_slopes[3];
    double paper_maximum_densities[3];
    double black_reference_density[3];
    double white_reference_density[3];
    double exposure_floor;
    double matrix_minimum_determinant;
} nf_physical_domains_f32_profile_v1;

NF_PHYSICAL_DOMAINS_F32_API uint32_t
nf_physical_domains_f32_abi_version_v1(void);

NF_PHYSICAL_DOMAINS_F32_API nf_physical_domains_f32_status_v1
nf_physical_domains_f32_validate_profile_v1(
    const nf_physical_domains_f32_profile_v1* profile);

NF_PHYSICAL_DOMAINS_F32_API nf_physical_domains_f32_status_v1
nf_physical_sensitometry_f32_apply_v1(
    const nf_physical_domains_f32_profile_v1* profile,
    const float* scene_linear_rgb,
    size_t rgb_count,
    float* developed_density_rgb);

/*
 * Versioned double-output companion for stochastic developed-density
 * consumers. The profile and scene samples remain identical to v1; only the
 * output storage avoids an otherwise irreversible float32 quantization.
 */
NF_PHYSICAL_DOMAINS_F32_API nf_physical_domains_f32_status_v1
nf_physical_sensitometry_f64_apply_v2(
    const nf_physical_domains_f32_profile_v1* profile,
    const float* scene_linear_rgb,
    size_t rgb_count,
    double* developed_density_rgb);

NF_PHYSICAL_DOMAINS_F32_API nf_physical_domains_f32_status_v1
nf_physical_interpretation_f32_apply_v1(
    const nf_physical_domains_f32_profile_v1* profile,
    const float* developed_density_rgb,
    size_t rgb_count,
    float* scan_linear_rgb);

#ifdef __cplusplus
}
#endif

#endif
