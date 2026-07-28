#ifndef NF_PHYSICAL_PRINT_V1_H
#define NF_PHYSICAL_PRINT_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_PHYSICAL_PRINT_BUILD)
#    define NF_PHYSICAL_PRINT_API __declspec(dllexport)
#  else
#    define NF_PHYSICAL_PRINT_API __declspec(dllimport)
#  endif
#else
#  define NF_PHYSICAL_PRINT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_PHYSICAL_PRINT_ABI_VERSION_V1 1u
#define NF_PHYSICAL_PRINT_MAX_KNOTS_V1 16u

typedef enum nf_physical_print_status_v1 {
    NF_PHYSICAL_PRINT_OK_V1 = 0,
    NF_PHYSICAL_PRINT_INVALID_ARGUMENT_V1 = 1,
    NF_PHYSICAL_PRINT_INVALID_PROFILE_V1 = 2,
    NF_PHYSICAL_PRINT_INVALID_INPUT_V1 = 3,
    NF_PHYSICAL_PRINT_NUMERIC_FAILURE_V1 = 4
} nf_physical_print_status_v1;

typedef struct nf_physical_print_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    double reference_linear;
    double black_offset;
    uint32_t knot_count[3];
    double x_knots[3][NF_PHYSICAL_PRINT_MAX_KNOTS_V1];
    double y_knots[3][NF_PHYSICAL_PRINT_MAX_KNOTS_V1];
    double derivatives[3][NF_PHYSICAL_PRINT_MAX_KNOTS_V1];
    double dye_absorption_matrix[3][3];
    double print_matrix[3][3];
    double paper_midpoints[3];
    double paper_slopes[3];
    double paper_maximum_densities[3];
    double black_reference_density[3];
    double white_reference_density[3];
    double exposure_floor;
    double matrix_minimum_determinant;
} nf_physical_print_profile_v1;

NF_PHYSICAL_PRINT_API uint32_t nf_physical_print_abi_version_v1(void);

NF_PHYSICAL_PRINT_API nf_physical_print_status_v1
nf_physical_print_validate_profile_v1(
    const nf_physical_print_profile_v1* profile);

/*
 * RGB arrays contain rgb_count interleaved float64 triplets.
 * Input and output may be the same pointer. Otherwise they must not overlap.
 * Any validation failure leaves output untouched.
 */
NF_PHYSICAL_PRINT_API nf_physical_print_status_v1
nf_physical_print_apply_v1(
    const nf_physical_print_profile_v1* profile,
    const double* input_rgb,
    size_t rgb_count,
    double* output_rgb);

#ifdef __cplusplus
}
#endif

#endif
