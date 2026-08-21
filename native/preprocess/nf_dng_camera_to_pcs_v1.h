#ifndef NF_DNG_CAMERA_TO_PCS_V1_H
#define NF_DNG_CAMERA_TO_PCS_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_DNG_CAMERA_TO_PCS_BUILD)
#    define NF_DNG_CAMERA_TO_PCS_API __declspec(dllexport)
#  else
#    define NF_DNG_CAMERA_TO_PCS_API __declspec(dllimport)
#  endif
#else
#  define NF_DNG_CAMERA_TO_PCS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_DNG_CAMERA_TO_PCS_ABI_VERSION_V1 1u

typedef enum nf_dng_camera_to_pcs_status_v1 {
    NF_DNG_CAMERA_TO_PCS_OK_V1 = 0,
    NF_DNG_CAMERA_TO_PCS_INVALID_ARGUMENT_V1 = 1,
    NF_DNG_CAMERA_TO_PCS_INVALID_MATRIX_V1 = 2,
    NF_DNG_CAMERA_TO_PCS_INVALID_INPUT_V1 = 3,
    NF_DNG_CAMERA_TO_PCS_NUMERIC_FAILURE_V1 = 4,
    NF_DNG_CAMERA_TO_PCS_OVERLAP_V1 = 5
} nf_dng_camera_to_pcs_status_v1;

NF_DNG_CAMERA_TO_PCS_API uint32_t
nf_dng_camera_to_pcs_abi_version_v1(void);

/*
 * Matrix is row-major float64 and samples are interleaved float64 triplets.
 * Input and output may be the same pointer. Any other overlap is invalid.
 * Every failure leaves the output buffer untouched.
 */
NF_DNG_CAMERA_TO_PCS_API nf_dng_camera_to_pcs_status_v1
nf_dng_camera_to_pcs_apply_v1(
    const double matrix[9],
    const double* input_camera,
    size_t triplet_count,
    double* output_pcs,
    size_t output_value_count);

#ifdef __cplusplus
}
#endif

#endif
