#ifndef NF_THOMAS_ROWS_F32_V1_H
#define NF_THOMAS_ROWS_F32_V1_H

#include "nf_thomas_field_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_THOMAS_ROWS_F32_BUILD)
#    define NF_THOMAS_ROWS_F32_API __declspec(dllexport)
#  else
#    define NF_THOMAS_ROWS_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_THOMAS_ROWS_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_THOMAS_ROWS_F32_ABI_VERSION_V1 1u

typedef enum nf_thomas_rows_f32_status_v1 {
    NF_THOMAS_ROWS_F32_OK_V1 = 0,
    NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1 = 1,
    NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1 = 2,
    NF_THOMAS_ROWS_F32_DOMAIN_ERROR_V1 = 3
} nf_thomas_rows_f32_status_v1;

NF_THOMAS_ROWS_F32_API uint32_t
nf_thomas_rows_f32_abi_version_v1(void);

NF_THOMAS_ROWS_F32_API nf_thomas_rows_f32_status_v1
nf_thomas_rows_f32_workspace_floats_v1(
    size_t width,
    size_t row_count,
    size_t* workspace_floats);

/* Render the unchanged uncentered P8BS field for one global row interval. */
NF_THOMAS_ROWS_F32_API nf_thomas_rows_f32_status_v1
nf_thomas_rows_f32_field_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height,
    size_t width,
    size_t row_start,
    size_t row_count,
    float* workspace,
    size_t workspace_floats,
    float* output,
    size_t output_floats);

/* Compose supplied row-local density/sigma after one caller-held full-field mean. */
NF_THOMAS_ROWS_F32_API nf_thomas_rows_f32_status_v1
nf_thomas_rows_f32_density_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height,
    size_t width,
    size_t row_start,
    size_t row_count,
    const float* base_density,
    size_t base_density_floats,
    const float* point_density_sigma,
    size_t point_density_sigma_floats,
    double raw_field_mean,
    float* workspace,
    size_t workspace_floats,
    float* output_transmittance,
    size_t output_floats);

#ifdef __cplusplus
}
#endif

#endif
