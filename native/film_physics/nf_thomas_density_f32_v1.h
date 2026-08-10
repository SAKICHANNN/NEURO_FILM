#ifndef NF_THOMAS_DENSITY_F32_V1_H
#define NF_THOMAS_DENSITY_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#include "nf_thomas_field_f32_v1.h"

#define NF_THOMAS_DENSITY_F32_ABI_VERSION_V1 1u

#if defined(_WIN32) && defined(NF_THOMAS_DENSITY_F32_BUILD)
#define NF_THOMAS_DENSITY_F32_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_THOMAS_DENSITY_F32_API __declspec(dllimport)
#else
#define NF_THOMAS_DENSITY_F32_API __attribute__((visibility("default")))
#endif

typedef enum nf_thomas_density_f32_status_v1 {
    NF_THOMAS_DENSITY_F32_OK_V1 = 0,
    NF_THOMAS_DENSITY_F32_INVALID_ARGUMENT_V1 = 1,
    NF_THOMAS_DENSITY_F32_INVALID_PROFILE_V1 = 2,
    NF_THOMAS_DENSITY_F32_DOMAIN_ERROR_V1 = 3
} nf_thomas_density_f32_status_v1;

NF_THOMAS_DENSITY_F32_API uint32_t
nf_thomas_density_f32_abi_version_v1(void);

NF_THOMAS_DENSITY_F32_API nf_thomas_density_f32_status_v1
nf_thomas_density_f32_workspace_floats_v1(
    size_t height,
    size_t width,
    size_t* workspace_floats);

/*
 * Compose a supplied developed-density mean and point-density sigma with the
 * unchanged P8BS DC-projected field, then convert once through T=10^-D.
 * Output and raw_mean are committed only after the complete candidate passes.
 */
NF_THOMAS_DENSITY_F32_API nf_thomas_density_f32_status_v1
nf_thomas_density_f32_apply_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t height,
    size_t width,
    const float* base_density,
    size_t base_density_floats,
    const float* point_density_sigma,
    size_t point_density_sigma_floats,
    float* workspace,
    size_t workspace_floats,
    float* output_transmittance,
    size_t output_floats,
    double* raw_field_mean);

#endif
