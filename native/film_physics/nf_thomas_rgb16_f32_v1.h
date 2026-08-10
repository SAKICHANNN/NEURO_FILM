#ifndef NF_THOMAS_RGB16_F32_V1_H
#define NF_THOMAS_RGB16_F32_V1_H

#include "nf_granularity_amplitude_f32_v1.h"
#include "nf_neutral_gauge_f32_v1.h"
#include "nf_thomas_rows_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_THOMAS_RGB16_F32_BUILD)
#    define NF_THOMAS_RGB16_F32_API __declspec(dllexport)
#  else
#    define NF_THOMAS_RGB16_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_THOMAS_RGB16_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_THOMAS_RGB16_F32_ABI_VERSION_V1 1u

typedef enum nf_thomas_rgb16_f32_status_v1 {
    NF_THOMAS_RGB16_F32_OK_V1 = 0,
    NF_THOMAS_RGB16_F32_INVALID_ARGUMENT_V1 = 1,
    NF_THOMAS_RGB16_F32_INVALID_PROFILE_V1 = 2,
    NF_THOMAS_RGB16_F32_DOMAIN_ERROR_V1 = 3,
    NF_THOMAS_RGB16_F32_CALLBACK_FAILED_V1 = 4
} nf_thomas_rgb16_f32_status_v1;

typedef int (*nf_thomas_rgb16_f32_sink_v1)(
    void* context,
    size_t row_start,
    size_t row_count,
    const uint16_t* rgb16,
    size_t rgb16_values);

NF_THOMAS_RGB16_F32_API uint32_t
nf_thomas_rgb16_f32_abi_version_v1(void);

NF_THOMAS_RGB16_F32_API nf_thomas_rgb16_f32_status_v1
nf_thomas_rgb16_f32_workspace_bytes_v1(
    size_t width,
    size_t row_partition,
    size_t* workspace_bytes);

/*
 * Complete generic RGB sample program: exposure -> amplitude -> Thomas
 * transmittance -> neutral gauge -> exact sRGB16 quantization. The sink view
 * is valid only during the callback; durable atomicity remains caller-owned.
 */
NF_THOMAS_RGB16_F32_API nf_thomas_rgb16_f32_status_v1
nf_thomas_rgb16_f32_apply_v1(
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile,
    const nf_thomas_field_f32_profile_v1 field_profiles[3],
    const nf_neutral_gauge_f32_profile_v1* gauge_profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    const float* relative_log_exposure_chw,
    size_t exposure_floats,
    void* workspace,
    size_t workspace_bytes,
    nf_thomas_rgb16_f32_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]);

#ifdef __cplusplus
}
#endif

#endif
