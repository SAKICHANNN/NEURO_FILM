#ifndef NF_THOMAS_RGB16_CACHED_F32_V1_H
#define NF_THOMAS_RGB16_CACHED_F32_V1_H

#include "nf_thomas_rgb16_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_THOMAS_RGB16_CACHED_F32_BUILD)
#    define NF_THOMAS_RGB16_CACHED_F32_API __declspec(dllexport)
#  else
#    define NF_THOMAS_RGB16_CACHED_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_THOMAS_RGB16_CACHED_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_THOMAS_RGB16_CACHED_F32_ABI_VERSION_V1 1u

typedef enum nf_thomas_rgb16_cached_f32_status_v1 {
    NF_THOMAS_RGB16_CACHED_F32_OK_V1 = 0,
    NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1 = 1,
    NF_THOMAS_RGB16_CACHED_F32_INVALID_PROFILE_V1 = 2,
    NF_THOMAS_RGB16_CACHED_F32_DOMAIN_ERROR_V1 = 3,
    NF_THOMAS_RGB16_CACHED_F32_CALLBACK_FAILED_V1 = 4,
    NF_THOMAS_RGB16_CACHED_F32_EXECUTION_FAILED_V1 = 5
} nf_thomas_rgb16_cached_f32_status_v1;

NF_THOMAS_RGB16_CACHED_F32_API uint32_t
nf_thomas_rgb16_cached_f32_abi_version_v1(void);

NF_THOMAS_RGB16_CACHED_F32_API nf_thomas_rgb16_cached_f32_status_v1
nf_thomas_rgb16_cached_f32_workspace_bytes_v1(
    size_t full_height,
    size_t width,
    size_t row_partition,
    uint32_t parallel_layers,
    size_t* workspace_bytes);

/*
 * Exact cached variant of nf_thomas_rgb16_f32_apply_v1. Three full-resolution
 * uncentered Thomas fields are retained, so the second rendering pass is
 * removed. parallel_layers is exactly 1 or 3. Layer-local generation,
 * Neumaier accumulation and development order remain unchanged; the sink is
 * always invoked in original row-major order.
 */
NF_THOMAS_RGB16_CACHED_F32_API nf_thomas_rgb16_cached_f32_status_v1
nf_thomas_rgb16_cached_f32_apply_v1(
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile,
    const nf_thomas_field_f32_profile_v1 field_profiles[3],
    const nf_neutral_gauge_f32_profile_v1* gauge_profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    uint32_t parallel_layers,
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
