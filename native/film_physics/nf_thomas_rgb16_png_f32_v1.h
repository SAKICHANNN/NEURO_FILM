#ifndef NF_THOMAS_RGB16_PNG_F32_V1_H
#define NF_THOMAS_RGB16_PNG_F32_V1_H

#include "nf_thomas_rgb16_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_THOMAS_RGB16_PNG_F32_BUILD)
#    define NF_THOMAS_RGB16_PNG_F32_API __declspec(dllexport)
#  else
#    define NF_THOMAS_RGB16_PNG_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_THOMAS_RGB16_PNG_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_THOMAS_RGB16_PNG_F32_ABI_VERSION_V1 1u
#define NF_THOMAS_RGB16_PNG_F32_MAX_IDAT_PAYLOAD_V1 65536u

typedef enum nf_thomas_rgb16_png_f32_status_v1 {
    NF_THOMAS_RGB16_PNG_F32_OK_V1 = 0,
    NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1 = 1,
    NF_THOMAS_RGB16_PNG_F32_INVALID_PROFILE_V1 = 2,
    NF_THOMAS_RGB16_PNG_F32_DOMAIN_ERROR_V1 = 3,
    NF_THOMAS_RGB16_PNG_F32_CALLBACK_FAILED_V1 = 4,
    NF_THOMAS_RGB16_PNG_F32_ENCODING_ERROR_V1 = 5
} nf_thomas_rgb16_png_f32_status_v1;

typedef int (*nf_thomas_rgb16_png_f32_byte_sink_v1)(
    void* context,
    const uint8_t* bytes,
    size_t byte_count);

NF_THOMAS_RGB16_PNG_F32_API uint32_t
nf_thomas_rgb16_png_f32_abi_version_v1(void);

NF_THOMAS_RGB16_PNG_F32_API nf_thomas_rgb16_png_f32_status_v1
nf_thomas_rgb16_png_f32_workspace_bytes_v1(
    size_t width,
    size_t row_partition,
    size_t* workspace_bytes);

/*
 * Complete generic PNG stream program. The emitted stream is deterministic
 * RGB16 PNG with filter 0, zlib stored blocks and the fixed sRGB ICC profile.
 * A failed byte sink can leave an incomplete stream; durable publication and
 * rollback remain caller-owned.
 */
NF_THOMAS_RGB16_PNG_F32_API nf_thomas_rgb16_png_f32_status_v1
nf_thomas_rgb16_png_f32_apply_v1(
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
    nf_thomas_rgb16_png_f32_byte_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]);

#ifdef __cplusplus
}
#endif

#endif
