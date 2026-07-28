#ifndef NF_NEUTRAL_GAUGE_F32_V1_H
#define NF_NEUTRAL_GAUGE_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_NEUTRAL_GAUGE_F32_BUILD)
#    define NF_NEUTRAL_GAUGE_F32_API __declspec(dllexport)
#  else
#    define NF_NEUTRAL_GAUGE_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_NEUTRAL_GAUGE_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_NEUTRAL_GAUGE_F32_ABI_VERSION_V1 1u
#define NF_NEUTRAL_GAUGE_F32_MAX_KNOTS_V1 1025u

typedef enum nf_neutral_gauge_f32_status_v1 {
    NF_NEUTRAL_GAUGE_F32_OK_V1 = 0,
    NF_NEUTRAL_GAUGE_F32_INVALID_ARGUMENT_V1 = 1,
    NF_NEUTRAL_GAUGE_F32_INVALID_PROFILE_V1 = 2,
    NF_NEUTRAL_GAUGE_F32_INVALID_INPUT_V1 = 3,
    NF_NEUTRAL_GAUGE_F32_NUMERIC_FAILURE_V1 = 4
} nf_neutral_gauge_f32_status_v1;

typedef struct nf_neutral_gauge_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    uint32_t knot_count[3];
    double x_knots[3][NF_NEUTRAL_GAUGE_F32_MAX_KNOTS_V1];
    double y_knots[3][NF_NEUTRAL_GAUGE_F32_MAX_KNOTS_V1];
    double derivatives[3][NF_NEUTRAL_GAUGE_F32_MAX_KNOTS_V1];
} nf_neutral_gauge_f32_profile_v1;

NF_NEUTRAL_GAUGE_F32_API uint32_t
nf_neutral_gauge_f32_abi_version_v1(void);

NF_NEUTRAL_GAUGE_F32_API nf_neutral_gauge_f32_status_v1
nf_neutral_gauge_f32_validate_profile_v1(
    const nf_neutral_gauge_f32_profile_v1* profile);

/*
 * Input and output contain rgb_count interleaved relative-linear float32 RGB
 * triplets. Exact in-place operation is supported. Validation failures leave
 * output untouched.
 */
NF_NEUTRAL_GAUGE_F32_API nf_neutral_gauge_f32_status_v1
nf_neutral_gauge_f32_apply_v1(
    const nf_neutral_gauge_f32_profile_v1* profile,
    const float* scan_linear_rgb,
    size_t rgb_count,
    float* gauged_linear_rgb);

#ifdef __cplusplus
}
#endif

#endif
