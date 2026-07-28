#ifndef NF_GAUSSIAN_RGB_F32_V1_H
#define NF_GAUSSIAN_RGB_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_GAUSSIAN_F32_BUILD)
#    define NF_GAUSSIAN_F32_API __declspec(dllexport)
#  else
#    define NF_GAUSSIAN_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_GAUSSIAN_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_GAUSSIAN_F32_ABI_VERSION_V1 1u
#define NF_GAUSSIAN_F32_MAX_RADIUS_V1 64u

typedef enum nf_gaussian_f32_status_v1 {
    NF_GAUSSIAN_F32_OK_V1 = 0,
    NF_GAUSSIAN_F32_INVALID_ARGUMENT_V1 = 1,
    NF_GAUSSIAN_F32_INVALID_PROFILE_V1 = 2,
    NF_GAUSSIAN_F32_INVALID_INPUT_V1 = 3
} nf_gaussian_f32_status_v1;

typedef struct nf_gaussian_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_component_sha256[65];
    double sigma_pixels_rgb[3];
    double truncate;
} nf_gaussian_f32_profile_v1;

NF_GAUSSIAN_F32_API uint32_t
nf_gaussian_f32_abi_version_v1(void);

NF_GAUSSIAN_F32_API nf_gaussian_f32_status_v1
nf_gaussian_f32_validate_profile_v1(
    const nf_gaussian_f32_profile_v1* profile);

NF_GAUSSIAN_F32_API nf_gaussian_f32_status_v1
nf_gaussian_f32_required_halo_v1(
    const nf_gaussian_f32_profile_v1* profile,
    uint32_t* halo);

/*
 * Arrays contain height*width*3 float32 samples. Workspace and output must
 * be distinct non-overlapping ranges. Accumulation is float64 and each pass
 * rounds once to float32. Boundary extension is nearest.
 */
NF_GAUSSIAN_F32_API nf_gaussian_f32_status_v1
nf_gaussian_f32_apply_v1(
    const nf_gaussian_f32_profile_v1* profile,
    const float* input_rgb,
    size_t height,
    size_t width,
    float* workspace_rgb,
    size_t workspace_floats,
    float* output_rgb);

#ifdef __cplusplus
}
#endif

#endif
