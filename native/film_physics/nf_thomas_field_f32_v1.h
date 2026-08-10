#ifndef NF_THOMAS_FIELD_F32_V1_H
#define NF_THOMAS_FIELD_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_THOMAS_FIELD_F32_BUILD)
#    define NF_THOMAS_FIELD_F32_API __declspec(dllexport)
#  else
#    define NF_THOMAS_FIELD_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_THOMAS_FIELD_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_THOMAS_FIELD_F32_ABI_VERSION_V1 1u
#define NF_THOMAS_FIELD_F32_MAX_RADIUS_V1 64u

typedef enum nf_thomas_field_f32_status_v1 {
    NF_THOMAS_FIELD_F32_OK_V1 = 0,
    NF_THOMAS_FIELD_F32_INVALID_ARGUMENT_V1 = 1,
    NF_THOMAS_FIELD_F32_INVALID_PROFILE_V1 = 2
} nf_thomas_field_f32_status_v1;

typedef struct nf_thomas_field_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    double particle_sigma_pixels;
    double cluster_sigma_pixels;
    double mean_offspring;
    double truncate;
    uint64_t component_seeds[2];
    uint64_t realization_seed;
} nf_thomas_field_f32_profile_v1;

NF_THOMAS_FIELD_F32_API uint32_t
nf_thomas_field_f32_abi_version_v1(void);

NF_THOMAS_FIELD_F32_API nf_thomas_field_f32_status_v1
nf_thomas_field_f32_validate_profile_v1(
    const nf_thomas_field_f32_profile_v1* profile);

NF_THOMAS_FIELD_F32_API nf_thomas_field_f32_status_v1
nf_thomas_field_f32_workspace_floats_v1(
    size_t height,
    size_t width,
    size_t* workspace_floats);

/*
 * Generates the unchanged finite-support two-component Thomas field and
 * removes only its full-field mean. Workspace contains 3*height*width floats.
 * Workspace and output must be distinct. Invalid requests leave both intact.
 */
NF_THOMAS_FIELD_F32_API nf_thomas_field_f32_status_v1
nf_thomas_field_f32_apply_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t height,
    size_t width,
    float* workspace,
    size_t workspace_floats,
    float* output,
    size_t output_floats,
    double* raw_mean);

#ifdef __cplusplus
}
#endif

#endif
