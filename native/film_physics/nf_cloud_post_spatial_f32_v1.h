#ifndef NF_CLOUD_POST_SPATIAL_F32_V1_H
#define NF_CLOUD_POST_SPATIAL_F32_V1_H

#include "nf_bounded_adjacency_f32_v1.h"
#include "nf_gaussian_rgb_f32_v1.h"
#include "nf_physical_domains_f32_v1.h"

#if defined(_WIN32)
# if defined(NF_CLOUD_POST_SPATIAL_F32_BUILD)
#  define NF_CLOUD_POST_SPATIAL_F32_API __declspec(dllexport)
# else
#  define NF_CLOUD_POST_SPATIAL_F32_API __declspec(dllimport)
# endif
#else
# define NF_CLOUD_POST_SPATIAL_F32_API
#endif

#define NF_CLOUD_POST_SPATIAL_F32_ABI_VERSION_V1 1u

NF_CLOUD_POST_SPATIAL_F32_API uint32_t
nf_cloud_post_spatial_f32_abi_version_v1(void);

NF_CLOUD_POST_SPATIAL_F32_API int
nf_cloud_post_spatial_f32_required_halo_v1(
    const nf_gaussian_f32_profile_v1* adjacency_blur_profile,
    const nf_gaussian_f32_profile_v1* dye_diffusion_profile,
    const nf_gaussian_f32_profile_v1* scanner_mtf_profile,
    uint32_t* required_halo);

NF_CLOUD_POST_SPATIAL_F32_API int
nf_cloud_post_spatial_f32_apply_core_v1(
    const nf_physical_domains_f32_profile_v1* domains_profile,
    const nf_gaussian_f32_profile_v1* adjacency_blur_profile,
    const nf_bounded_adjacency_f32_profile_v1* adjacency_profile,
    const nf_gaussian_f32_profile_v1* dye_diffusion_profile,
    const nf_gaussian_f32_profile_v1* scanner_mtf_profile,
    const float* cloud_density_rgb, size_t input_height, size_t width,
    size_t full_height, size_t core_logical_start,
    size_t core_offset, size_t core_height, float* gaussian_workspace,
    float* blurred_density_workspace, float* adjacent_density_workspace,
    float* diffused_density_workspace, float* interpreted_workspace,
    float* scan_workspace, size_t workspace_values,
    float* output_scan_linear_rgb, size_t output_values);

#endif
