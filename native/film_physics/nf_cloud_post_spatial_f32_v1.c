#define NF_CLOUD_POST_SPATIAL_F32_BUILD
#include "nf_cloud_post_spatial_f32_v1.h"

uint32_t nf_cloud_post_spatial_f32_abi_version_v1(void) {
    return NF_CLOUD_POST_SPATIAL_F32_ABI_VERSION_V1;
}

int nf_cloud_post_spatial_f32_required_halo_v1(
    const nf_gaussian_f32_profile_v1* adjacency_blur_profile,
    const nf_gaussian_f32_profile_v1* dye_diffusion_profile,
    const nf_gaussian_f32_profile_v1* scanner_mtf_profile,
    uint32_t* required_halo) {
    uint32_t first=0u, second=0u, third=0u;
    if (required_halo == NULL ||
        nf_gaussian_f32_required_halo_v1(adjacency_blur_profile,&first)!=0 ||
        nf_gaussian_f32_required_halo_v1(dye_diffusion_profile,&second)!=0 ||
        nf_gaussian_f32_required_halo_v1(scanner_mtf_profile,&third)!=0 ||
        first > UINT32_MAX-second || first+second > UINT32_MAX-third)
        return 1;
    *required_halo=first+second+third;
    return 0;
}

int nf_cloud_post_spatial_f32_apply_core_v1(
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
    float* output_scan_linear_rgb, size_t output_values) {
    uint32_t halo=0u; size_t values, core_values, index, source_index;
    int status;
    if (cloud_density_rgb==NULL || gaussian_workspace==NULL ||
        blurred_density_workspace==NULL || adjacent_density_workspace==NULL ||
        diffused_density_workspace==NULL || interpreted_workspace==NULL ||
        scan_workspace==NULL || output_scan_linear_rgb==NULL ||
        input_height==0u || width==0u || full_height==0u || core_height==0u ||
        core_logical_start>full_height ||
        core_height>full_height-core_logical_start ||
        input_height>SIZE_MAX/width || input_height*width>SIZE_MAX/3u ||
        core_height>SIZE_MAX/width || core_height*width>SIZE_MAX/3u ||
        nf_cloud_post_spatial_f32_required_halo_v1(
            adjacency_blur_profile,dye_diffusion_profile,scanner_mtf_profile,&halo)!=0 ||
        core_offset!=(core_logical_start<(size_t)halo?
            core_logical_start:(size_t)halo) || core_offset>input_height ||
        core_height>input_height-core_offset ||
        input_height-(core_offset+core_height)!=
            (full_height-(core_logical_start+core_height)<(size_t)halo?
             full_height-(core_logical_start+core_height):(size_t)halo)) return 1;
    values=input_height*width*3u; core_values=core_height*width*3u;
    if(workspace_values<values || output_values<core_values)return 1;
    status=(int)nf_gaussian_f32_apply_v1(adjacency_blur_profile,cloud_density_rgb,
        input_height,width,gaussian_workspace,workspace_values,blurred_density_workspace);
    if(status!=0)return 10+status;
    status=(int)nf_bounded_adjacency_f32_apply_v1(adjacency_profile,cloud_density_rgb,
        blurred_density_workspace,input_height*width,adjacent_density_workspace);
    if(status!=0)return 20+status;
    status=(int)nf_gaussian_f32_apply_v1(dye_diffusion_profile,adjacent_density_workspace,
        input_height,width,gaussian_workspace,workspace_values,diffused_density_workspace);
    if(status!=0)return 30+status;
    status=(int)nf_physical_interpretation_f32_apply_v1(domains_profile,
        diffused_density_workspace,input_height*width,interpreted_workspace);
    if(status!=0)return 40+status;
    status=(int)nf_gaussian_f32_apply_v1(scanner_mtf_profile,interpreted_workspace,
        input_height,width,gaussian_workspace,workspace_values,scan_workspace);
    if(status!=0)return 50+status;
    source_index=core_offset*width*3u;
    for(index=0u;index<core_values;++index)
        output_scan_linear_rgb[index]=scan_workspace[source_index+index];
    return 0;
}
