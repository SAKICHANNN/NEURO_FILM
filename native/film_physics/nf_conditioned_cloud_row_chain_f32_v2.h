#ifndef NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_V2_H
#define NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_V2_H

#include "nf_conditioned_cloud_row_chain_f32_v1.h"

#define NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_ABI_VERSION_V2 2u

NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_API uint32_t
nf_conditioned_cloud_row_chain_f32_abi_version_v2(void);

NF_CONDITIONED_CLOUD_ROW_CHAIN_F32_API
nf_conditioned_cloud_row_chain_f32_status_v1
nf_conditioned_cloud_row_chain_f32_apply_v2(
    const nf_density_conditioned_poisson_u16_profile_v3* count_profile,
    const nf_cloud_spatial_response_f32_profile_v2* spatial_profile,
    size_t full_height, size_t width, size_t origin_y, size_t core_height,
    size_t halo, const double* full_scale_cmy, size_t scale_values,
    const float* expected_transmittance_cmy, size_t expected_values,
    const float channel_gain_cmy[3], uint16_t* count_workspace,
    size_t count_workspace_values, double* convolution_workspace,
    size_t convolution_workspace_doubles, float* spatial_density_workspace,
    float* spatial_transmittance_workspace, size_t spatial_workspace_values,
    float* output_density_cmy, float* output_transmittance_cmy,
    size_t output_values);

#endif
