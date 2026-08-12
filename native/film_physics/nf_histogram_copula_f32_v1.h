#ifndef NF_HISTOGRAM_COPULA_F32_V1_H
#define NF_HISTOGRAM_COPULA_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef NF_HISTOGRAM_COPULA_F32_BUILD
#    define NF_HISTOGRAM_COPULA_F32_API __declspec(dllexport)
#  else
#    define NF_HISTOGRAM_COPULA_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_HISTOGRAM_COPULA_F32_API
#endif

#define NF_HISTOGRAM_COPULA_F32_ABI_VERSION_V1 1u

typedef enum nf_histogram_copula_f32_status_v1 {
    NF_HISTOGRAM_COPULA_F32_OK_V1 = 0,
    NF_HISTOGRAM_COPULA_F32_INVALID_ARGUMENT_V1 = 1,
    NF_HISTOGRAM_COPULA_F32_INSUFFICIENT_WORKSPACE_V1 = 2,
    NF_HISTOGRAM_COPULA_F32_NONFINITE_INPUT_V1 = 3,
    NF_HISTOGRAM_COPULA_F32_DEGENERATE_INPUT_V1 = 4,
    NF_HISTOGRAM_COPULA_F32_INVALID_CORRELATION_V1 = 5
} nf_histogram_copula_f32_status_v1;

typedef struct nf_histogram_copula_f32_diagnostics_v1 {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t rank_bins;
    size_t sample_count;
    size_t workspace_bytes;
    double input_normal_correlation[9];
    double output_uniform_correlation[9];
    double output_uniform_mean[3];
} nf_histogram_copula_f32_diagnostics_v1;

NF_HISTOGRAM_COPULA_F32_API uint32_t
nf_histogram_copula_f32_abi_version_v1(void);

NF_HISTOGRAM_COPULA_F32_API nf_histogram_copula_f32_status_v1
nf_histogram_copula_f32_workspace_bytes_v1(
    size_t sample_count,
    uint32_t rank_bins,
    size_t* workspace_bytes);

NF_HISTOGRAM_COPULA_F32_API nf_histogram_copula_f32_status_v1
nf_histogram_copula_f32_apply_v1(
    const float* interleaved_fields,
    size_t sample_count,
    const double* target_correlation_3x3,
    uint32_t rank_bins,
    void* workspace,
    size_t workspace_bytes,
    float* interleaved_uniforms,
    size_t output_count,
    nf_histogram_copula_f32_diagnostics_v1* diagnostics);

#endif
