#ifndef NF_GAMMA_DENSITY_FAST_F64_V1_H
#define NF_GAMMA_DENSITY_FAST_F64_V1_H

#include "nf_gamma_density_hybrid_f64_v1.h"

typedef struct nf_gamma_density_fast_diagnostics_v1 {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t direct_iterations;
    uint32_t newton_iterations;
    size_t sample_count;
    size_t direct_branch_count;
    size_t newton_branch_count;
    size_t asymptotic_branch_count;
    double minimum_output;
    double maximum_output;
} nf_gamma_density_fast_diagnostics_v1;

NF_GAMMA_API uint32_t nf_gamma_density_fast_f64_abi_version_v1(void);
NF_GAMMA_API int nf_gamma_density_fast_f64_apply_v1(
    const double *uniforms,
    const double *shapes,
    const double *scales,
    size_t sample_count,
    uint32_t direct_iterations,
    uint32_t newton_iterations,
    double direct_shape_upper,
    double newton_shape_upper,
    double *output,
    size_t output_count,
    nf_gamma_density_fast_diagnostics_v1 *diagnostics);

#endif
