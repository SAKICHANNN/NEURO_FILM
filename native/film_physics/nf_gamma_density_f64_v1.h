#ifndef NF_GAMMA_DENSITY_F64_V1_H
#define NF_GAMMA_DENSITY_F64_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define NF_GAMMA_API __declspec(dllexport)
#else
#define NF_GAMMA_API __attribute__((visibility("default")))
#endif

typedef struct nf_gamma_density_diagnostics_v1 {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t inverse_iterations;
    size_t sample_count;
    double minimum_output;
    double maximum_output;
} nf_gamma_density_diagnostics_v1;

NF_GAMMA_API uint32_t nf_gamma_density_f64_abi_version_v1(void);
NF_GAMMA_API int nf_gamma_density_f64_apply_v1(
    const double *uniforms,
    const double *shapes,
    const double *scales,
    size_t sample_count,
    uint32_t inverse_iterations,
    double *output,
    size_t output_count,
    nf_gamma_density_diagnostics_v1 *diagnostics);

#endif
