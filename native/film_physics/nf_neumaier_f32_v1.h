#ifndef NF_NEUMAIER_F32_V1_H
#define NF_NEUMAIER_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NF_NEUMAIER_F32_BUILD)
#    define NF_NEUMAIER_F32_API __declspec(dllexport)
#  else
#    define NF_NEUMAIER_F32_API __declspec(dllimport)
#  endif
#else
#  define NF_NEUMAIER_F32_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_NEUMAIER_F32_ABI_VERSION_V1 1u

NF_NEUMAIER_F32_API uint32_t nf_neumaier_f32_abi_version_v1(void);

/* Update one caller-held ordered float32-to-double Neumaier accumulation. */
NF_NEUMAIER_F32_API int nf_neumaier_f32_accumulate_v1(
    const float* values,
    size_t value_count,
    double* total,
    double* compensation);

#ifdef __cplusplus
}
#endif

#endif
