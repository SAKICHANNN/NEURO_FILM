#ifndef NF_AO6_BASE_F32_V2_H
#define NF_AO6_BASE_F32_V2_H

#include "nf_ao6_base_f32_v1.h"

#if defined(_WIN32)
#  if defined(NF_AO6_BASE_F32_V2_BUILD)
#    define NF_AO6_BASE_F32_V2_API __declspec(dllexport)
#  else
#    define NF_AO6_BASE_F32_V2_API __declspec(dllimport)
#  endif
#else
#  define NF_AO6_BASE_F32_V2_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NF_AO6_BASE_F32_ABI_VERSION_V2 2u

/*
 * Same profile/context/wire result as v1. V2 analytically resolves the
 * legacy 14-step final scale when the full-chroma target is already in gamut.
 */
NF_AO6_BASE_F32_V2_API nf_ao6_base_f32_status_v1
nf_ao6_base_f32_apply_v2(
    const nf_ao6_base_f32_profile_v1* profile,
    const nf_ao6_base_f32_context_v1* context,
    const float* encoded_srgb,
    size_t rgb_count,
    float* output_encoded_srgb);

#ifdef __cplusplus
}
#endif

#endif
