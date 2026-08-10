#ifndef NF_GRANULARITY_AMPLITUDE_F32_V1_H
#define NF_GRANULARITY_AMPLITUDE_F32_V1_H

#include <stddef.h>
#include <stdint.h>

#define NF_GRANULARITY_AMPLITUDE_F32_ABI_VERSION_V1 1u
#define NF_GRANULARITY_AMPLITUDE_F32_MAX_KNOTS_V1 16u

#if defined(_WIN32) && defined(NF_GRANULARITY_AMPLITUDE_F32_BUILD)
#define NF_GRANULARITY_AMPLITUDE_F32_API __declspec(dllexport)
#elif defined(_WIN32)
#define NF_GRANULARITY_AMPLITUDE_F32_API __declspec(dllimport)
#else
#define NF_GRANULARITY_AMPLITUDE_F32_API __attribute__((visibility("default")))
#endif

typedef enum nf_granularity_amplitude_f32_status_v1 {
    NF_GRANULARITY_AMPLITUDE_F32_OK_V1 = 0,
    NF_GRANULARITY_AMPLITUDE_F32_INVALID_ARGUMENT_V1 = 1,
    NF_GRANULARITY_AMPLITUDE_F32_INVALID_PROFILE_V1 = 2,
    NF_GRANULARITY_AMPLITUDE_F32_DOMAIN_ERROR_V1 = 3
} nf_granularity_amplitude_f32_status_v1;

typedef struct nf_granularity_amplitude_f32_profile_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    char source_profile_sha256[65];
    uint32_t knot_count[3];
    double log_exposure_knots[3][NF_GRANULARITY_AMPLITUDE_F32_MAX_KNOTS_V1];
    double density_knots[3][NF_GRANULARITY_AMPLITUDE_F32_MAX_KNOTS_V1];
    double channel_floor_variance[3];
    double shared_amplitude;
    double measurement_energy;
} nf_granularity_amplitude_f32_profile_v1;

NF_GRANULARITY_AMPLITUDE_F32_API uint32_t
nf_granularity_amplitude_f32_abi_version_v1(void);

NF_GRANULARITY_AMPLITUDE_F32_API nf_granularity_amplitude_f32_status_v1
nf_granularity_amplitude_f32_validate_profile_v1(
    const nf_granularity_amplitude_f32_profile_v1* profile);

/* Input and both outputs are planar RGB arrays with 3*sample_count floats. */
NF_GRANULARITY_AMPLITUDE_F32_API nf_granularity_amplitude_f32_status_v1
nf_granularity_amplitude_f32_apply_v1(
    const nf_granularity_amplitude_f32_profile_v1* profile,
    const float* relative_log_exposure_chw,
    size_t sample_count,
    float* developed_density_chw,
    float* point_density_sigma_chw);

/* One planar layer, for serial bounded-memory RGB composition. */
NF_GRANULARITY_AMPLITUDE_F32_API nf_granularity_amplitude_f32_status_v1
nf_granularity_amplitude_f32_apply_layer_v1(
    const nf_granularity_amplitude_f32_profile_v1* profile,
    uint32_t channel,
    const float* relative_log_exposure,
    size_t sample_count,
    float* developed_density,
    float* point_density_sigma);

#endif
