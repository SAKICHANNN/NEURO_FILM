#define NF_THOMAS_FIELD_F32_BUILD
#include "nf_thomas_field_f32_v1.h"

#include <math.h>
#include <stdint.h>

#define NF_THOMAS_FIELD_F32_MAX_DIAMETER_V1 \
    (2u * NF_THOMAS_FIELD_F32_MAX_RADIUS_V1 + 1u)
#define NF_TWO_PI 6.2831853071795864769252867665590057683943387987502

static uint64_t nf_splitmix64(uint64_t value) {
    uint64_t state = value + UINT64_C(0x9E3779B97F4A7C15);
    state = (state ^ (state >> 30u)) * UINT64_C(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27u)) * UINT64_C(0x94D049BB133111EB);
    return state ^ (state >> 31u);
}

static double nf_counter_normal(uint64_t counter) {
    const uint64_t first = nf_splitmix64(counter);
    const uint64_t second = nf_splitmix64(
        counter ^ UINT64_C(0xD2B74407B1CE6E93));
    const double inverse_2pow53 = 1.0 / 9007199254740992.0;
    const double u1 = ((double)(first >> 11u) + 0.5) * inverse_2pow53;
    const double u2 = ((double)(second >> 11u) + 0.5) * inverse_2pow53;
    return sqrt(-2.0 * log(u1)) * cos(NF_TWO_PI * u2);
}

static uint32_t nf_radius(double sigma, double truncate) {
    const double scaled = sigma * truncate;
    if (!isfinite(scaled) || scaled <= 0.0 ||
        scaled + 0.5 >= (double)(NF_THOMAS_FIELD_F32_MAX_RADIUS_V1 + 1u)) {
        return NF_THOMAS_FIELD_F32_MAX_RADIUS_V1 + 1u;
    }
    return (uint32_t)(scaled + 0.5);
}

static int nf_profile_valid(const nf_thomas_field_f32_profile_v1* profile) {
    const double combined = profile == NULL ? 0.0 : hypot(
        profile->particle_sigma_pixels, profile->cluster_sigma_pixels);
    return profile != NULL &&
        profile->struct_size == sizeof(nf_thomas_field_f32_profile_v1) &&
        profile->abi_version == NF_THOMAS_FIELD_F32_ABI_VERSION_V1 &&
        isfinite(profile->particle_sigma_pixels) &&
        profile->particle_sigma_pixels > 0.0 &&
        isfinite(profile->cluster_sigma_pixels) &&
        profile->cluster_sigma_pixels > 0.0 &&
        isfinite(profile->mean_offspring) &&
        profile->mean_offspring > 0.0 &&
        isfinite(profile->truncate) &&
        profile->truncate > 0.0 && profile->truncate <= 8.0 &&
        nf_radius(profile->particle_sigma_pixels, profile->truncate) <=
            NF_THOMAS_FIELD_F32_MAX_RADIUS_V1 &&
        nf_radius(combined, profile->truncate) <=
            NF_THOMAS_FIELD_F32_MAX_RADIUS_V1;
}

static int nf_ranges_overlap(
    const float* first, size_t first_count,
    const float* second, size_t second_count) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const size_t first_bytes = first_count * sizeof(float);
    const size_t second_bytes = second_count * sizeof(float);
    const uintptr_t first_end = first_start + first_bytes;
    const uintptr_t second_end = second_start + second_bytes;
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

static void nf_build_kernel(
    double sigma,
    uint32_t radius,
    double weights[NF_THOMAS_FIELD_F32_MAX_DIAMETER_V1],
    double* variance_2d) {
    uint32_t index;
    double sum = 0.0;
    double squared_sum = 0.0;
    for (index = 0; index <= 2u * radius; ++index) {
        const double offset = (double)((int64_t)index - (int64_t)radius);
        const double value = exp(-0.5 * (offset / sigma) * (offset / sigma));
        weights[index] = value;
        sum += value;
    }
    for (index = 0; index <= 2u * radius; ++index) {
        weights[index] /= sum;
        squared_sum += weights[index] * weights[index];
    }
    *variance_2d = squared_sum * squared_sum;
}

static void nf_blur_constant(
    float* field,
    float* temporary,
    size_t height,
    size_t width,
    const double weights[NF_THOMAS_FIELD_F32_MAX_DIAMETER_V1],
    uint32_t radius) {
    size_t y;
    size_t x;
    uint32_t kernel;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            double value = 0.0;
            for (kernel = 0; kernel <= 2u * radius; ++kernel) {
                const int64_t sx = (int64_t)x + (int64_t)kernel - (int64_t)radius;
                if (sx >= 0 && (uint64_t)sx < (uint64_t)width) {
                    value += weights[kernel] * (double)field[y * width + (size_t)sx];
                }
            }
            temporary[y * width + x] = (float)value;
        }
    }
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            double value = 0.0;
            for (kernel = 0; kernel <= 2u * radius; ++kernel) {
                const int64_t sy = (int64_t)y + (int64_t)kernel - (int64_t)radius;
                if (sy >= 0 && (uint64_t)sy < (uint64_t)height) {
                    value += weights[kernel] * (double)temporary[(size_t)sy * width + x];
                }
            }
            field[y * width + x] = (float)value;
        }
    }
}

static void nf_neumaier_add(double* total, double* compensation, double value) {
    const double updated = *total + value;
    if (fabs(*total) >= fabs(value)) {
        *compensation += (*total - updated) + value;
    } else {
        *compensation += (value - updated) + *total;
    }
    *total = updated;
}

uint32_t nf_thomas_field_f32_abi_version_v1(void) {
    return NF_THOMAS_FIELD_F32_ABI_VERSION_V1;
}

nf_thomas_field_f32_status_v1 nf_thomas_field_f32_validate_profile_v1(
    const nf_thomas_field_f32_profile_v1* profile) {
    if (profile == NULL) {
        return NF_THOMAS_FIELD_F32_INVALID_ARGUMENT_V1;
    }
    return nf_profile_valid(profile) ? NF_THOMAS_FIELD_F32_OK_V1 :
        NF_THOMAS_FIELD_F32_INVALID_PROFILE_V1;
}

nf_thomas_field_f32_status_v1 nf_thomas_field_f32_workspace_floats_v1(
    size_t height, size_t width, size_t* workspace_floats) {
    if (workspace_floats == NULL || height == 0u || width == 0u ||
        height > SIZE_MAX / width || height * width > SIZE_MAX / 3u) {
        return NF_THOMAS_FIELD_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_floats = 3u * height * width;
    return NF_THOMAS_FIELD_F32_OK_V1;
}

nf_thomas_field_f32_status_v1 nf_thomas_field_f32_apply_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t height,
    size_t width,
    float* workspace,
    size_t workspace_floats,
    float* output,
    size_t output_floats,
    double* raw_mean) {
    size_t count;
    size_t required;
    size_t index;
    float* first;
    float* second;
    float* temporary;
    double particle_weights[NF_THOMAS_FIELD_F32_MAX_DIAMETER_V1];
    double combined_weights[NF_THOMAS_FIELD_F32_MAX_DIAMETER_V1];
    double particle_variance;
    double combined_variance;
    double total = 0.0;
    double compensation = 0.0;
    double mean;
    double normalization;
    uint32_t particle_radius;
    uint32_t combined_radius;
    const nf_thomas_field_f32_status_v1 profile_status =
        nf_thomas_field_f32_validate_profile_v1(profile);
    if (profile_status != NF_THOMAS_FIELD_F32_OK_V1) {
        return profile_status;
    }
    if (nf_thomas_field_f32_workspace_floats_v1(
            height, width, &required) != NF_THOMAS_FIELD_F32_OK_V1 ||
        workspace == NULL || output == NULL || raw_mean == NULL) {
        return NF_THOMAS_FIELD_F32_INVALID_ARGUMENT_V1;
    }
    count = height * width;
    if (workspace_floats < required || output_floats < count ||
        nf_ranges_overlap(workspace, required, output, count)) {
        return NF_THOMAS_FIELD_F32_INVALID_ARGUMENT_V1;
    }
    particle_radius = nf_radius(profile->particle_sigma_pixels, profile->truncate);
    combined_radius = nf_radius(
        hypot(profile->particle_sigma_pixels, profile->cluster_sigma_pixels),
        profile->truncate);
    nf_build_kernel(
        profile->particle_sigma_pixels, particle_radius,
        particle_weights, &particle_variance);
    nf_build_kernel(
        hypot(profile->particle_sigma_pixels, profile->cluster_sigma_pixels),
        combined_radius, combined_weights, &combined_variance);
    normalization = sqrt(
        particle_variance + profile->mean_offspring * combined_variance);
    if (!isfinite(normalization) || normalization <= 0.0) {
        return NF_THOMAS_FIELD_F32_INVALID_PROFILE_V1;
    }
    first = workspace;
    second = workspace + count;
    temporary = workspace + 2u * count;
    for (index = 0; index < count; ++index) {
        const uint64_t counter = (uint64_t)index;
        first[index] = (float)nf_counter_normal(
            counter + (profile->component_seeds[0] ^ profile->realization_seed));
        second[index] = (float)nf_counter_normal(
            counter + (profile->component_seeds[1] ^ profile->realization_seed));
    }
    nf_blur_constant(
        first, temporary, height, width, particle_weights, particle_radius);
    nf_blur_constant(
        second, temporary, height, width, combined_weights, combined_radius);
    for (index = 0; index < count; ++index) {
        const double value = (
            (double)first[index] + sqrt(profile->mean_offspring) * (double)second[index]
        ) / normalization;
        output[index] = (float)value;
        nf_neumaier_add(&total, &compensation, (double)output[index]);
    }
    mean = (total + compensation) / (double)count;
    for (index = 0; index < count; ++index) {
        output[index] = (float)((double)output[index] - mean);
    }
    *raw_mean = mean;
    return NF_THOMAS_FIELD_F32_OK_V1;
}
