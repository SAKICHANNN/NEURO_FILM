#define NF_THOMAS_ROWS_F32_BUILD
#define NF_THOMAS_FIELD_F32_BUILD
#include "nf_thomas_rows_f32_v1.h"

#include <math.h>
#include <stdint.h>

#define NF_ROWS_DIAMETER (2u * NF_THOMAS_FIELD_F32_MAX_RADIUS_V1 + 1u)
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
    double sigma, uint32_t radius, double weights[NF_ROWS_DIAMETER],
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

static void nf_component_rows(
    size_t full_height, size_t width, size_t row_start, size_t row_count,
    uint32_t radius, const double weights[NF_ROWS_DIAMETER], uint64_t seed,
    float* raw, float* horizontal, float* candidate, int second,
    double offspring_sqrt, double normalization) {
    const size_t window_start = row_start > radius ? row_start - radius : 0u;
    const size_t requested_stop = row_start + row_count;
    const size_t window_stop = requested_stop + radius < full_height ?
        requested_stop + radius : full_height;
    const size_t window_rows = window_stop - window_start;
    size_t local_y;
    size_t x;
    uint32_t kernel;
    for (local_y = 0; local_y < window_rows; ++local_y) {
        const size_t global_y = window_start + local_y;
        for (x = 0; x < width; ++x) {
            const uint64_t counter = (uint64_t)(global_y * width + x);
            raw[local_y * width + x] = (float)nf_counter_normal(counter + seed);
        }
    }
    for (local_y = 0; local_y < window_rows; ++local_y) {
        for (x = 0; x < width; ++x) {
            double value = 0.0;
            for (kernel = 0; kernel <= 2u * radius; ++kernel) {
                const int64_t sx = (int64_t)x + (int64_t)kernel - (int64_t)radius;
                if (sx >= 0 && (uint64_t)sx < (uint64_t)width) {
                    value += weights[kernel] * (double)raw[local_y * width + (size_t)sx];
                }
            }
            horizontal[local_y * width + x] = (float)value;
        }
    }
    for (local_y = 0; local_y < row_count; ++local_y) {
        const size_t global_y = row_start + local_y;
        for (x = 0; x < width; ++x) {
            double value = 0.0;
            for (kernel = 0; kernel <= 2u * radius; ++kernel) {
                const int64_t sy = (int64_t)global_y + (int64_t)kernel - (int64_t)radius;
                if (sy >= 0 && (uint64_t)sy < (uint64_t)full_height) {
                    value += weights[kernel] *
                        (double)horizontal[((size_t)sy - window_start) * width + x];
                }
            }
            if (second) {
                candidate[local_y * width + x] = (float)(
                    ((double)candidate[local_y * width + x] +
                     offspring_sqrt * (double)(float)value) / normalization);
            } else {
                candidate[local_y * width + x] = (float)value;
            }
        }
    }
}

static nf_thomas_rows_f32_status_v1 nf_render_rows(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height, size_t width, size_t row_start, size_t row_count,
    float* workspace, size_t workspace_floats, float** candidate_out) {
    size_t required;
    const size_t padded_rows = row_count + 2u * NF_THOMAS_FIELD_F32_MAX_RADIUS_V1;
    const size_t plane = padded_rows * width;
    float* raw = workspace;
    float* horizontal = workspace + plane;
    float* candidate = workspace + 2u * plane;
    double particle_weights[NF_ROWS_DIAMETER];
    double combined_weights[NF_ROWS_DIAMETER];
    double particle_variance;
    double combined_variance;
    double normalization;
    const double combined_sigma = hypot(
        profile->particle_sigma_pixels, profile->cluster_sigma_pixels);
    const uint32_t particle_radius = nf_radius(
        profile->particle_sigma_pixels, profile->truncate);
    const uint32_t combined_radius = nf_radius(combined_sigma, profile->truncate);
    if (nf_thomas_rows_f32_workspace_floats_v1(width, row_count, &required) !=
            NF_THOMAS_ROWS_F32_OK_V1 || workspace_floats < required) {
        return NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1;
    }
    nf_build_kernel(
        profile->particle_sigma_pixels, particle_radius,
        particle_weights, &particle_variance);
    nf_build_kernel(
        combined_sigma, combined_radius, combined_weights, &combined_variance);
    normalization = sqrt(
        particle_variance + profile->mean_offspring * combined_variance);
    if (!isfinite(normalization) || normalization <= 0.0) {
        return NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1;
    }
    nf_component_rows(
        full_height, width, row_start, row_count, particle_radius,
        particle_weights, profile->component_seeds[0] ^ profile->realization_seed,
        raw, horizontal, candidate, 0, 0.0, normalization);
    nf_component_rows(
        full_height, width, row_start, row_count, combined_radius,
        combined_weights, profile->component_seeds[1] ^ profile->realization_seed,
        raw, horizontal, candidate, 1, sqrt(profile->mean_offspring), normalization);
    *candidate_out = candidate;
    return NF_THOMAS_ROWS_F32_OK_V1;
}

uint32_t nf_thomas_rows_f32_abi_version_v1(void) {
    return NF_THOMAS_ROWS_F32_ABI_VERSION_V1;
}

nf_thomas_rows_f32_status_v1 nf_thomas_rows_f32_workspace_floats_v1(
    size_t width, size_t row_count, size_t* workspace_floats) {
    size_t padded_rows;
    if (workspace_floats == NULL || width == 0u || row_count == 0u ||
        row_count > SIZE_MAX - 2u * NF_THOMAS_FIELD_F32_MAX_RADIUS_V1) {
        return NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1;
    }
    padded_rows = row_count + 2u * NF_THOMAS_FIELD_F32_MAX_RADIUS_V1;
    if (padded_rows > SIZE_MAX / width ||
        2u * padded_rows * width > SIZE_MAX - row_count * width) {
        return NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_floats = (2u * padded_rows + row_count) * width;
    return NF_THOMAS_ROWS_F32_OK_V1;
}

static nf_thomas_rows_f32_status_v1 nf_validate_request(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height, size_t width, size_t row_start, size_t row_count,
    float* workspace, size_t workspace_floats, float* output, size_t output_floats) {
    size_t required;
    size_t count;
    if (nf_thomas_field_f32_validate_profile_v1(profile) !=
            NF_THOMAS_FIELD_F32_OK_V1) {
        return NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1;
    }
    if (full_height == 0u || width == 0u || row_count == 0u ||
        row_start >= full_height || row_count > full_height - row_start ||
        row_count > SIZE_MAX / width || workspace == NULL || output == NULL ||
        nf_thomas_rows_f32_workspace_floats_v1(width, row_count, &required) !=
            NF_THOMAS_ROWS_F32_OK_V1) {
        return NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1;
    }
    count = row_count * width;
    if (workspace_floats < required || output_floats < count ||
        nf_ranges_overlap(workspace, required, output, count)) {
        return NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1;
    }
    return NF_THOMAS_ROWS_F32_OK_V1;
}

nf_thomas_rows_f32_status_v1 nf_thomas_rows_f32_field_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height, size_t width, size_t row_start, size_t row_count,
    float* workspace, size_t workspace_floats,
    float* output, size_t output_floats) {
    float* candidate;
    size_t index;
    const size_t count = row_count * width;
    nf_thomas_rows_f32_status_v1 status = nf_validate_request(
        profile, full_height, width, row_start, row_count,
        workspace, workspace_floats, output, output_floats);
    if (status != NF_THOMAS_ROWS_F32_OK_V1) {
        return status;
    }
    status = nf_render_rows(
        profile, full_height, width, row_start, row_count,
        workspace, workspace_floats, &candidate);
    if (status != NF_THOMAS_ROWS_F32_OK_V1) {
        return status;
    }
    for (index = 0; index < count; ++index) {
        output[index] = candidate[index];
    }
    return NF_THOMAS_ROWS_F32_OK_V1;
}

nf_thomas_rows_f32_status_v1 nf_thomas_rows_f32_density_v1(
    const nf_thomas_field_f32_profile_v1* profile,
    size_t full_height, size_t width, size_t row_start, size_t row_count,
    const float* base_density, size_t base_density_floats,
    const float* point_density_sigma, size_t point_density_sigma_floats,
    double raw_field_mean, float* workspace, size_t workspace_floats,
    float* output_transmittance, size_t output_floats) {
    float* candidate;
    size_t required;
    size_t index;
    const size_t count = row_count * width;
    nf_thomas_rows_f32_status_v1 status = nf_validate_request(
        profile, full_height, width, row_start, row_count,
        workspace, workspace_floats, output_transmittance, output_floats);
    if (status != NF_THOMAS_ROWS_F32_OK_V1 || base_density == NULL ||
        point_density_sigma == NULL || !isfinite(raw_field_mean) ||
        base_density_floats < count || point_density_sigma_floats < count ||
        nf_thomas_rows_f32_workspace_floats_v1(width, row_count, &required) !=
            NF_THOMAS_ROWS_F32_OK_V1 ||
        nf_ranges_overlap(base_density, count, point_density_sigma, count) ||
        nf_ranges_overlap(base_density, count, workspace, required) ||
        nf_ranges_overlap(base_density, count, output_transmittance, count) ||
        nf_ranges_overlap(point_density_sigma, count, workspace, required) ||
        nf_ranges_overlap(point_density_sigma, count, output_transmittance, count)) {
        return status == NF_THOMAS_ROWS_F32_INVALID_PROFILE_V1 ? status :
            NF_THOMAS_ROWS_F32_INVALID_ARGUMENT_V1;
    }
    for (index = 0; index < count; ++index) {
        if (!isfinite((double)base_density[index]) || base_density[index] < 0.0f ||
            !isfinite((double)point_density_sigma[index]) ||
            point_density_sigma[index] < 0.0f) {
            return NF_THOMAS_ROWS_F32_DOMAIN_ERROR_V1;
        }
    }
    status = nf_render_rows(
        profile, full_height, width, row_start, row_count,
        workspace, workspace_floats, &candidate);
    if (status != NF_THOMAS_ROWS_F32_OK_V1) {
        return status;
    }
    for (index = 0; index < count; ++index) {
        const float centered = (float)((double)candidate[index] - raw_field_mean);
        const double density = (double)base_density[index] +
            (double)point_density_sigma[index] * centered;
        const double transmittance = pow(10.0, -density);
        if (!isfinite(density) || density < 0.0 || !isfinite(transmittance) ||
            transmittance <= 0.0 || transmittance > 1.0) {
            return NF_THOMAS_ROWS_F32_DOMAIN_ERROR_V1;
        }
        candidate[index] = (float)transmittance;
    }
    for (index = 0; index < count; ++index) {
        output_transmittance[index] = candidate[index];
    }
    return NF_THOMAS_ROWS_F32_OK_V1;
}
