#define NF_HISTOGRAM_COPULA_F32_BUILD
#include "nf_histogram_copula_f32_v1.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int nf_mul_size(size_t a, size_t b, size_t* result) {
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *result = a * b;
    return 1;
}

static int nf_add_size(size_t a, size_t b, size_t* result) {
    if (b > SIZE_MAX - a) {
        return 0;
    }
    *result = a + b;
    return 1;
}

static size_t nf_align8(size_t value) {
    return (value + 7u) & ~(size_t)7u;
}

static int nf_overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const uintptr_t a0 = (uintptr_t)a;
    const uintptr_t b0 = (uintptr_t)b;
    const uintptr_t a1 = a0 + a_bytes;
    const uintptr_t b1 = b0 + b_bytes;
    if (a1 < a0 || b1 < b0) {
        return 1;
    }
    return a0 < b1 && b0 < a1;
}

static uint32_t nf_quantize(double value, double minimum, double maximum, uint32_t bins) {
    const double scaled = (value - minimum) * (double)(bins - 1u) / (maximum - minimum);
    double rounded = floor(scaled + 0.5);
    if (rounded < 0.0) {
        rounded = 0.0;
    } else if (rounded > (double)(bins - 1u)) {
        rounded = (double)(bins - 1u);
    }
    return (uint32_t)rounded;
}

static double nf_inverse_normal(double p) {
    static const double a[6] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};
    static const double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    const double low = 0.02425;
    const double high = 1.0 - low;
    double q;
    double r;
    if (!(p > 0.0 && p < 1.0)) {
        return NAN;
    }
    if (p < low) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > high) {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

static int nf_cholesky3(const double matrix[9], double lower[9]) {
    size_t i;
    size_t j;
    size_t k;
    memset(lower, 0, 9u * sizeof(double));
    for (i = 0u; i < 3u; ++i) {
        for (j = 0u; j <= i; ++j) {
            double sum = matrix[i * 3u + j];
            for (k = 0u; k < j; ++k) {
                sum -= lower[i * 3u + k] * lower[j * 3u + k];
            }
            if (i == j) {
                if (!isfinite(sum) || sum <= DBL_EPSILON) {
                    return 0;
                }
                lower[i * 3u + j] = sqrt(sum);
            } else {
                lower[i * 3u + j] = sum / lower[j * 3u + j];
            }
        }
    }
    return 1;
}

static void nf_solve_lower3(const double lower[9], const double value[3], double output[3]) {
    output[0] = value[0] / lower[0];
    output[1] = (value[1] - lower[3] * output[0]) / lower[4];
    output[2] = (value[2] - lower[6] * output[0] - lower[7] * output[1]) / lower[8];
}

static void nf_multiply_lower3(const double lower[9], const double value[3], double output[3]) {
    output[0] = lower[0] * value[0];
    output[1] = lower[3] * value[0] + lower[4] * value[1];
    output[2] = lower[6] * value[0] + lower[7] * value[1] + lower[8] * value[2];
}

static void nf_correlated_value(
    const float* fields,
    size_t index,
    uint32_t bins,
    const double raw_min[3],
    const double raw_max[3],
    const double* raw_table,
    const double mean[3],
    const double input_lower[9],
    const double target_lower[9],
    double output[3]) {
    double centered[3];
    double whitened[3];
    size_t channel;
    for (channel = 0u; channel < 3u; ++channel) {
        const uint32_t bin = nf_quantize(
            (double)fields[index * 3u + channel], raw_min[channel], raw_max[channel], bins);
        centered[channel] = raw_table[channel * (size_t)bins + bin] - mean[channel];
    }
    nf_solve_lower3(input_lower, centered, whitened);
    nf_multiply_lower3(target_lower, whitened, output);
}

uint32_t nf_histogram_copula_f32_abi_version_v1(void) {
    return NF_HISTOGRAM_COPULA_F32_ABI_VERSION_V1;
}

nf_histogram_copula_f32_status_v1 nf_histogram_copula_f32_workspace_bytes_v1(
    size_t sample_count, uint32_t rank_bins, size_t* workspace_bytes) {
    size_t histogram_count;
    size_t histogram_bytes;
    size_t table_bytes;
    size_t output_count;
    size_t output_bytes;
    size_t total;
    if (workspace_bytes == NULL || sample_count < 2u || sample_count > UINT64_MAX / 2u ||
        rank_bins < 2u || rank_bins > 65536u ||
        !nf_mul_size((size_t)rank_bins, 3u, &histogram_count) ||
        !nf_mul_size(histogram_count, sizeof(uint64_t), &histogram_bytes) ||
        !nf_mul_size(histogram_count, sizeof(double), &table_bytes) ||
        !nf_mul_size(sample_count, 3u, &output_count) ||
        !nf_mul_size(output_count, sizeof(float), &output_bytes) ||
        !nf_add_size(nf_align8(histogram_bytes), nf_align8(table_bytes), &total) ||
        !nf_add_size(total, nf_align8(histogram_bytes), &total) ||
        !nf_add_size(total, nf_align8(output_bytes), &total)) {
        return NF_HISTOGRAM_COPULA_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_bytes = total;
    return NF_HISTOGRAM_COPULA_F32_OK_V1;
}

nf_histogram_copula_f32_status_v1 nf_histogram_copula_f32_apply_v1(
    const float* fields, size_t sample_count, const double* target, uint32_t bins,
    void* workspace, size_t workspace_bytes, float* output, size_t output_count,
    nf_histogram_copula_f32_diagnostics_v1* diagnostics) {
    size_t required;
    size_t field_count;
    size_t field_bytes;
    size_t offset;
    uint64_t* raw_counts;
    double* raw_table;
    uint64_t* correlated_counts;
    float* temporary_output;
    double raw_min[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
    double raw_max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    double correlated_min[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
    double correlated_max[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    double sum[3] = {0.0, 0.0, 0.0};
    double cross[9] = {0.0};
    double mean[3];
    double covariance[9];
    double input_correlation[9];
    double input_lower[9];
    double target_lower[9];
    double output_sum[3] = {0.0, 0.0, 0.0};
    double output_cross[9] = {0.0};
    double output_correlation[9];
    nf_histogram_copula_f32_diagnostics_v1 local_diagnostics;
    size_t i;
    size_t j;
    size_t k;
    if (fields == NULL || target == NULL || workspace == NULL || output == NULL || diagnostics == NULL ||
        diagnostics->struct_size != sizeof(nf_histogram_copula_f32_diagnostics_v1) ||
        diagnostics->abi_version != NF_HISTOGRAM_COPULA_F32_ABI_VERSION_V1 ||
        nf_histogram_copula_f32_workspace_bytes_v1(sample_count, bins, &required) != NF_HISTOGRAM_COPULA_F32_OK_V1 ||
        workspace_bytes < required || !nf_mul_size(sample_count, 3u, &field_count) ||
        output_count != field_count || !nf_mul_size(field_count, sizeof(float), &field_bytes) ||
        nf_overlap(fields, field_bytes, output, field_bytes) ||
        nf_overlap(workspace, workspace_bytes, output, field_bytes) ||
        nf_overlap(workspace, workspace_bytes, fields, field_bytes)) {
        return NF_HISTOGRAM_COPULA_F32_INVALID_ARGUMENT_V1;
    }
    for (i = 0u; i < 9u; ++i) {
        if (!isfinite(target[i]) || fabs(target[i] - target[(i % 3u) * 3u + i / 3u]) > 1e-15) {
            return NF_HISTOGRAM_COPULA_F32_INVALID_CORRELATION_V1;
        }
    }
    if (target[0] != 1.0 || target[4] != 1.0 || target[8] != 1.0 ||
        !nf_cholesky3(target, target_lower)) {
        return NF_HISTOGRAM_COPULA_F32_INVALID_CORRELATION_V1;
    }
    offset = 0u;
    raw_counts = (uint64_t*)((uint8_t*)workspace + offset);
    offset += nf_align8(3u * (size_t)bins * sizeof(uint64_t));
    raw_table = (double*)((uint8_t*)workspace + offset);
    offset += nf_align8(3u * (size_t)bins * sizeof(double));
    correlated_counts = (uint64_t*)((uint8_t*)workspace + offset);
    offset += nf_align8(3u * (size_t)bins * sizeof(uint64_t));
    temporary_output = (float*)((uint8_t*)workspace + offset);
    memset(raw_counts, 0, 3u * (size_t)bins * sizeof(uint64_t));
    memset(correlated_counts, 0, 3u * (size_t)bins * sizeof(uint64_t));
    for (i = 0u; i < sample_count; ++i) {
        for (j = 0u; j < 3u; ++j) {
            const double value = (double)fields[i * 3u + j];
            if (!isfinite(value)) {
                return NF_HISTOGRAM_COPULA_F32_NONFINITE_INPUT_V1;
            }
            if (value < raw_min[j]) raw_min[j] = value;
            if (value > raw_max[j]) raw_max[j] = value;
        }
    }
    if (raw_max[0] <= raw_min[0] || raw_max[1] <= raw_min[1] || raw_max[2] <= raw_min[2]) {
        return NF_HISTOGRAM_COPULA_F32_DEGENERATE_INPUT_V1;
    }
    for (i = 0u; i < sample_count; ++i) {
        for (j = 0u; j < 3u; ++j) {
            const uint32_t bin = nf_quantize((double)fields[i * 3u + j], raw_min[j], raw_max[j], bins);
            raw_counts[j * (size_t)bins + bin] += 1u;
        }
    }
    for (j = 0u; j < 3u; ++j) {
        uint64_t before = 0u;
        for (i = 0u; i < (size_t)bins; ++i) {
            const uint64_t count = raw_counts[j * (size_t)bins + i];
            const double p = ((double)before + 0.5 * (double)count) / (double)sample_count;
            raw_table[j * (size_t)bins + i] = count == 0u ? 0.0 : nf_inverse_normal(p);
            before += count;
        }
    }
    for (i = 0u; i < sample_count; ++i) {
        double normal[3];
        for (j = 0u; j < 3u; ++j) {
            const uint32_t bin = nf_quantize((double)fields[i * 3u + j], raw_min[j], raw_max[j], bins);
            normal[j] = raw_table[j * (size_t)bins + bin];
            sum[j] += normal[j];
        }
        for (j = 0u; j < 3u; ++j) for (k = 0u; k < 3u; ++k) cross[j * 3u + k] += normal[j] * normal[k];
    }
    for (j = 0u; j < 3u; ++j) mean[j] = sum[j] / (double)sample_count;
    for (j = 0u; j < 3u; ++j) for (k = 0u; k < 3u; ++k)
        covariance[j * 3u + k] = cross[j * 3u + k] - (double)sample_count * mean[j] * mean[k];
    for (j = 0u; j < 3u; ++j) for (k = 0u; k < 3u; ++k)
        input_correlation[j * 3u + k] = covariance[j * 3u + k] / sqrt(covariance[j * 3u + j] * covariance[k * 3u + k]);
    if (!nf_cholesky3(input_correlation, input_lower)) {
        return NF_HISTOGRAM_COPULA_F32_DEGENERATE_INPUT_V1;
    }
    for (i = 0u; i < sample_count; ++i) {
        double value[3];
        nf_correlated_value(fields, i, bins, raw_min, raw_max, raw_table, mean, input_lower, target_lower, value);
        for (j = 0u; j < 3u; ++j) {
            if (value[j] < correlated_min[j]) correlated_min[j] = value[j];
            if (value[j] > correlated_max[j]) correlated_max[j] = value[j];
        }
    }
    for (i = 0u; i < sample_count; ++i) {
        double value[3];
        nf_correlated_value(fields, i, bins, raw_min, raw_max, raw_table, mean, input_lower, target_lower, value);
        for (j = 0u; j < 3u; ++j) {
            const uint32_t bin = nf_quantize(value[j], correlated_min[j], correlated_max[j], bins);
            correlated_counts[j * (size_t)bins + bin] += 1u;
        }
    }
    for (j = 0u; j < 3u; ++j) {
        uint64_t before = 0u;
        for (i = 0u; i < (size_t)bins; ++i) {
            const uint64_t count = correlated_counts[j * (size_t)bins + i];
            raw_counts[j * (size_t)bins + i] = 2u * before + count;
            before += count;
        }
    }
    for (i = 0u; i < sample_count; ++i) {
        double value[3];
        nf_correlated_value(fields, i, bins, raw_min, raw_max, raw_table, mean, input_lower, target_lower, value);
        for (j = 0u; j < 3u; ++j) {
            uint32_t bin;
            double uniform;
            bin = nf_quantize(value[j], correlated_min[j], correlated_max[j], bins);
            uniform = (double)raw_counts[j * (size_t)bins + bin] /
                (2.0 * (double)sample_count);
            temporary_output[i * 3u + j] = (float)uniform;
            output_sum[j] += (double)temporary_output[i * 3u + j];
        }
        for (j = 0u; j < 3u; ++j) for (k = 0u; k < 3u; ++k)
            output_cross[j * 3u + k] += (double)temporary_output[i * 3u + j] * (double)temporary_output[i * 3u + k];
    }
    for (j = 0u; j < 3u; ++j) mean[j] = output_sum[j] / (double)sample_count;
    for (j = 0u; j < 3u; ++j) for (k = 0u; k < 3u; ++k) {
        const double cov = output_cross[j * 3u + k] - (double)sample_count * mean[j] * mean[k];
        const double var_j = output_cross[j * 3u + j] - (double)sample_count * mean[j] * mean[j];
        const double var_k = output_cross[k * 3u + k] - (double)sample_count * mean[k] * mean[k];
        output_correlation[j * 3u + k] = cov / sqrt(var_j * var_k);
    }
    memset(&local_diagnostics, 0, sizeof(local_diagnostics));
    local_diagnostics.struct_size = sizeof(local_diagnostics);
    local_diagnostics.abi_version = NF_HISTOGRAM_COPULA_F32_ABI_VERSION_V1;
    local_diagnostics.rank_bins = bins;
    local_diagnostics.sample_count = sample_count;
    local_diagnostics.workspace_bytes = required;
    memcpy(local_diagnostics.input_normal_correlation, input_correlation, sizeof(input_correlation));
    memcpy(local_diagnostics.output_uniform_correlation, output_correlation, sizeof(output_correlation));
    memcpy(local_diagnostics.output_uniform_mean, mean, sizeof(mean));
    memcpy(output, temporary_output, field_bytes);
    *diagnostics = local_diagnostics;
    return NF_HISTOGRAM_COPULA_F32_OK_V1;
}
