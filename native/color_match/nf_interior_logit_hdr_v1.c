#define NF_INTERIOR_LOGIT_HDR_BUILD
#include "nf_interior_logit_hdr_v1.h"

#include <math.h>
#include <stdint.h>

#define NF_OUTPUT_MAXIMUM 10000.0f
#define NF_INTERIOR_EPSILON (1.0f / 65536.0f)
#define NF_MAXIMUM_EXTRAPOLATION_SLOPE 1.0f

static int nf_ranges_overlap(const void* first, const void* second, size_t bytes) {
    const uintptr_t first_begin = (uintptr_t)first;
    const uintptr_t second_begin = (uintptr_t)second;
    const uintptr_t first_end = first_begin + bytes;
    const uintptr_t second_end = second_begin + bytes;
    if (first_end < first_begin || second_end < second_begin) {
        return 1;
    }
    return first_begin < second_end && second_begin < first_end;
}

static int nf_knots_valid(const float knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1]) {
    size_t row;
    size_t channel;
    for (channel = 0; channel < NF_INTERIOR_LOGIT_HDR_CHANNELS_V1; ++channel) {
        if (!isfinite(knots[channel])) {
            return 0;
        }
        for (row = 1; row < NF_INTERIOR_LOGIT_HDR_KNOT_COUNT_V1; ++row) {
            const float value = knots[row * 3u + channel];
            const float previous = knots[(row - 1u) * 3u + channel];
            if (!isfinite(value) || !(value > previous)) {
                return 0;
            }
        }
    }
    return 1;
}

static float nf_clampf(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

static float nf_logit(float value) {
    const float normalized = value / NF_OUTPUT_MAXIMUM;
    const float bounded = nf_clampf(
        normalized,
        NF_INTERIOR_EPSILON,
        1.0f - NF_INTERIOR_EPSILON);
    return (float)(log((double)bounded) - log1p(-(double)bounded));
}

static float nf_interpolate(
    float value,
    const float source_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    const float reference_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    size_t channel) {
    size_t row;
    size_t lower = 0u;
    float x0;
    float x1;
    float y0;
    float y1;
    float slope;
    if (value < source_knots[channel]) {
        x0 = source_knots[channel];
        x1 = source_knots[3u + channel];
        y0 = reference_knots[channel];
        y1 = reference_knots[3u + channel];
        slope = nf_clampf((y1 - y0) / (x1 - x0), 0.0f, NF_MAXIMUM_EXTRAPOLATION_SLOPE);
        return y0 + slope * (value - x0);
    }
    if (value > source_knots[96u + channel]) {
        x0 = source_knots[93u + channel];
        x1 = source_knots[96u + channel];
        y0 = reference_knots[93u + channel];
        y1 = reference_knots[96u + channel];
        slope = nf_clampf((y1 - y0) / (x1 - x0), 0.0f, NF_MAXIMUM_EXTRAPOLATION_SLOPE);
        return y1 + slope * (value - x1);
    }
    for (row = 1u; row < NF_INTERIOR_LOGIT_HDR_KNOT_COUNT_V1; ++row) {
        if (value < source_knots[row * 3u + channel]) {
            break;
        }
        lower = row;
    }
    if (lower >= NF_INTERIOR_LOGIT_HDR_KNOT_COUNT_V1 - 1u) {
        lower = NF_INTERIOR_LOGIT_HDR_KNOT_COUNT_V1 - 2u;
    }
    x0 = source_knots[lower * 3u + channel];
    x1 = source_knots[(lower + 1u) * 3u + channel];
    y0 = reference_knots[lower * 3u + channel];
    y1 = reference_knots[(lower + 1u) * 3u + channel];
    value = nf_clampf(value, x0, x1);
    return (float)(
        (double)y0 +
        (((double)value - (double)x0) / ((double)x1 - (double)x0)) *
            ((double)y1 - (double)y0));
}

static float nf_sigmoid(float value) {
    if (value >= 0.0f) {
        return (float)(1.0 / (1.0 + exp(-(double)value)));
    }
    {
        const double exponent = exp((double)value);
        return (float)(exponent / (1.0 + exponent));
    }
}

static int nf_transform_value(
    float input,
    const float source_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    const float reference_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    uint32_t identity,
    size_t channel,
    float* output) {
    float mapped;
    if (!isfinite(input) || input < 0.0f || input > NF_OUTPUT_MAXIMUM) {
        return 0;
    }
    if (identity != 0u || input == 0.0f || input == NF_OUTPUT_MAXIMUM) {
        *output = input;
        return 1;
    }
    mapped = nf_interpolate(
        nf_logit(input), source_knots, reference_knots, channel);
    mapped = NF_OUTPUT_MAXIMUM * nf_sigmoid(mapped);
    if (!isfinite(mapped) || !(mapped > 0.0f) || !(mapped < NF_OUTPUT_MAXIMUM)) {
        return 0;
    }
    *output = mapped;
    return 1;
}

uint32_t nf_interior_logit_hdr_abi_version_v1(void) {
    return NF_INTERIOR_LOGIT_HDR_ABI_VERSION_V1;
}

nf_interior_logit_hdr_status_v1 nf_interior_logit_hdr_apply_v1(
    const float source_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    const float reference_knots[NF_INTERIOR_LOGIT_HDR_KNOT_VALUES_V1],
    uint32_t identity,
    const float* input_rgb,
    size_t triplet_count,
    float* output_rgb,
    size_t output_value_count,
    nf_interior_logit_hdr_diagnostics_v1* diagnostics) {
    size_t value_count;
    size_t index;
    uint64_t boundaries = 0u;
    uint64_t interiors = 0u;
    nf_interior_logit_hdr_diagnostics_v1 result_diagnostics;
    if (
        source_knots == NULL || reference_knots == NULL || input_rgb == NULL ||
        output_rgb == NULL || diagnostics == NULL || triplet_count == 0u ||
        triplet_count > SIZE_MAX / 3u || identity > 1u
    ) {
        return NF_INTERIOR_LOGIT_HDR_INVALID_ARGUMENT_V1;
    }
    value_count = triplet_count * 3u;
    if (output_value_count < value_count || value_count > SIZE_MAX / sizeof(float)) {
        return NF_INTERIOR_LOGIT_HDR_INVALID_ARGUMENT_V1;
    }
    if (!nf_knots_valid(source_knots) || !nf_knots_valid(reference_knots)) {
        return NF_INTERIOR_LOGIT_HDR_INVALID_KNOTS_V1;
    }
    if (
        input_rgb != output_rgb &&
        nf_ranges_overlap(input_rgb, output_rgb, value_count * sizeof(float))
    ) {
        return NF_INTERIOR_LOGIT_HDR_OVERLAP_V1;
    }
    for (index = 0u; index < value_count; ++index) {
        float transformed;
        if (!nf_transform_value(
                input_rgb[index], source_knots, reference_knots, identity,
                index % 3u, &transformed)) {
            if (!isfinite(input_rgb[index]) || input_rgb[index] < 0.0f || input_rgb[index] > NF_OUTPUT_MAXIMUM) {
                return NF_INTERIOR_LOGIT_HDR_INVALID_INPUT_V1;
            }
            return NF_INTERIOR_LOGIT_HDR_NUMERIC_FAILURE_V1;
        }
        if (input_rgb[index] == 0.0f || input_rgb[index] == NF_OUTPUT_MAXIMUM) {
            ++boundaries;
        } else {
            ++interiors;
        }
    }
    for (index = 0u; index < value_count; ++index) {
        float transformed;
        (void)nf_transform_value(
            input_rgb[index], source_knots, reference_knots, identity,
            index % 3u, &transformed);
        output_rgb[index] = transformed;
    }
    result_diagnostics.abi_version = NF_INTERIOR_LOGIT_HDR_ABI_VERSION_V1;
    result_diagnostics.identity = identity;
    result_diagnostics.triplet_count = (uint64_t)triplet_count;
    result_diagnostics.preserved_boundary_values = boundaries;
    result_diagnostics.strict_interior_values = interiors;
    *diagnostics = result_diagnostics;
    return NF_INTERIOR_LOGIT_HDR_OK_V1;
}
