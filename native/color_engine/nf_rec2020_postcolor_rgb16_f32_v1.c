#include "nf_rec2020_postcolor_rgb16_f32_v1.h"

#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

typedef struct nf_postcolor_context {
    const float *source;
    const float *candidate;
    const float *thresholds;
    const uint32_t *buckets;
    uint16_t *output;
    float *scale;
    uint64_t begin;
    uint64_t end;
    double margin;
    int status;
} nf_postcolor_context;

static uint16_t nf_quantize(
    float linear, const float *thresholds, const uint32_t *buckets) {
    uint32_t bits;
    uint32_t prefix;
    uint32_t low;
    uint32_t high;
    memcpy(&bits, &linear, sizeof(bits));
    prefix = bits >> 16U;
    low = buckets[prefix];
    high = buckets[prefix + 1U];
    while (low < high) {
        uint32_t mid = low + (high - low) / 2U;
        if (linear >= thresholds[mid]) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    return (uint16_t)low;
}

static int nf_postcolor_range(nf_postcolor_context *context) {
    uint64_t index;
    for (index = context->begin; index < context->end; ++index) {
        double source[3];
        double residual[3];
        double scale = 1.0;
        uint32_t channel;
        for (channel = 0; channel < 3; ++channel) {
            double value;
            double lower;
            double upper;
            source[channel] = (double)context->source[index * 3U + channel];
            residual[channel] =
                (double)context->candidate[index * 3U + channel] - source[channel];
            value = residual[channel];
            lower = fmin(source[channel], context->margin);
            upper = fmax(source[channel], 1.0 - context->margin);
            if (value > 0.0 && source[channel] + value > upper) {
                scale = fmin(scale, (upper - source[channel]) / value);
            } else if (value < 0.0 && source[channel] + value < lower) {
                scale = fmin(scale, (lower - source[channel]) / value);
            }
        }
        scale = fmax(0.0, fmin(1.0, scale));
        context->scale[index] = (float)scale;
        for (channel = 0; channel < 3; ++channel) {
            float linear = (float)(source[channel] + scale * residual[channel]);
            if (!isfinite(linear) || linear < 0.0F || linear > 1.0F) {
                return NF_POSTCOLOR_INVALID_OUTPUT;
            }
            context->output[index * 3U + channel] =
                nf_quantize(linear, context->thresholds, context->buckets);
        }
    }
    return NF_POSTCOLOR_OK;
}

#if defined(_WIN32)
static DWORD WINAPI nf_thread_entry(LPVOID value) {
    nf_postcolor_context *context = (nf_postcolor_context *)value;
    context->status = nf_postcolor_range(context);
    return 0;
}
#endif

int nf_rec2020_postcolor_rgb16_f32_v1(
    const float *source_rgb,
    const float *candidate_rgb,
    const float *quantization_thresholds,
    const uint32_t *quantization_buckets,
    uint16_t *output_rgb16,
    float *residual_scale,
    uint64_t pixel_count,
    uint32_t threshold_count,
    uint32_t bucket_count,
    double margin,
    uint32_t thread_count) {
    nf_postcolor_context serial;
    uint64_t index;
    if (source_rgb == NULL || candidate_rgb == NULL ||
        quantization_thresholds == NULL || quantization_buckets == NULL ||
        output_rgb16 == NULL ||
        residual_scale == NULL || pixel_count == 0U || !isfinite(margin) ||
        margin <= 0.0 || margin >= 0.5 || threshold_count != 65535U ||
        bucket_count != 65537U ||
        thread_count == 0U || thread_count > 64U) {
        return NF_POSTCOLOR_INVALID_ARGUMENT;
    }
    for (index = 0U; index < bucket_count; ++index) {
        if (quantization_buckets[index] > threshold_count ||
            (index > 0U && quantization_buckets[index] <
                               quantization_buckets[index - 1U])) {
            return NF_POSTCOLOR_INVALID_ARGUMENT;
        }
    }
    for (index = 0U; index < threshold_count; ++index) {
        if (!isfinite(quantization_thresholds[index]) ||
            quantization_thresholds[index] < 0.0F ||
            quantization_thresholds[index] > 1.0F ||
            (index > 0U && quantization_thresholds[index] <
                               quantization_thresholds[index - 1U])) {
            return NF_POSTCOLOR_INVALID_ARGUMENT;
        }
    }
    for (index = 0; index < pixel_count * 3U; ++index) {
        if (!isfinite(source_rgb[index]) || !isfinite(candidate_rgb[index]) ||
            source_rgb[index] < 0.0F || source_rgb[index] > 1.0F ||
            candidate_rgb[index] < 0.0F || candidate_rgb[index] > 1.0F) {
            return NF_POSTCOLOR_INVALID_INPUT;
        }
    }
    serial.source = source_rgb;
    serial.candidate = candidate_rgb;
    serial.thresholds = quantization_thresholds;
    serial.buckets = quantization_buckets;
    serial.output = output_rgb16;
    serial.scale = residual_scale;
    serial.begin = 0U;
    serial.end = pixel_count;
    serial.margin = margin;
    serial.status = NF_POSTCOLOR_OK;
#if defined(_WIN32)
    if (thread_count > 1U && pixel_count >= (uint64_t)thread_count) {
        HANDLE *handles = (HANDLE *)calloc(thread_count, sizeof(HANDLE));
        nf_postcolor_context *contexts =
            (nf_postcolor_context *)calloc(thread_count, sizeof(nf_postcolor_context));
        uint32_t created = 0U;
        uint32_t thread;
        if (handles != NULL && contexts != NULL) {
            for (thread = 0U; thread < thread_count; ++thread) {
                contexts[thread] = serial;
                contexts[thread].begin = pixel_count * thread / thread_count;
                contexts[thread].end = pixel_count * (thread + 1U) / thread_count;
                handles[thread] = CreateThread(
                    NULL, 0, nf_thread_entry, &contexts[thread], 0, NULL);
                if (handles[thread] == NULL) {
                    break;
                }
                ++created;
            }
            for (thread = 0U; thread < created; ++thread) {
                (void)WaitForSingleObject(handles[thread], INFINITE);
                CloseHandle(handles[thread]);
            }
            if (created == thread_count) {
                int status = NF_POSTCOLOR_OK;
                for (thread = 0U; thread < thread_count; ++thread) {
                    if (contexts[thread].status != NF_POSTCOLOR_OK) {
                        status = contexts[thread].status;
                        break;
                    }
                }
                free(contexts);
                free(handles);
                return status;
            }
        }
        free(contexts);
        free(handles);
    }
#endif
    return nf_postcolor_range(&serial);
}
