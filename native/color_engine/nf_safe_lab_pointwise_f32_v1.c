#include "nf_safe_lab_pointwise_f32_v1.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

typedef struct nf_safe_lab_context {
    const float *source_lab;
    float *output_lab;
    uint64_t begin;
    uint64_t end;
    float source_mean[3];
    float source_std[3];
    float destination_mean[3];
    float destination_std[3];
    float strength;
    float luma_strength;
    float chroma_curve_strength;
    float neutral_protect;
    float skin_protect;
    float max_chroma_gain;
    float max_chroma_boost;
    float max_chroma_absolute;
    int status;
} nf_safe_lab_context;

static float nf_clip(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

static float nf_smoothstep(float edge0, float edge1, float value) {
    float denominator = edge1 - edge0;
    float x;
    if (denominator < 1e-6F) {
        denominator = 1e-6F;
    }
    x = nf_clip((value - edge0) / denominator, 0.0F, 1.0F);
    return x * x * (3.0F - 2.0F * x);
}

static int nf_apply_range(nf_safe_lab_context *context) {
    uint64_t index;
    for (index = context->begin; index < context->end; ++index) {
        const float *source = context->source_lab + index * 3U;
        float *output = context->output_lab + index * 3U;
        float transferred[3];
        float source_chroma;
        float output_chroma;
        float saturated_weight;
        float chroma_scale;
        float neutral_weight;
        float neutral_blend;
        float skin_weight;
        float cap;
        float cap_scale;
        float knee;
        uint32_t channel;
        for (channel = 0; channel < 3; ++channel) {
            transferred[channel] =
                (source[channel] - context->source_mean[channel]) /
                    context->source_std[channel] * context->destination_std[channel] +
                context->destination_mean[channel];
        }
        output[0] = source[0] + context->luma_strength * context->strength *
                                      (transferred[0] - source[0]);
        output[1] = source[1] + context->strength * (transferred[1] - source[1]);
        output[2] = source[2] + context->strength * (transferred[2] - source[2]);

        source_chroma = hypotf(source[1], source[2]);
        saturated_weight = nf_smoothstep(24.0F, 60.0F, source_chroma);
        chroma_scale = 1.0F - context->chroma_curve_strength * saturated_weight;
        output[1] = source[1] + (output[1] - source[1]) * chroma_scale;
        output[2] = source[2] + (output[2] - source[2]) * chroma_scale;

        neutral_weight = 1.0F - nf_smoothstep(4.0F, 14.0F, source_chroma);
        neutral_blend = nf_clip(
            neutral_weight * context->neutral_protect, 0.0F, 1.0F);
        output[1] = output[1] * (1.0F - neutral_blend) + source[1] * neutral_blend;
        output[2] = output[2] * (1.0F - neutral_blend) + source[2] * neutral_blend;

        skin_weight = source[0] > 20.0F && source[0] < 92.0F &&
                              source[1] > 4.0F && source[1] < 28.0F &&
                              source[2] > 4.0F && source[2] < 46.0F
                          ? nf_clip(context->skin_protect, 0.0F, 1.0F)
                          : 0.0F;
        output[1] = output[1] * (1.0F - skin_weight) + source[1] * skin_weight;
        output[2] = output[2] * (1.0F - skin_weight) + source[2] * skin_weight;

        cap = fmaxf(
            source_chroma * context->max_chroma_gain,
            source_chroma + context->max_chroma_boost);
        if (context->max_chroma_absolute >= 0.0F) {
            cap = fminf(cap, context->max_chroma_absolute);
        }
        output_chroma = fmaxf(hypotf(output[1], output[2]), 1e-6F);
        cap_scale = fminf(1.0F, cap / output_chroma);
        knee = nf_smoothstep(0.82F, 1.0F, output_chroma / fmaxf(cap, 1e-6F));
        output[1] *= (1.0F - knee) + knee * cap_scale;
        output[2] *= (1.0F - knee) + knee * cap_scale;
        if (!isfinite(output[0]) || !isfinite(output[1]) || !isfinite(output[2])) {
            return NF_SAFE_LAB_NONFINITE_OUTPUT;
        }
    }
    return NF_SAFE_LAB_OK;
}

#if defined(_WIN32)
static DWORD WINAPI nf_thread_entry(LPVOID value) {
    nf_safe_lab_context *context = (nf_safe_lab_context *)value;
    context->status = nf_apply_range(context);
    return 0;
}
#endif

int nf_safe_lab_pointwise_f32_v1(
    const float *source_lab,
    float *output_lab,
    uint64_t pixel_count,
    const float source_mean[3],
    const float source_std[3],
    const float destination_mean[3],
    const float destination_std[3],
    float strength,
    float luma_strength,
    float chroma_curve_strength,
    float neutral_protect,
    float skin_protect,
    float max_chroma_gain,
    float max_chroma_boost,
    float max_chroma_absolute,
    uint32_t thread_count) {
    nf_safe_lab_context serial;
    uint64_t index;
    uint32_t channel;
    if (source_lab == NULL || output_lab == NULL || source_mean == NULL ||
        source_std == NULL || destination_mean == NULL || destination_std == NULL ||
        source_lab == output_lab || pixel_count == 0 || thread_count == 0 ||
        thread_count > 64 || !isfinite(strength) || !isfinite(luma_strength) ||
        !isfinite(chroma_curve_strength) || !isfinite(neutral_protect) ||
        !isfinite(skin_protect) || !isfinite(max_chroma_gain) ||
        !isfinite(max_chroma_boost) || !isfinite(max_chroma_absolute) ||
        strength < 0.0F || strength > 1.0F || luma_strength < 0.0F ||
        luma_strength > 1.0F || chroma_curve_strength < 0.0F ||
        chroma_curve_strength > 1.0F || neutral_protect < 0.0F ||
        skin_protect < 0.0F || max_chroma_gain <= 0.0F ||
        max_chroma_boost < 0.0F) {
        return NF_SAFE_LAB_INVALID_ARGUMENT;
    }
    for (channel = 0; channel < 3; ++channel) {
        if (!isfinite(source_mean[channel]) || !isfinite(source_std[channel]) ||
            !isfinite(destination_mean[channel]) ||
            !isfinite(destination_std[channel]) || source_std[channel] < 1e-3F ||
            destination_std[channel] <= 0.0F) {
            return NF_SAFE_LAB_INVALID_ARGUMENT;
        }
    }
    for (index = 0; index < pixel_count * 3U; ++index) {
        if (!isfinite(source_lab[index])) {
            return NF_SAFE_LAB_NONFINITE_INPUT;
        }
    }
    serial.source_lab = source_lab;
    serial.output_lab = output_lab;
    serial.begin = 0;
    serial.end = pixel_count;
    for (channel = 0; channel < 3; ++channel) {
        serial.source_mean[channel] = source_mean[channel];
        serial.source_std[channel] = source_std[channel];
        serial.destination_mean[channel] = destination_mean[channel];
        serial.destination_std[channel] = destination_std[channel];
    }
    serial.strength = strength;
    serial.luma_strength = luma_strength;
    serial.chroma_curve_strength = chroma_curve_strength;
    serial.neutral_protect = neutral_protect;
    serial.skin_protect = skin_protect;
    serial.max_chroma_gain = max_chroma_gain;
    serial.max_chroma_boost = max_chroma_boost;
    serial.max_chroma_absolute = max_chroma_absolute;
    serial.status = NF_SAFE_LAB_OK;
#if defined(_WIN32)
    if (thread_count > 1 && pixel_count >= (uint64_t)thread_count) {
        HANDLE *handles = (HANDLE *)calloc(thread_count, sizeof(HANDLE));
        nf_safe_lab_context *contexts =
            (nf_safe_lab_context *)calloc(thread_count, sizeof(nf_safe_lab_context));
        uint32_t created = 0;
        uint32_t thread;
        if (handles != NULL && contexts != NULL) {
            for (thread = 0; thread < thread_count; ++thread) {
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
            for (thread = 0; thread < created; ++thread) {
                (void)WaitForSingleObject(handles[thread], INFINITE);
                CloseHandle(handles[thread]);
            }
            if (created == thread_count) {
                int status = NF_SAFE_LAB_OK;
                for (thread = 0; thread < thread_count; ++thread) {
                    if (contexts[thread].status != NF_SAFE_LAB_OK) {
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
    return nf_apply_range(&serial);
}
