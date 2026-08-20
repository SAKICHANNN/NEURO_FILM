#define NF_SAFE_LAB_SPATIAL_TONE_BUILD
#include "nf_safe_lab_spatial_tone_f32_v1.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define NF_SAFE_LAB_MAX_RADIUS 32U
#define NF_SAFE_LAB_MAX_DIAMETER (2U * NF_SAFE_LAB_MAX_RADIUS + 1U)

typedef struct nf_spatial_context {
    const float *source_lab;
    const float *pointwise_lab;
    float *output_lab;
    float *source_workspace;
    float *target_workspace;
    uint64_t height;
    uint64_t width;
    uint64_t begin;
    uint64_t end;
    uint32_t radius;
    double weights[NF_SAFE_LAB_MAX_DIAMETER];
    float detail_strength;
    float tone_strength;
    float shadow_floor_l;
    float highlight_ceiling_l;
    int stage;
    int status;
} nf_spatial_context;

static float nf_clip(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

static uint64_t nf_reflect_index(int64_t value, uint64_t count) {
    while (value < 0 || (uint64_t)value >= count) {
        if (value < 0) {
            value = -value - 1;
        } else {
            value = (int64_t)(2U * count) - value - 1;
        }
    }
    return (uint64_t)value;
}

static int nf_ranges_overlap(
    const float *first, uint64_t first_count,
    const float *second, uint64_t second_count) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const uintptr_t first_end = first_start + first_count * sizeof(float);
    const uintptr_t second_end = second_start + second_count * sizeof(float);
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

static int nf_apply_vertical(nf_spatial_context *context) {
    uint64_t y;
    for (y = context->begin; y < context->end; ++y) {
        uint64_t x;
        for (x = 0; x < context->width; ++x) {
            double source_value = 0.0;
            double target_value = 0.0;
            uint32_t kernel;
            for (kernel = 0; kernel <= 2U * context->radius; ++kernel) {
                const int64_t offset =
                    (int64_t)kernel - (int64_t)context->radius;
                const uint64_t source_y = nf_reflect_index(
                    (int64_t)y + offset, context->height);
                const uint64_t pixel = source_y * context->width + x;
                source_value += context->weights[kernel] *
                    (double)context->source_lab[pixel * 3U];
                target_value += context->weights[kernel] *
                    (double)context->pointwise_lab[pixel * 3U];
            }
            context->source_workspace[y * context->width + x] =
                (float)source_value;
            context->target_workspace[y * context->width + x] =
                (float)target_value;
        }
    }
    return NF_SAFE_LAB_SPATIAL_TONE_OK;
}

static int nf_apply_horizontal(nf_spatial_context *context) {
    uint64_t y;
    for (y = context->begin; y < context->end; ++y) {
        uint64_t x;
        for (x = 0; x < context->width; ++x) {
            double source_base = 0.0;
            double target_base = 0.0;
            const uint64_t pixel = y * context->width + x;
            uint32_t kernel;
            float luminance;
            for (kernel = 0; kernel <= 2U * context->radius; ++kernel) {
                const int64_t offset =
                    (int64_t)kernel - (int64_t)context->radius;
                const uint64_t source_x = nf_reflect_index(
                    (int64_t)x + offset, context->width);
                const uint64_t source_pixel = y * context->width + source_x;
                source_base += context->weights[kernel] *
                    (double)context->source_workspace[source_pixel];
                target_base += context->weights[kernel] *
                    (double)context->target_workspace[source_pixel];
            }
            luminance = context->pointwise_lab[pixel * 3U] +
                context->detail_strength * (
                    context->source_lab[pixel * 3U] - (float)source_base -
                    context->pointwise_lab[pixel * 3U] + (float)target_base);
            luminance = nf_clip(luminance, 0.0F, 100.0F);
            if (context->tone_strength > 0.0F) {
                const float normalized = nf_clip(luminance / 100.0F, 0.0F, 1.0F);
                const float smooth = normalized * normalized *
                    (3.0F - 2.0F * normalized);
                const float low = nf_clip(
                    context->shadow_floor_l / 100.0F, 0.0F, 0.25F);
                const float high = nf_clip(
                    context->highlight_ceiling_l / 100.0F, 0.75F, 1.0F);
                const float rolled = low + smooth * (high - low);
                luminance = 100.0F * (
                    (1.0F - context->tone_strength) * normalized +
                    context->tone_strength * rolled);
            }
            context->output_lab[pixel * 3U] = luminance;
            context->output_lab[pixel * 3U + 1U] =
                context->pointwise_lab[pixel * 3U + 1U];
            context->output_lab[pixel * 3U + 2U] =
                context->pointwise_lab[pixel * 3U + 2U];
            if (!isfinite(context->output_lab[pixel * 3U]) ||
                !isfinite(context->output_lab[pixel * 3U + 1U]) ||
                !isfinite(context->output_lab[pixel * 3U + 2U])) {
                return NF_SAFE_LAB_SPATIAL_TONE_NONFINITE_OUTPUT;
            }
        }
    }
    return NF_SAFE_LAB_SPATIAL_TONE_OK;
}

static int nf_apply_stage(nf_spatial_context *context) {
    return context->stage == 0
        ? nf_apply_vertical(context)
        : nf_apply_horizontal(context);
}

#if defined(_WIN32)
static DWORD WINAPI nf_thread_entry(LPVOID value) {
    nf_spatial_context *context = (nf_spatial_context *)value;
    context->status = nf_apply_stage(context);
    return 0;
}
#endif

static int nf_run_stage(nf_spatial_context *serial, uint32_t thread_count) {
#if defined(_WIN32)
    if (thread_count > 1U && serial->height >= (uint64_t)thread_count) {
        HANDLE *handles = (HANDLE *)calloc(thread_count, sizeof(HANDLE));
        nf_spatial_context *contexts = (nf_spatial_context *)calloc(
            thread_count, sizeof(nf_spatial_context));
        uint32_t created = 0U;
        uint32_t thread;
        if (handles != NULL && contexts != NULL) {
            for (thread = 0U; thread < thread_count; ++thread) {
                contexts[thread] = *serial;
                contexts[thread].begin = serial->height * thread / thread_count;
                contexts[thread].end = serial->height * (thread + 1U) / thread_count;
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
                int status = NF_SAFE_LAB_SPATIAL_TONE_OK;
                for (thread = 0U; thread < thread_count; ++thread) {
                    if (contexts[thread].status != NF_SAFE_LAB_SPATIAL_TONE_OK) {
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
#else
    (void)thread_count;
#endif
    serial->begin = 0U;
    serial->end = serial->height;
    return nf_apply_stage(serial);
}

int nf_safe_lab_spatial_tone_f32_v1(
    const float *source_lab,
    const float *pointwise_lab,
    float *output_lab,
    float *source_workspace,
    float *target_workspace,
    uint64_t height,
    uint64_t width,
    float sigma,
    float truncate,
    float detail_strength,
    float tone_strength,
    float shadow_floor_l,
    float highlight_ceiling_l,
    uint32_t thread_count) {
    nf_spatial_context context;
    uint64_t pixels;
    uint64_t index;
    uint32_t kernel;
    double sum = 0.0;
    const double scaled_radius = (double)sigma * (double)truncate;
    if (source_lab == NULL || pointwise_lab == NULL || output_lab == NULL ||
        source_workspace == NULL || target_workspace == NULL || height == 0U ||
        width == 0U || height > UINT64_MAX / width || !isfinite(sigma) ||
        !isfinite(truncate) || !isfinite(detail_strength) ||
        !isfinite(tone_strength) || !isfinite(shadow_floor_l) ||
        !isfinite(highlight_ceiling_l) || sigma <= 0.0F || truncate <= 0.0F ||
        detail_strength < 0.0F || detail_strength > 1.0F ||
        tone_strength < 0.0F || tone_strength > 1.0F || thread_count == 0U ||
        thread_count > 64U || scaled_radius + 0.5 >= NF_SAFE_LAB_MAX_RADIUS + 1U) {
        return NF_SAFE_LAB_SPATIAL_TONE_INVALID_ARGUMENT;
    }
    pixels = height * width;
    if (pixels > UINT64_MAX / 3U ||
        nf_ranges_overlap(source_lab, pixels * 3U, pointwise_lab, pixels * 3U) ||
        nf_ranges_overlap(source_lab, pixels * 3U, output_lab, pixels * 3U) ||
        nf_ranges_overlap(pointwise_lab, pixels * 3U, output_lab, pixels * 3U) ||
        nf_ranges_overlap(source_workspace, pixels, target_workspace, pixels) ||
        nf_ranges_overlap(source_lab, pixels * 3U, source_workspace, pixels) ||
        nf_ranges_overlap(source_lab, pixels * 3U, target_workspace, pixels) ||
        nf_ranges_overlap(pointwise_lab, pixels * 3U, source_workspace, pixels) ||
        nf_ranges_overlap(pointwise_lab, pixels * 3U, target_workspace, pixels) ||
        nf_ranges_overlap(output_lab, pixels * 3U, source_workspace, pixels) ||
        nf_ranges_overlap(output_lab, pixels * 3U, target_workspace, pixels)) {
        return NF_SAFE_LAB_SPATIAL_TONE_INVALID_ARGUMENT;
    }
    for (index = 0U; index < pixels * 3U; ++index) {
        if (!isfinite(source_lab[index]) || !isfinite(pointwise_lab[index])) {
            return NF_SAFE_LAB_SPATIAL_TONE_NONFINITE_INPUT;
        }
    }
    context.source_lab = source_lab;
    context.pointwise_lab = pointwise_lab;
    context.output_lab = output_lab;
    context.source_workspace = source_workspace;
    context.target_workspace = target_workspace;
    context.height = height;
    context.width = width;
    context.radius = (uint32_t)(scaled_radius + 0.5);
    context.detail_strength = detail_strength;
    context.tone_strength = tone_strength;
    context.shadow_floor_l = shadow_floor_l;
    context.highlight_ceiling_l = highlight_ceiling_l;
    context.stage = 0;
    context.status = NF_SAFE_LAB_SPATIAL_TONE_OK;
    for (kernel = 0U; kernel <= 2U * context.radius; ++kernel) {
        const double offset = (double)((int64_t)kernel - (int64_t)context.radius);
        const double value = exp(-0.5 * (offset / sigma) * (offset / sigma));
        context.weights[kernel] = value;
        sum += value;
    }
    for (kernel = 0U; kernel <= 2U * context.radius; ++kernel) {
        context.weights[kernel] /= sum;
    }
    {
        const int status = nf_run_stage(&context, thread_count);
        if (status != NF_SAFE_LAB_SPATIAL_TONE_OK) {
            return status;
        }
    }
    context.stage = 1;
    return nf_run_stage(&context, thread_count);
}
