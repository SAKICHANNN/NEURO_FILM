#include "nf_rec2020_lab_source_compress_f32_v1.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

typedef struct nf_lab_range_context {
    const float *source_lab;
    const float *target_lab;
    float *output_lab;
    float *output_rgb;
    float *scale;
    uint64_t begin;
    uint64_t end;
    uint32_t iterations;
    double tolerance;
    int status;
} nf_lab_range_context;

static const double NF_D65_WHITE[3] = {0.95047, 1.0, 1.08883};
static const double NF_LAB_TO_REC2020[3][3] = {
    {1.716341645291047, -0.35552084503207726, -0.25330458233854103},
    {-0.66672834453056, 1.6165172268966244, 0.01577834986012067},
    {0.01764568230852517, -0.04282249947940328, 0.9424085120533846}};

static void nf_lab_to_rgb(const float *lab, float rgb[3]) {
    double fy = ((double)lab[0] + 16.0) / 116.0;
    double f[3];
    double relative[3];
    float xyz[3];
    uint32_t channel;
    f[0] = fy + (double)lab[1] / 500.0;
    f[1] = fy;
    f[2] = fy - (double)lab[2] / 200.0;
    for (channel = 0; channel < 3; ++channel) {
        relative[channel] = f[channel] > 0.2068966
            ? f[channel] * f[channel] * f[channel]
            : (f[channel] - 16.0 / 116.0) / 7.787;
        xyz[channel] = (float)(relative[channel] * NF_D65_WHITE[channel]);
    }
    for (channel = 0; channel < 3; ++channel) {
        rgb[channel] = (float)(NF_LAB_TO_REC2020[channel][0] * (double)xyz[0] +
                               NF_LAB_TO_REC2020[channel][1] * (double)xyz[1] +
                               NF_LAB_TO_REC2020[channel][2] * (double)xyz[2]);
    }
}

static int nf_rgb_in_gamut(const float rgb[3], double tolerance) {
    return isfinite(rgb[0]) && isfinite(rgb[1]) && isfinite(rgb[2]) &&
           (double)rgb[0] >= -tolerance && (double)rgb[0] <= 1.0 + tolerance &&
           (double)rgb[1] >= -tolerance && (double)rgb[1] <= 1.0 + tolerance &&
           (double)rgb[2] >= -tolerance && (double)rgb[2] <= 1.0 + tolerance;
}

static int nf_compress_range(nf_lab_range_context *context) {
    uint64_t index;
    float valid_scale = 0.0F;
    uint32_t iteration;
    for (iteration = 0; iteration < context->iterations; ++iteration) {
        valid_scale = (float)((valid_scale + 1.0F) * 0.5F);
    }
    for (index = context->begin; index < context->end; ++index) {
        const float *source = context->source_lab + index * 3U;
        const float *target = context->target_lab + index * 3U;
        float *result = context->output_lab + index * 3U;
        float *rgb = context->output_rgb + index * 3U;
        float target_rgb[3];
        float candidate[3];
        float candidate_rgb[3];
        float low = 0.0F;
        float high = 1.0F;
        nf_lab_to_rgb(target, target_rgb);
        if (nf_rgb_in_gamut(target_rgb, 0.0)) {
            result[0] = source[0] + (target[0] - source[0]) * valid_scale;
            result[1] = source[1] + (target[1] - source[1]) * valid_scale;
            result[2] = source[2] + (target[2] - source[2]) * valid_scale;
            nf_lab_to_rgb(result, rgb);
            context->scale[index] = valid_scale;
            continue;
        }
        for (iteration = 0; iteration < context->iterations; ++iteration) {
            float mid = (low + high) * 0.5F;
            candidate[0] = source[0] + (target[0] - source[0]) * mid;
            candidate[1] = source[1] + (target[1] - source[1]) * mid;
            candidate[2] = source[2] + (target[2] - source[2]) * mid;
            nf_lab_to_rgb(candidate, candidate_rgb);
            if (nf_rgb_in_gamut(candidate_rgb, 0.0)) {
                low = mid;
            } else {
                high = mid;
            }
        }
        result[0] = source[0] + (target[0] - source[0]) * low;
        result[1] = source[1] + (target[1] - source[1]) * low;
        result[2] = source[2] + (target[2] - source[2]) * low;
        nf_lab_to_rgb(result, rgb);
        if (!nf_rgb_in_gamut(rgb, context->tolerance)) {
            return NF_LAB_COMPRESS_OUTPUT_OUT_OF_GAMUT;
        }
        context->scale[index] = low;
    }
    return NF_LAB_COMPRESS_OK;
}

#if defined(_WIN32)
static DWORD WINAPI nf_thread_entry(LPVOID value) {
    nf_lab_range_context *context = (nf_lab_range_context *)value;
    context->status = nf_compress_range(context);
    return 0;
}
#endif

int nf_rec2020_lab_source_compress_f32_v1(
    const float *source_lab,
    const float *target_lab,
    float *output_lab,
    float *output_rgb,
    float *scale,
    uint64_t pixel_count,
    uint32_t iterations,
    double tolerance,
    uint32_t thread_count) {
    nf_lab_range_context serial;
    uint64_t index;
    if (source_lab == NULL || target_lab == NULL || output_lab == NULL ||
        output_rgb == NULL || scale == NULL || pixel_count == 0 || iterations == 0 ||
        iterations > 64 || !isfinite(tolerance) || tolerance < 0.0 ||
        thread_count == 0 || thread_count > 64 || source_lab == output_lab ||
        target_lab == output_lab) {
        return NF_LAB_COMPRESS_INVALID_ARGUMENT;
    }
    for (index = 0; index < pixel_count * 3U; ++index) {
        if (!isfinite(source_lab[index]) || !isfinite(target_lab[index])) {
            return NF_LAB_COMPRESS_NONFINITE_INPUT;
        }
    }
    for (index = 0; index < pixel_count; ++index) {
        float source_rgb[3];
        nf_lab_to_rgb(source_lab + index * 3U, source_rgb);
        if (!nf_rgb_in_gamut(source_rgb, tolerance)) {
            return NF_LAB_COMPRESS_SOURCE_OUT_OF_GAMUT;
        }
    }
    serial.source_lab = source_lab;
    serial.target_lab = target_lab;
    serial.output_lab = output_lab;
    serial.output_rgb = output_rgb;
    serial.scale = scale;
    serial.begin = 0;
    serial.end = pixel_count;
    serial.iterations = iterations;
    serial.tolerance = tolerance;
    serial.status = NF_LAB_COMPRESS_OK;
#if defined(_WIN32)
    if (thread_count > 1 && pixel_count >= (uint64_t)thread_count) {
        HANDLE *handles = (HANDLE *)calloc(thread_count, sizeof(HANDLE));
        nf_lab_range_context *contexts =
            (nf_lab_range_context *)calloc(thread_count, sizeof(nf_lab_range_context));
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
                int status = NF_LAB_COMPRESS_OK;
                for (thread = 0; thread < thread_count; ++thread) {
                    if (contexts[thread].status != NF_LAB_COMPRESS_OK) {
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
    return nf_compress_range(&serial);
}
