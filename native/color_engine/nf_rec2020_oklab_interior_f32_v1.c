#include "nf_rec2020_oklab_interior_f32_v1.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

typedef struct nf_range_context {
    const float *input_rgb;
    float *output_rgb;
    float *chroma_scale;
    uint64_t begin;
    uint64_t end;
    double softness;
    double margin;
    uint32_t iterations;
    int status;
} nf_range_context;

static const double NF_REC2020_TO_XYZ[3][3] = {
    {0.63695804830129132, 0.14461690358620838, 0.16888097516417205},
    {0.26270021201126703, 0.67799807151887104, 0.059301716469861945},
    {0.0, 0.028072693049087508, 1.0609850577107909}};
static const double NF_XYZ_TO_LMS[3][3] = {
    {0.8190224379967030, 0.3619062600528904, -0.1288737815209879},
    {0.0329836539323885, 0.9292868615863434, 0.0361446663506424},
    {0.0481771893596242, 0.2642395317527308, 0.6335478284694309}};
static const double NF_LMS_TO_OKLAB[3][3] = {
    {0.2104542683093140, 0.7936177747023056, -0.0040720430116193},
    {1.9779985324311684, -2.4285922420485799, 0.4505937096174110},
    {0.0259040424655478, 0.7827717124575296, -0.8086757549230774}};
static const double NF_OKLAB_TO_LMS[3][3] = {
    {1.0, 0.3963377773761749, 0.2158037573099136},
    {1.0, -0.1055613458156586, -0.0638541728258133},
    {1.0, -0.0894841775298119, -1.2914855480194092}};
static const double NF_LMS_TO_XYZ[3][3] = {
    {1.2268798758459243, -0.5578149944602171, 0.2813910456659647},
    {-0.0405757452148008, 1.1122868032803170, -0.0717110580655164},
    {-0.0763729366746601, -0.4214933324022432, 1.5869240198367816}};
static const double NF_XYZ_TO_REC2020[3][3] = {
    {1.7166511879712676, -0.35567078377639239, -0.2533662813736598},
    {-0.66668435183248898, 1.616481236634939, 0.015768545813911131},
    {0.017639857445310915, -0.042770613257808655, 0.94210312123547402}};

static void nf_mat3(const double matrix[3][3], const double value[3], double output[3]) {
    uint32_t row;
    for (row = 0; row < 3; ++row) {
        output[row] = matrix[row][0] * value[0] + matrix[row][1] * value[1] +
                      matrix[row][2] * value[2];
    }
}

static void nf_linear_rec2020_to_oklab(const float *rgb, double oklab[3]) {
    double input[3] = {(double)rgb[0], (double)rgb[1], (double)rgb[2]};
    double xyz[3];
    double lms[3];
    double roots[3];
    nf_mat3(NF_REC2020_TO_XYZ, input, xyz);
    nf_mat3(NF_XYZ_TO_LMS, xyz, lms);
    roots[0] = cbrt(lms[0]);
    roots[1] = cbrt(lms[1]);
    roots[2] = cbrt(lms[2]);
    nf_mat3(NF_LMS_TO_OKLAB, roots, oklab);
}

static void nf_oklab_to_linear_rec2020(const double oklab[3], double rgb[3]) {
    double roots[3];
    double lms[3];
    double xyz[3];
    nf_mat3(NF_OKLAB_TO_LMS, oklab, roots);
    lms[0] = roots[0] * roots[0] * roots[0];
    lms[1] = roots[1] * roots[1] * roots[1];
    lms[2] = roots[2] * roots[2] * roots[2];
    nf_mat3(NF_LMS_TO_XYZ, lms, xyz);
    nf_mat3(NF_XYZ_TO_REC2020, xyz, rgb);
}

static double nf_logaddexp_zero(double value) {
    double maximum = value > 0.0 ? value : 0.0;
    return maximum + log1p(exp(-fabs(value)));
}

static int nf_inside(const double rgb[3], double margin) {
    return isfinite(rgb[0]) && isfinite(rgb[1]) && isfinite(rgb[2]) &&
           rgb[0] >= margin && rgb[0] <= 1.0 - margin &&
           rgb[1] >= margin && rgb[1] <= 1.0 - margin &&
           rgb[2] >= margin && rgb[2] <= 1.0 - margin;
}

static int nf_map_range(nf_range_context *context) {
    uint64_t index;
    double neutral_minimum = cbrt(context->margin);
    double neutral_maximum = cbrt(1.0 - context->margin);
    for (index = context->begin; index < context->end; ++index) {
        const float *source = context->input_rgb + index * 3U;
        float *target = context->output_rgb + index * 3U;
        double oklab[3];
        double mapped[3];
        double rgb[3];
        double chroma;
        double direction_a = 0.0;
        double direction_b = 0.0;
        double low = 0.0;
        double high = 1.0;
        double normalized_lightness;
        double mapped_lightness;
        uint32_t iteration;
        if (source[0] >= 0.0F && source[0] <= 1.0F && source[1] >= 0.0F &&
            source[1] <= 1.0F && source[2] >= 0.0F && source[2] <= 1.0F) {
            target[0] = source[0];
            target[1] = source[1];
            target[2] = source[2];
            context->chroma_scale[index] = 1.0F;
            continue;
        }
        nf_linear_rec2020_to_oklab(source, oklab);
        chroma = hypot(oklab[1], oklab[2]);
        if (chroma > 4e-12) {
            direction_a = oklab[1] / chroma;
            direction_b = oklab[2] / chroma;
        }
        normalized_lightness = context->softness *
            (nf_logaddexp_zero(oklab[0] / context->softness) -
             nf_logaddexp_zero((oklab[0] - 1.0) / context->softness));
        mapped_lightness = neutral_minimum +
            (neutral_maximum - neutral_minimum) * normalized_lightness;
        mapped[0] = mapped_lightness;
        mapped[1] = 0.0;
        mapped[2] = 0.0;
        nf_oklab_to_linear_rec2020(mapped, rgb);
        if (!nf_inside(rgb, context->margin)) {
            return NF_COLOR_MAPPING_FAILED;
        }
        for (iteration = 0; iteration < context->iterations; ++iteration) {
            double current = (low + high) * 0.5;
            mapped[1] = direction_a * chroma * current;
            mapped[2] = direction_b * chroma * current;
            nf_oklab_to_linear_rec2020(mapped, rgb);
            if (nf_inside(rgb, context->margin)) {
                low = current;
            } else {
                high = current;
            }
        }
        mapped[1] = direction_a * chroma * low;
        mapped[2] = direction_b * chroma * low;
        nf_oklab_to_linear_rec2020(mapped, rgb);
        if (!nf_inside(rgb, context->margin)) {
            return NF_COLOR_MAPPING_FAILED;
        }
        target[0] = (float)rgb[0];
        target[1] = (float)rgb[1];
        target[2] = (float)rgb[2];
        context->chroma_scale[index] = (float)low;
    }
    return NF_COLOR_OK;
}

#if defined(_WIN32)
static DWORD WINAPI nf_thread_entry(LPVOID value) {
    nf_range_context *context = (nf_range_context *)value;
    context->status = nf_map_range(context);
    return 0;
}
#endif

int nf_rec2020_oklab_interior_f32_v1(
    const float *input_rgb,
    float *output_rgb,
    float *chroma_scale,
    uint64_t pixel_count,
    double softness,
    double margin,
    uint32_t iterations,
    uint32_t thread_count) {
    uint64_t index;
    nf_range_context serial;
    if (input_rgb == NULL || output_rgb == NULL || chroma_scale == NULL ||
        input_rgb == output_rgb || pixel_count == 0 || !isfinite(softness) ||
        softness <= 0.0 || !isfinite(margin) || margin <= 0.0 || margin >= 0.5 ||
        iterations == 0 || thread_count == 0 || thread_count > 64) {
        return NF_COLOR_INVALID_ARGUMENT;
    }
    for (index = 0; index < pixel_count * 3U; ++index) {
        if (!isfinite(input_rgb[index])) {
            return NF_COLOR_NONFINITE_INPUT;
        }
    }
    serial.input_rgb = input_rgb;
    serial.output_rgb = output_rgb;
    serial.chroma_scale = chroma_scale;
    serial.begin = 0;
    serial.end = pixel_count;
    serial.softness = softness;
    serial.margin = margin;
    serial.iterations = iterations;
    serial.status = NF_COLOR_OK;
#if defined(_WIN32)
    if (thread_count > 1 && pixel_count >= (uint64_t)thread_count) {
        HANDLE *handles = (HANDLE *)calloc(thread_count, sizeof(HANDLE));
        nf_range_context *contexts =
            (nf_range_context *)calloc(thread_count, sizeof(nf_range_context));
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
                int status = NF_COLOR_OK;
                for (thread = 0; thread < thread_count; ++thread) {
                    if (contexts[thread].status != NF_COLOR_OK) {
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
        /* A partial launch is overwritten by one complete deterministic pass. */
    }
#endif
    return nf_map_range(&serial);
}
