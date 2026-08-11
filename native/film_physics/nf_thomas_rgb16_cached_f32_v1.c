#define NF_THOMAS_RGB16_CACHED_F32_BUILD
#define NF_GRANULARITY_AMPLITUDE_F32_BUILD
#define NF_NEUTRAL_GAUGE_F32_BUILD
#define NF_THOMAS_ROWS_F32_BUILD
#define NF_THOMAS_FIELD_F32_BUILD
#define NF_NEUMAIER_F32_BUILD
#include "nf_thomas_rgb16_cached_f32_v1.h"
#include "nf_neumaier_f32_v1.h"
#include "reference_srgb_oetf_quantize_v1.h"

#include <math.h>
#include <stdint.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#endif

typedef struct nf_field_job_v1 {
    const nf_thomas_field_f32_profile_v1* profile;
    size_t full_height;
    size_t width;
    size_t row_partition;
    float* workspace;
    size_t workspace_floats;
    float* field;
    double mean;
    int ok;
} nf_field_job_v1;

typedef struct nf_develop_job_v1 {
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile;
    uint32_t channel;
    size_t full_count;
    size_t width;
    size_t row_partition;
    const float* exposure;
    float* field;
    double mean;
    float* density;
    float* sigma;
    int ok;
} nf_develop_job_v1;

typedef struct nf_output_job_v1 {
    const nf_neutral_gauge_f32_profile_v1* gauge_profile;
    const float* fields;
    size_t full_count;
    size_t field_offset;
    size_t tile_start;
    size_t count;
    float* interleaved;
    uint16_t* quantized;
    int ok;
} nf_output_job_v1;

static int nf_ranges_overlap_bytes(
    const void* first,
    size_t first_bytes,
    const void* second,
    size_t second_bytes) {
    const uintptr_t first_start = (uintptr_t)first;
    const uintptr_t second_start = (uintptr_t)second;
    const uintptr_t first_end = first_start + first_bytes;
    const uintptr_t second_end = second_start + second_bytes;
    if (first_end < first_start || second_end < second_start) {
        return 1;
    }
    return first_start < second_end && second_start < first_end;
}

static void nf_generate_field(nf_field_job_v1* job) {
    double total = 0.0;
    double compensation = 0.0;
    size_t row_start;
    const size_t full_count = job->full_height * job->width;
    job->ok = 0;
    for (row_start = 0u; row_start < job->full_height;
         row_start += job->row_partition) {
        const size_t row_count =
            job->row_partition < job->full_height - row_start ?
            job->row_partition : job->full_height - row_start;
        const size_t count = row_count * job->width;
        float* output = job->field + row_start * job->width;
        if (nf_thomas_rows_f32_field_v1(
                job->profile,
                job->full_height,
                job->width,
                row_start,
                row_count,
                job->workspace,
                job->workspace_floats,
                output,
                count) != NF_THOMAS_ROWS_F32_OK_V1 ||
            nf_neumaier_f32_accumulate_v1(
                output, count, &total, &compensation) != 1) {
            return;
        }
    }
    job->mean = (total + compensation) / (double)full_count;
    job->ok = isfinite(job->mean);
}

static void nf_develop_field(nf_develop_job_v1* job) {
    size_t offset;
    const size_t tile_count = job->row_partition * job->width;
    job->ok = 0;
    for (offset = 0u; offset < job->full_count; offset += tile_count) {
        const size_t count =
            tile_count < job->full_count - offset ?
            tile_count : job->full_count - offset;
        size_t index;
        if (nf_granularity_amplitude_f32_apply_layer_v1(
                job->amplitude_profile,
                job->channel,
                job->exposure + offset,
                count,
                job->density,
                job->sigma) != NF_GRANULARITY_AMPLITUDE_F32_OK_V1) {
            return;
        }
        for (index = 0u; index < count; ++index) {
            const float centered = (float)(
                (double)job->field[offset + index] - job->mean);
            const double density = (double)job->density[index] +
                (double)job->sigma[index] * centered;
            const double transmittance = pow(10.0, -density);
            if (!isfinite(density) || density < 0.0 ||
                !isfinite(transmittance) || transmittance <= 0.0 ||
                transmittance > 1.0) {
                return;
            }
            job->field[offset + index] = (float)transmittance;
        }
    }
    job->ok = 1;
}

static void nf_finish_output(nf_output_job_v1* job) {
    size_t index;
    size_t channel;
    job->ok = 0;
    for (index = 0u; index < job->count; ++index) {
        const size_t tile_index = job->tile_start + index;
        for (channel = 0u; channel < 3u; ++channel) {
            job->interleaved[3u * tile_index + channel] =
                job->fields[channel * job->full_count + job->field_offset +
                    tile_index];
        }
    }
    if (nf_neutral_gauge_f32_apply_v1(
            job->gauge_profile,
            job->interleaved + 3u * job->tile_start,
            job->count,
            job->interleaved + 3u * job->tile_start) !=
            NF_NEUTRAL_GAUGE_F32_OK_V1 ||
        nf_srgb_oetf_quantize_apply_v1(
            job->interleaved + 3u * job->tile_start,
            3u * job->count,
            16u,
            job->quantized + 3u * job->tile_start,
            3u * job->count) != 1) {
        return;
    }
    job->ok = 1;
}

#if defined(_WIN32)
static DWORD WINAPI nf_field_thread(LPVOID context) {
    nf_generate_field((nf_field_job_v1*)context);
    return 0u;
}

static DWORD WINAPI nf_develop_thread(LPVOID context) {
    nf_develop_field((nf_develop_job_v1*)context);
    return 0u;
}

static DWORD WINAPI nf_output_thread(LPVOID context) {
    nf_finish_output((nf_output_job_v1*)context);
    return 0u;
}
#else
static void* nf_field_thread(void* context) {
    nf_generate_field((nf_field_job_v1*)context);
    return NULL;
}

static void* nf_develop_thread(void* context) {
    nf_develop_field((nf_develop_job_v1*)context);
    return NULL;
}

static void* nf_output_thread(void* context) {
    nf_finish_output((nf_output_job_v1*)context);
    return NULL;
}
#endif

static int nf_run_field_jobs(nf_field_job_v1 jobs[3], uint32_t parallel_layers) {
    size_t channel;
    if (parallel_layers == 1u) {
        for (channel = 0u; channel < 3u; ++channel) {
            nf_generate_field(&jobs[channel]);
            if (!jobs[channel].ok) {
                return 0;
            }
        }
        return 1;
    }
#if defined(_WIN32)
    {
        HANDLE threads[3] = {NULL, NULL, NULL};
        for (channel = 0u; channel < 3u; ++channel) {
            threads[channel] = CreateThread(
                NULL, 0u, nf_field_thread, &jobs[channel], 0u, NULL);
            if (threads[channel] == NULL) {
                size_t prior;
                for (prior = 0u; prior < channel; ++prior) {
                    WaitForSingleObject(threads[prior], INFINITE);
                    CloseHandle(threads[prior]);
                }
                return 0;
            }
        }
        WaitForMultipleObjects(3u, threads, TRUE, INFINITE);
        for (channel = 0u; channel < 3u; ++channel) {
            CloseHandle(threads[channel]);
            if (!jobs[channel].ok) {
                return 0;
            }
        }
    }
#else
    {
        pthread_t threads[3];
        for (channel = 0u; channel < 3u; ++channel) {
            if (pthread_create(
                    &threads[channel], NULL, nf_field_thread, &jobs[channel]) != 0) {
                size_t prior;
                for (prior = 0u; prior < channel; ++prior) {
                    pthread_join(threads[prior], NULL);
                }
                return 0;
            }
        }
        for (channel = 0u; channel < 3u; ++channel) {
            if (pthread_join(threads[channel], NULL) != 0 ||
                !jobs[channel].ok) {
                return 0;
            }
        }
    }
#endif
    return 1;
}

static int nf_run_develop_jobs(
    nf_develop_job_v1 jobs[3], uint32_t parallel_layers) {
    size_t channel;
    if (parallel_layers == 1u) {
        for (channel = 0u; channel < 3u; ++channel) {
            nf_develop_field(&jobs[channel]);
            if (!jobs[channel].ok) {
                return 0;
            }
        }
        return 1;
    }
#if defined(_WIN32)
    {
        HANDLE threads[3] = {NULL, NULL, NULL};
        for (channel = 0u; channel < 3u; ++channel) {
            threads[channel] = CreateThread(
                NULL, 0u, nf_develop_thread, &jobs[channel], 0u, NULL);
            if (threads[channel] == NULL) {
                size_t prior;
                for (prior = 0u; prior < channel; ++prior) {
                    WaitForSingleObject(threads[prior], INFINITE);
                    CloseHandle(threads[prior]);
                }
                return 0;
            }
        }
        WaitForMultipleObjects(3u, threads, TRUE, INFINITE);
        for (channel = 0u; channel < 3u; ++channel) {
            CloseHandle(threads[channel]);
            if (!jobs[channel].ok) {
                return 0;
            }
        }
    }
#else
    {
        pthread_t threads[3];
        for (channel = 0u; channel < 3u; ++channel) {
            if (pthread_create(
                    &threads[channel], NULL, nf_develop_thread, &jobs[channel]) !=
                0) {
                size_t prior;
                for (prior = 0u; prior < channel; ++prior) {
                    pthread_join(threads[prior], NULL);
                }
                return 0;
            }
        }
        for (channel = 0u; channel < 3u; ++channel) {
            if (pthread_join(threads[channel], NULL) != 0 ||
                !jobs[channel].ok) {
                return 0;
            }
        }
    }
#endif
    return 1;
}

static int nf_run_output_jobs(
    nf_output_job_v1 jobs[3], uint32_t output_workers) {
    size_t worker;
    if (output_workers == 1u) {
        nf_finish_output(&jobs[0]);
        return jobs[0].ok;
    }
#if defined(_WIN32)
    {
        HANDLE threads[3] = {NULL, NULL, NULL};
        for (worker = 0u; worker < 3u; ++worker) {
            threads[worker] = CreateThread(
                NULL, 0u, nf_output_thread, &jobs[worker], 0u, NULL);
            if (threads[worker] == NULL) {
                size_t prior;
                for (prior = 0u; prior < worker; ++prior) {
                    WaitForSingleObject(threads[prior], INFINITE);
                    CloseHandle(threads[prior]);
                }
                return 0;
            }
        }
        WaitForMultipleObjects(3u, threads, TRUE, INFINITE);
        for (worker = 0u; worker < 3u; ++worker) {
            CloseHandle(threads[worker]);
            if (!jobs[worker].ok) {
                return 0;
            }
        }
    }
#else
    {
        pthread_t threads[3];
        for (worker = 0u; worker < 3u; ++worker) {
            if (pthread_create(
                    &threads[worker], NULL, nf_output_thread, &jobs[worker]) != 0) {
                size_t prior;
                for (prior = 0u; prior < worker; ++prior) {
                    pthread_join(threads[prior], NULL);
                }
                return 0;
            }
        }
        for (worker = 0u; worker < 3u; ++worker) {
            if (pthread_join(threads[worker], NULL) != 0 || !jobs[worker].ok) {
                return 0;
            }
        }
    }
#endif
    return 1;
}

uint32_t nf_thomas_rgb16_cached_f32_abi_version_v1(void) {
    return NF_THOMAS_RGB16_CACHED_F32_ABI_VERSION_V1;
}

nf_thomas_rgb16_cached_f32_status_v1
nf_thomas_rgb16_cached_f32_workspace_bytes_v1(
    size_t full_height,
    size_t width,
    size_t row_partition,
    uint32_t parallel_layers,
    size_t* workspace_bytes) {
    size_t full_count;
    size_t tile_count;
    size_t row_workspace;
    size_t float_count;
    size_t quantized_bytes;
    if (workspace_bytes == NULL || full_height == 0u || width == 0u ||
        row_partition == 0u || row_partition > full_height ||
        (parallel_layers != 1u && parallel_layers != 3u) ||
        full_height > SIZE_MAX / width || row_partition > SIZE_MAX / width ||
        nf_thomas_rows_f32_workspace_floats_v1(
            width, row_partition, &row_workspace) != NF_THOMAS_ROWS_F32_OK_V1) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    full_count = full_height * width;
    tile_count = row_partition * width;
    if (full_count > SIZE_MAX / 3u ||
        parallel_layers > (SIZE_MAX - 3u * full_count) / row_workspace) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    float_count = 3u * full_count + parallel_layers * row_workspace;
    if (tile_count > (SIZE_MAX - float_count) / (2u * parallel_layers + 3u)) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    float_count += (2u * parallel_layers + 3u) * tile_count;
    if (float_count > SIZE_MAX / sizeof(float) ||
        tile_count > SIZE_MAX / (3u * sizeof(uint16_t))) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    quantized_bytes = 3u * tile_count * sizeof(uint16_t);
    if (float_count * sizeof(float) > SIZE_MAX - quantized_bytes) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_bytes = float_count * sizeof(float) + quantized_bytes;
    return NF_THOMAS_RGB16_CACHED_F32_OK_V1;
}

static nf_thomas_rgb16_cached_f32_status_v1
nf_thomas_rgb16_cached_f32_apply_internal_v1(
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile,
    const nf_thomas_field_f32_profile_v1 field_profiles[3],
    const nf_neutral_gauge_f32_profile_v1* gauge_profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    uint32_t parallel_layers,
    uint32_t output_workers,
    const float* relative_log_exposure_chw,
    size_t exposure_floats,
    void* workspace,
    size_t workspace_bytes,
    nf_thomas_rgb16_f32_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]) {
    size_t required_bytes;
    size_t row_workspace;
    size_t full_count;
    size_t tile_count;
    size_t channel;
    size_t index;
    size_t row_start;
    float* cursor;
    float* fields;
    float* row_workspaces;
    float* density;
    float* sigma;
    float* interleaved;
    uint16_t* quantized;
    nf_field_job_v1 field_jobs[3];
    nf_develop_job_v1 develop_jobs[3];
    nf_output_job_v1 output_jobs[3];
    if (nf_granularity_amplitude_f32_validate_profile_v1(amplitude_profile) !=
            NF_GRANULARITY_AMPLITUDE_F32_OK_V1 || field_profiles == NULL ||
        nf_neutral_gauge_f32_validate_profile_v1(gauge_profile) !=
            NF_NEUTRAL_GAUGE_F32_OK_V1) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_PROFILE_V1;
    }
    for (channel = 0u; channel < 3u; ++channel) {
        if (nf_thomas_field_f32_validate_profile_v1(&field_profiles[channel]) !=
            NF_THOMAS_FIELD_F32_OK_V1) {
            return NF_THOMAS_RGB16_CACHED_F32_INVALID_PROFILE_V1;
        }
    }
    if (relative_log_exposure_chw == NULL || workspace == NULL || sink == NULL ||
        raw_field_means == NULL || (uintptr_t)workspace % _Alignof(float) != 0u ||
        (output_workers != 1u && output_workers != 3u) ||
        nf_thomas_rgb16_cached_f32_workspace_bytes_v1(
            full_height, width, row_partition, parallel_layers,
            &required_bytes) != NF_THOMAS_RGB16_CACHED_F32_OK_V1 ||
        nf_thomas_rows_f32_workspace_floats_v1(
            width, row_partition, &row_workspace) != NF_THOMAS_ROWS_F32_OK_V1 ||
        workspace_bytes < required_bytes) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    full_count = full_height * width;
    tile_count = row_partition * width;
    if (exposure_floats < 3u * full_count ||
        nf_ranges_overlap_bytes(
            relative_log_exposure_chw,
            3u * full_count * sizeof(float),
            workspace,
            required_bytes)) {
        return NF_THOMAS_RGB16_CACHED_F32_INVALID_ARGUMENT_V1;
    }
    for (channel = 0u; channel < 3u; ++channel) {
        const uint32_t knot_count = amplitude_profile->knot_count[channel];
        const double lower = amplitude_profile->log_exposure_knots[channel][0];
        const double upper =
            amplitude_profile->log_exposure_knots[channel][knot_count - 1u];
        for (index = 0u; index < full_count; ++index) {
            const double value = (double)relative_log_exposure_chw[
                channel * full_count + index];
            if (!isfinite(value) || value < lower || value > upper) {
                return NF_THOMAS_RGB16_CACHED_F32_DOMAIN_ERROR_V1;
            }
        }
    }
    cursor = (float*)workspace;
    fields = cursor;
    cursor += 3u * full_count;
    row_workspaces = cursor;
    cursor += parallel_layers * row_workspace;
    density = cursor;
    cursor += parallel_layers * tile_count;
    sigma = cursor;
    cursor += parallel_layers * tile_count;
    interleaved = cursor;
    cursor += 3u * tile_count;
    quantized = (uint16_t*)cursor;
    for (channel = 0u; channel < 3u; ++channel) {
        const size_t slot = parallel_layers == 1u ? 0u : channel;
        field_jobs[channel].profile = &field_profiles[channel];
        field_jobs[channel].full_height = full_height;
        field_jobs[channel].width = width;
        field_jobs[channel].row_partition = row_partition;
        field_jobs[channel].workspace = row_workspaces + slot * row_workspace;
        field_jobs[channel].workspace_floats = row_workspace;
        field_jobs[channel].field = fields + channel * full_count;
        field_jobs[channel].mean = -13.0;
        field_jobs[channel].ok = 0;
    }
    if (!nf_run_field_jobs(field_jobs, parallel_layers)) {
        return NF_THOMAS_RGB16_CACHED_F32_EXECUTION_FAILED_V1;
    }
    for (channel = 0u; channel < 3u; ++channel) {
        const size_t slot = parallel_layers == 1u ? 0u : channel;
        develop_jobs[channel].amplitude_profile = amplitude_profile;
        develop_jobs[channel].channel = (uint32_t)channel;
        develop_jobs[channel].full_count = full_count;
        develop_jobs[channel].width = width;
        develop_jobs[channel].row_partition = row_partition;
        develop_jobs[channel].exposure =
            relative_log_exposure_chw + channel * full_count;
        develop_jobs[channel].field = fields + channel * full_count;
        develop_jobs[channel].mean = field_jobs[channel].mean;
        develop_jobs[channel].density = density + slot * tile_count;
        develop_jobs[channel].sigma = sigma + slot * tile_count;
        develop_jobs[channel].ok = 0;
    }
    if (!nf_run_develop_jobs(develop_jobs, parallel_layers)) {
        return NF_THOMAS_RGB16_CACHED_F32_DOMAIN_ERROR_V1;
    }
    for (row_start = 0u; row_start < full_height; row_start += row_partition) {
        const size_t row_count =
            row_partition < full_height - row_start ?
            row_partition : full_height - row_start;
        const size_t count = row_count * width;
        const size_t offset = row_start * width;
        const size_t base_count = count / output_workers;
        const size_t remainder = count % output_workers;
        size_t tile_start = 0u;
        size_t worker;
        for (worker = 0u; worker < output_workers; ++worker) {
            const size_t worker_count =
                base_count + (worker < remainder ? 1u : 0u);
            output_jobs[worker].gauge_profile = gauge_profile;
            output_jobs[worker].fields = fields;
            output_jobs[worker].full_count = full_count;
            output_jobs[worker].field_offset = offset;
            output_jobs[worker].tile_start = tile_start;
            output_jobs[worker].count = worker_count;
            output_jobs[worker].interleaved = interleaved;
            output_jobs[worker].quantized = quantized;
            output_jobs[worker].ok = 0;
            tile_start += worker_count;
        }
        if (!nf_run_output_jobs(output_jobs, output_workers)) {
            return NF_THOMAS_RGB16_CACHED_F32_DOMAIN_ERROR_V1;
        }
        if (sink(
                sink_context, row_start, row_count, quantized, 3u * count) == 0) {
            return NF_THOMAS_RGB16_CACHED_F32_CALLBACK_FAILED_V1;
        }
    }
    for (channel = 0u; channel < 3u; ++channel) {
        raw_field_means[channel] = field_jobs[channel].mean;
    }
    return NF_THOMAS_RGB16_CACHED_F32_OK_V1;
}

nf_thomas_rgb16_cached_f32_status_v1 nf_thomas_rgb16_cached_f32_apply_v1(
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile,
    const nf_thomas_field_f32_profile_v1 field_profiles[3],
    const nf_neutral_gauge_f32_profile_v1* gauge_profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    uint32_t parallel_layers,
    const float* relative_log_exposure_chw,
    size_t exposure_floats,
    void* workspace,
    size_t workspace_bytes,
    nf_thomas_rgb16_f32_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]) {
    return nf_thomas_rgb16_cached_f32_apply_internal_v1(
        amplitude_profile,
        field_profiles,
        gauge_profile,
        full_height,
        width,
        row_partition,
        parallel_layers,
        1u,
        relative_log_exposure_chw,
        exposure_floats,
        workspace,
        workspace_bytes,
        sink,
        sink_context,
        raw_field_means);
}

nf_thomas_rgb16_cached_f32_status_v1
nf_thomas_rgb16_cached_f32_apply_parallel_output_v1(
    const nf_granularity_amplitude_f32_profile_v1* amplitude_profile,
    const nf_thomas_field_f32_profile_v1 field_profiles[3],
    const nf_neutral_gauge_f32_profile_v1* gauge_profile,
    size_t full_height,
    size_t width,
    size_t row_partition,
    const float* relative_log_exposure_chw,
    size_t exposure_floats,
    void* workspace,
    size_t workspace_bytes,
    nf_thomas_rgb16_f32_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]) {
    return nf_thomas_rgb16_cached_f32_apply_internal_v1(
        amplitude_profile,
        field_profiles,
        gauge_profile,
        full_height,
        width,
        row_partition,
        3u,
        3u,
        relative_log_exposure_chw,
        exposure_floats,
        workspace,
        workspace_bytes,
        sink,
        sink_context,
        raw_field_means);
}
