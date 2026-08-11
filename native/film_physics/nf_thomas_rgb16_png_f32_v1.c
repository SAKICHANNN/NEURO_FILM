#define NF_THOMAS_RGB16_PNG_F32_BUILD
#define NF_THOMAS_RGB16_F32_BUILD
#include "nf_thomas_rgb16_png_f32_v1.h"
#include "reference_srgb_icc_profile_v1.h"

#include <stdint.h>

typedef struct nf_png_state_v1 {
    nf_thomas_rgb16_png_f32_byte_sink_v1 sink;
    void* sink_context;
    uint8_t* payload;
    size_t payload_capacity;
    uint8_t* raw_row;
    size_t width;
    size_t height;
    size_t next_row;
    uint32_t adler_s1;
    uint32_t adler_s2;
    int started;
} nf_png_state_v1;

static void nf_store_be32(uint8_t output[4], uint32_t value) {
    output[0] = (uint8_t)(value >> 24u);
    output[1] = (uint8_t)(value >> 16u);
    output[2] = (uint8_t)(value >> 8u);
    output[3] = (uint8_t)value;
}

static uint32_t nf_crc32_update(uint32_t crc, const uint8_t* bytes, size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        uint32_t value = crc ^ (uint32_t)bytes[index];
        uint32_t bit;
        for (bit = 0u; bit < 8u; ++bit) {
            value = (value >> 1u) ^ (0xedb88320u & (0u - (value & 1u)));
        }
        crc = value;
    }
    return crc;
}

static void nf_adler32_update(
    uint32_t* s1,
    uint32_t* s2,
    const uint8_t* bytes,
    size_t count) {
    const uint32_t modulus = 65521u;
    size_t offset = 0u;
    while (offset < count) {
        size_t block = count - offset;
        size_t index;
        if (block > 5552u) {
            block = 5552u;
        }
        for (index = 0u; index < block; ++index) {
            *s1 += bytes[offset + index];
            *s2 += *s1;
        }
        *s1 %= modulus;
        *s2 %= modulus;
        offset += block;
    }
}

static int nf_emit(
    nf_png_state_v1* state,
    const uint8_t* bytes,
    size_t count) {
    return count != 0u && state->sink(state->sink_context, bytes, count) != 0;
}

static int nf_emit_chunk(
    nf_png_state_v1* state,
    const uint8_t type[4],
    const uint8_t* payload,
    size_t payload_bytes) {
    uint8_t length[4];
    uint8_t checksum[4];
    uint32_t crc;
    if (payload_bytes > UINT32_MAX) {
        return 0;
    }
    nf_store_be32(length, (uint32_t)payload_bytes);
    crc = nf_crc32_update(0xffffffffu, type, 4u);
    crc = nf_crc32_update(crc, payload, payload_bytes) ^ 0xffffffffu;
    nf_store_be32(checksum, crc);
    return nf_emit(state, length, 4u) && nf_emit(state, type, 4u) &&
        (payload_bytes == 0u || nf_emit(state, payload, payload_bytes)) &&
        nf_emit(state, checksum, 4u);
}

static size_t nf_stored_zlib(
    uint8_t* output,
    size_t capacity,
    const uint8_t* input,
    size_t input_bytes) {
    uint32_t s1 = 1u;
    uint32_t s2 = 0u;
    size_t offset = 0u;
    size_t written = 0u;
    if (capacity < 2u) {
        return 0u;
    }
    output[written++] = 0x78u;
    output[written++] = 0x01u;
    while (offset < input_bytes) {
        size_t block = input_bytes - offset;
        uint16_t length;
        uint16_t inverse;
        if (block > 65535u) {
            block = 65535u;
        }
        if (written > capacity - 5u - block) {
            return 0u;
        }
        length = (uint16_t)block;
        inverse = (uint16_t)~length;
        output[written++] = (offset + block == input_bytes) ? 0x01u : 0x00u;
        output[written++] = (uint8_t)length;
        output[written++] = (uint8_t)(length >> 8u);
        output[written++] = (uint8_t)inverse;
        output[written++] = (uint8_t)(inverse >> 8u);
        {
            size_t index;
            for (index = 0u; index < block; ++index) {
                output[written + index] = input[offset + index];
            }
        }
        nf_adler32_update(&s1, &s2, input + offset, block);
        written += block;
        offset += block;
    }
    if (written > capacity - 4u) {
        return 0u;
    }
    nf_store_be32(output + written, (s2 << 16u) | s1);
    return written + 4u;
}

static int nf_emit_header(nf_png_state_v1* state) {
    static const uint8_t signature[8] = {
        0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au};
    static const uint8_t ihdr_type[4] = {'I', 'H', 'D', 'R'};
    static const uint8_t iccp_type[4] = {'i', 'C', 'C', 'P'};
    uint8_t ihdr[13];
    uint8_t profile[NF_SRGB_ICC_PROFILE_V1_SIZE];
    uint8_t compressed[NF_SRGB_ICC_PROFILE_V1_SIZE + 16u];
    uint8_t payload[11u + 2u + NF_SRGB_ICC_PROFILE_V1_SIZE + 16u];
    size_t compressed_bytes;
    static const uint8_t name[11] = {
        'I', 'C', 'C', ' ', 'P', 'r', 'o', 'f', 'i', 'l', 'e'};
    size_t index;
    if (state->width > UINT32_MAX || state->height > UINT32_MAX ||
        nf_srgb_icc_profile_copy_v1(profile, sizeof(profile)) != 1) {
        return 0;
    }
    nf_store_be32(ihdr, (uint32_t)state->width);
    nf_store_be32(ihdr + 4u, (uint32_t)state->height);
    ihdr[8] = 16u;
    ihdr[9] = 2u;
    ihdr[10] = 0u;
    ihdr[11] = 0u;
    ihdr[12] = 0u;
    compressed_bytes = nf_stored_zlib(
        compressed, sizeof(compressed), profile, sizeof(profile));
    if (compressed_bytes == 0u || compressed_bytes > sizeof(payload) - 13u) {
        return 0;
    }
    for (index = 0u; index < sizeof(name); ++index) {
        payload[index] = name[index];
    }
    payload[sizeof(name)] = 0u;
    payload[sizeof(name) + 1u] = 0u;
    for (index = 0u; index < compressed_bytes; ++index) {
        payload[sizeof(name) + 2u + index] = compressed[index];
    }
    return nf_emit(state, signature, sizeof(signature)) &&
        nf_emit_chunk(state, ihdr_type, ihdr, sizeof(ihdr)) &&
        nf_emit_chunk(
            state,
            iccp_type,
            payload,
            sizeof(name) + 2u + compressed_bytes);
}

static int nf_png_rows(
    void* context,
    size_t row_start,
    size_t row_count,
    const uint16_t* rgb16,
    size_t rgb16_values) {
    static const uint8_t idat_type[4] = {'I', 'D', 'A', 'T'};
    nf_png_state_v1* state = (nf_png_state_v1*)context;
    size_t row;
    size_t raw_bytes;
    if (state == NULL || rgb16 == NULL || row_start != state->next_row ||
        row_count == 0u || row_count > state->height - row_start ||
        state->width > (SIZE_MAX - 1u) / 6u ||
        rgb16_values != row_count * state->width * 3u) {
        return 0;
    }
    raw_bytes = 1u + 6u * state->width;
    if (!state->started) {
        if (!nf_emit_header(state)) {
            return 0;
        }
        state->started = 1;
    }
    for (row = 0u; row < row_count; ++row) {
        uint8_t* output = state->payload;
        uint8_t* raw = state->raw_row;
        size_t written = 0u;
        size_t sample;
        size_t remaining;
        const int first = row_start + row == 0u;
        const int final = row_start + row + 1u == state->height;
        if (first) {
            output[written++] = 0x78u;
            output[written++] = 0x01u;
        }
        raw[0] = 0u;
        for (sample = 0u; sample < state->width * 3u; ++sample) {
            const uint16_t value = rgb16[row * state->width * 3u + sample];
            raw[1u + 2u * sample] = (uint8_t)(value >> 8u);
            raw[2u + 2u * sample] = (uint8_t)value;
        }
        nf_adler32_update(
            &state->adler_s1, &state->adler_s2, raw, raw_bytes);
        remaining = raw_bytes;
        while (remaining != 0u) {
            const size_t block = remaining > 65535u ? 65535u : remaining;
            const uint16_t length = (uint16_t)block;
            const uint16_t inverse = (uint16_t)~length;
            const size_t source = raw_bytes - remaining;
            size_t index;
            if (written > state->payload_capacity - 5u - block) {
                return 0;
            }
            output[written++] = (final && block == remaining) ? 0x01u : 0x00u;
            output[written++] = (uint8_t)length;
            output[written++] = (uint8_t)(length >> 8u);
            output[written++] = (uint8_t)inverse;
            output[written++] = (uint8_t)(inverse >> 8u);
            for (index = 0u; index < block; ++index) {
                output[written + index] = raw[source + index];
            }
            written += block;
            remaining -= block;
        }
        if (final) {
            if (written > state->payload_capacity - 4u) {
                return 0;
            }
            nf_store_be32(
                output + written, (state->adler_s2 << 16u) | state->adler_s1);
            written += 4u;
        }
        if (written > NF_THOMAS_RGB16_PNG_F32_MAX_IDAT_PAYLOAD_V1 ||
            !nf_emit_chunk(state, idat_type, output, written)) {
            return 0;
        }
        state->next_row += 1u;
    }
    return 1;
}

uint32_t nf_thomas_rgb16_png_f32_abi_version_v1(void) {
    return NF_THOMAS_RGB16_PNG_F32_ABI_VERSION_V1;
}

nf_thomas_rgb16_png_f32_status_v1 nf_thomas_rgb16_png_f32_workspace_bytes_v1(
    size_t width,
    size_t row_partition,
    size_t* workspace_bytes) {
    size_t inner;
    size_t raw;
    size_t blocks;
    size_t payload;
    if (workspace_bytes == NULL || width == 0u ||
        nf_thomas_rgb16_f32_workspace_bytes_v1(width, row_partition, &inner) !=
            NF_THOMAS_RGB16_F32_OK_V1 ||
        width > (SIZE_MAX - 1u) / 6u) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    raw = 1u + 6u * width;
    blocks = (raw + 65534u) / 65535u;
    if (blocks > (SIZE_MAX - raw - 6u) / 5u) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    payload = raw + 6u + 5u * blocks;
    if (payload > NF_THOMAS_RGB16_PNG_F32_MAX_IDAT_PAYLOAD_V1 ||
        inner > SIZE_MAX - payload || inner + payload > SIZE_MAX - raw) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_bytes = inner + payload + raw;
    return NF_THOMAS_RGB16_PNG_F32_OK_V1;
}

nf_thomas_rgb16_png_f32_status_v1
nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1(
    size_t full_height,
    size_t width,
    size_t row_partition,
    size_t* workspace_bytes) {
    size_t inner;
    size_t raw;
    size_t blocks;
    size_t payload;
    if (workspace_bytes == NULL || width == 0u ||
        nf_thomas_rgb16_cached_f32_workspace_bytes_v1(
            full_height, width, row_partition, 3u, &inner) !=
            NF_THOMAS_RGB16_CACHED_F32_OK_V1 ||
        width > (SIZE_MAX - 1u) / 6u) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    raw = 1u + 6u * width;
    blocks = (raw + 65534u) / 65535u;
    if (blocks > (SIZE_MAX - raw - 6u) / 5u) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    payload = raw + 6u + 5u * blocks;
    if (payload > NF_THOMAS_RGB16_PNG_F32_MAX_IDAT_PAYLOAD_V1 ||
        inner > SIZE_MAX - payload || inner + payload > SIZE_MAX - raw) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    *workspace_bytes = inner + payload + raw;
    return NF_THOMAS_RGB16_PNG_F32_OK_V1;
}

nf_thomas_rgb16_png_f32_status_v1 nf_thomas_rgb16_png_f32_apply_v1(
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
    nf_thomas_rgb16_png_f32_byte_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]) {
    static const uint8_t iend_type[4] = {'I', 'E', 'N', 'D'};
    size_t required;
    size_t inner;
    size_t raw;
    size_t blocks;
    size_t payload;
    double means[3] = {-13.0, -13.0, -13.0};
    nf_png_state_v1 state;
    nf_thomas_rgb16_f32_status_v1 status;
    if (sink == NULL || raw_field_means == NULL || workspace == NULL ||
        (uintptr_t)workspace % _Alignof(float) != 0u ||
        nf_thomas_rgb16_png_f32_workspace_bytes_v1(
            width, row_partition, &required) != NF_THOMAS_RGB16_PNG_F32_OK_V1 ||
        nf_thomas_rgb16_f32_workspace_bytes_v1(width, row_partition, &inner) !=
            NF_THOMAS_RGB16_F32_OK_V1 ||
        workspace_bytes < required) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    state.sink = sink;
    state.sink_context = sink_context;
    state.payload = (uint8_t*)workspace + inner;
    raw = 1u + 6u * width;
    blocks = (raw + 65534u) / 65535u;
    payload = raw + 6u + 5u * blocks;
    state.payload_capacity = payload;
    state.raw_row = state.payload + payload;
    state.width = width;
    state.height = full_height;
    state.next_row = 0u;
    state.adler_s1 = 1u;
    state.adler_s2 = 0u;
    state.started = 0;
    status = nf_thomas_rgb16_f32_apply_v1(
        amplitude_profile,
        field_profiles,
        gauge_profile,
        full_height,
        width,
        row_partition,
        relative_log_exposure_chw,
        exposure_floats,
        workspace,
        inner,
        nf_png_rows,
        &state,
        means);
    if (status != NF_THOMAS_RGB16_F32_OK_V1) {
        return (nf_thomas_rgb16_png_f32_status_v1)status;
    }
    if (!state.started || state.next_row != full_height ||
        !nf_emit_chunk(&state, iend_type, NULL, 0u)) {
        return NF_THOMAS_RGB16_PNG_F32_CALLBACK_FAILED_V1;
    }
    raw_field_means[0] = means[0];
    raw_field_means[1] = means[1];
    raw_field_means[2] = means[2];
    return NF_THOMAS_RGB16_PNG_F32_OK_V1;
}

nf_thomas_rgb16_png_f32_status_v1
nf_thomas_rgb16_png_cached_parallel_apply_v1(
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
    nf_thomas_rgb16_png_f32_byte_sink_v1 sink,
    void* sink_context,
    double raw_field_means[3]) {
    static const uint8_t iend_type[4] = {'I', 'E', 'N', 'D'};
    size_t required;
    size_t inner;
    size_t raw;
    size_t blocks;
    size_t payload;
    double means[3] = {-13.0, -13.0, -13.0};
    nf_png_state_v1 state;
    nf_thomas_rgb16_cached_f32_status_v1 status;
    if (sink == NULL || raw_field_means == NULL || workspace == NULL ||
        (uintptr_t)workspace % _Alignof(float) != 0u ||
        nf_thomas_rgb16_png_cached_parallel_workspace_bytes_v1(
            full_height, width, row_partition, &required) !=
            NF_THOMAS_RGB16_PNG_F32_OK_V1 ||
        nf_thomas_rgb16_cached_f32_workspace_bytes_v1(
            full_height, width, row_partition, 3u, &inner) !=
            NF_THOMAS_RGB16_CACHED_F32_OK_V1 ||
        workspace_bytes < required) {
        return NF_THOMAS_RGB16_PNG_F32_INVALID_ARGUMENT_V1;
    }
    state.sink = sink;
    state.sink_context = sink_context;
    state.payload = (uint8_t*)workspace + inner;
    raw = 1u + 6u * width;
    blocks = (raw + 65534u) / 65535u;
    payload = raw + 6u + 5u * blocks;
    state.payload_capacity = payload;
    state.raw_row = state.payload + payload;
    state.width = width;
    state.height = full_height;
    state.next_row = 0u;
    state.adler_s1 = 1u;
    state.adler_s2 = 0u;
    state.started = 0;
    status = nf_thomas_rgb16_cached_f32_apply_parallel_output_v1(
        amplitude_profile,
        field_profiles,
        gauge_profile,
        full_height,
        width,
        row_partition,
        relative_log_exposure_chw,
        exposure_floats,
        workspace,
        inner,
        nf_png_rows,
        &state,
        means);
    if (status != NF_THOMAS_RGB16_CACHED_F32_OK_V1) {
        return (nf_thomas_rgb16_png_f32_status_v1)status;
    }
    if (!state.started || state.next_row != full_height ||
        !nf_emit_chunk(&state, iend_type, NULL, 0u)) {
        return NF_THOMAS_RGB16_PNG_F32_CALLBACK_FAILED_V1;
    }
    raw_field_means[0] = means[0];
    raw_field_means[1] = means[1];
    raw_field_means[2] = means[2];
    return NF_THOMAS_RGB16_PNG_F32_OK_V1;
}
