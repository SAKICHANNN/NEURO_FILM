#include <dlfcn.h>
#include <jni.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "nf_srgb_quantizer_vectors_v1.h"

#ifndef NF_SRGB_QUANTIZER_CORE_LIBRARY
#error "NF_SRGB_QUANTIZER_CORE_LIBRARY must name the packaged ABI core"
#endif

typedef int (*nf_apply_fn)(
    const float *, size_t, uint32_t, void *, size_t);
typedef const char *(*nf_identity_fn)(void);

_Static_assert(
    sizeof(nf_apply_fn) == sizeof(void *),
    "Android function and object pointers must have equal size");
_Static_assert(
    sizeof(nf_identity_fn) == sizeof(void *),
    "Android function and object pointers must have equal size");

static void nf_store_f32_bits(float *destination, uint32_t bits) {
    unsigned char *bytes = (unsigned char *)destination;
    bytes[0] = (unsigned char)(bits & 0xffu);
    bytes[1] = (unsigned char)((bits >> 8u) & 0xffu);
    bytes[2] = (unsigned char)((bits >> 16u) & 0xffu);
    bytes[3] = (unsigned char)((bits >> 24u) & 0xffu);
}

static jstring nf_throw(
    JNIEnv *environment,
    void *core,
    const char *message) {
    jclass exception_class;
    if (core != NULL) {
        (void)dlclose(core);
    }
    exception_class = (*environment)->FindClass(
        environment, "java/lang/RuntimeException");
    if (exception_class != NULL) {
        (*environment)->ThrowNew(
            environment, exception_class, message);
    }
    return NULL;
}

JNIEXPORT jstring JNICALL
Java_com_neurofilm_srgbquantizer_QuantizerInstrumentation_nativeRun(
    JNIEnv *environment,
    jclass owner) {
    void *core;
    void *symbol;
    nf_apply_fn apply = NULL;
    nf_identity_fn identity = NULL;
    float inputs[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    float invalid[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8_replay[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8_sentinel[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8_before[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint16_t q16[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint16_t q16_replay[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    size_t index;
    char result[1024];
    int written;
    (void)owner;

    core = dlopen(
        NF_SRGB_QUANTIZER_CORE_LIBRARY,
        RTLD_NOW | RTLD_LOCAL);
    if (core == NULL) {
        return nf_throw(environment, NULL, "quantizer core dlopen failed");
    }
    symbol = dlsym(core, "nf_srgb_oetf_quantize_apply_v1");
    memcpy(&apply, &symbol, sizeof(apply));
    symbol = dlsym(
        core, "nf_srgb_oetf_quantize_thresholds_sha256_v1");
    memcpy(&identity, &symbol, sizeof(identity));
    if (apply == NULL || identity == NULL) {
        return nf_throw(environment, core, "quantizer symbol missing");
    }
    if (strcmp(identity(), NF_SRGB_QUANTIZER_THRESHOLD_ID) != 0) {
        return nf_throw(environment, core, "threshold identity mismatch");
    }
    for (index = 0u; index < NF_SRGB_QUANTIZER_VECTOR_COUNT; ++index) {
        nf_store_f32_bits(&inputs[index], NF_SRGB_QUANTIZER_INPUT_BITS[index]);
    }
    if (
        apply(
            inputs,
            NF_SRGB_QUANTIZER_VECTOR_COUNT,
            8u,
            q8,
            NF_SRGB_QUANTIZER_VECTOR_COUNT) != 1
        || apply(
            inputs,
            NF_SRGB_QUANTIZER_VECTOR_COUNT,
            8u,
            q8_replay,
            NF_SRGB_QUANTIZER_VECTOR_COUNT) != 1
        || memcmp(q8, NF_SRGB_QUANTIZER_EXPECTED_Q8, sizeof(q8)) != 0
        || memcmp(q8, q8_replay, sizeof(q8)) != 0
    ) {
        return nf_throw(environment, core, "sRGB8 exact replay failed");
    }
    if (
        apply(
            inputs,
            NF_SRGB_QUANTIZER_VECTOR_COUNT,
            16u,
            q16,
            NF_SRGB_QUANTIZER_VECTOR_COUNT) != 1
        || apply(
            inputs,
            NF_SRGB_QUANTIZER_VECTOR_COUNT,
            16u,
            q16_replay,
            NF_SRGB_QUANTIZER_VECTOR_COUNT) != 1
        || memcmp(q16, NF_SRGB_QUANTIZER_EXPECTED_Q16, sizeof(q16)) != 0
        || memcmp(q16, q16_replay, sizeof(q16)) != 0
    ) {
        return nf_throw(environment, core, "sRGB16 exact replay failed");
    }
    memcpy(invalid, inputs, sizeof(inputs));
    nf_store_f32_bits(
        &invalid[NF_SRGB_QUANTIZER_VECTOR_COUNT - 1u],
        0x7f800000u);
    memset(q8_sentinel, 0xa5, sizeof(q8_sentinel));
    memcpy(q8_before, q8_sentinel, sizeof(q8_sentinel));
    if (
        apply(
            invalid,
            NF_SRGB_QUANTIZER_VECTOR_COUNT,
            8u,
            q8_sentinel,
            NF_SRGB_QUANTIZER_VECTOR_COUNT) != 0
        || memcmp(q8_sentinel, q8_before, sizeof(q8_sentinel)) != 0
        || apply(
            inputs,
            NF_SRGB_QUANTIZER_VECTOR_COUNT,
            8u,
            q8_sentinel,
            NF_SRGB_QUANTIZER_VECTOR_COUNT - 1u) != 0
        || memcmp(q8_sentinel, q8_before, sizeof(q8_sentinel)) != 0
    ) {
        return nf_throw(environment, core, "failure atomicity failed");
    }
    if (dlclose(core) != 0) {
        return nf_throw(environment, NULL, "quantizer core dlclose failed");
    }
    written = snprintf(
        result,
        sizeof(result),
        "{\"schema\":\"neuro-film.android-srgb-quantizer-runtime.v1\","
        "\"status\":\"PASS\","
        "\"sample_count\":%u,"
        "\"threshold_identity\":\"%s\","
        "\"vector_sha256\":\"%s\","
        "\"input_sha256\":\"%s\","
        "\"q8_sha256\":\"%s\","
        "\"q16_sha256\":\"%s\","
        "\"q8_exact\":true,"
        "\"q16_exact\":true,"
        "\"inner_replay_exact\":true,"
        "\"failure_atomic\":true}",
        (unsigned int)NF_SRGB_QUANTIZER_VECTOR_COUNT,
        NF_SRGB_QUANTIZER_THRESHOLD_ID,
        NF_SRGB_QUANTIZER_VECTOR_SHA256,
        NF_SRGB_QUANTIZER_INPUT_SHA256,
        NF_SRGB_QUANTIZER_Q8_SHA256,
        NF_SRGB_QUANTIZER_Q16_SHA256);
    if (written < 0 || (size_t)written >= sizeof(result)) {
        return nf_throw(environment, NULL, "runtime result overflow");
    }
    return (*environment)->NewStringUTF(environment, result);
}
