#include <dlfcn.h>
#include <jni.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "nf_srgb_quantizer_vectors_v1.h"
#include "nf_srgb_icc_expected_v1.h"

#ifndef NF_SRGB_QUANTIZER_CORE_LIBRARY
#error "NF_SRGB_QUANTIZER_CORE_LIBRARY must name the packaged ABI core"
#endif
#ifndef NF_SRGB_EOTF_LIBRARY
#error "NF_SRGB_EOTF_LIBRARY must name the packaged ABI EOTF"
#endif
#ifndef NF_SRGB_ICC_LIBRARY
#error "NF_SRGB_ICC_LIBRARY must name the packaged ABI ICC accessor"
#endif

typedef int (*nf_apply_fn)(
    const float *, size_t, uint32_t, void *, size_t);
typedef const char *(*nf_identity_fn)(void);
typedef int (*nf_eotf_apply_fn)(
    const void *, size_t, uint32_t, float *, size_t);
typedef size_t (*nf_icc_size_fn)(void);
typedef int (*nf_icc_copy_fn)(uint8_t *, size_t);

static uint16_t NF_CODES16[65536u];
static uint16_t NF_ROUNDTRIP16[65536u];
static float NF_LINEAR16[65536u];
static uint8_t NF_CODES8[256u];
static uint8_t NF_ROUNDTRIP8[256u];
static float NF_LINEAR8[256u];

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

static jstring nf_throw_all(
    JNIEnv *environment,
    void *core,
    void *eotf,
    void *icc,
    const char *message) {
    if (icc != NULL) {
        (void)dlclose(icc);
    }
    if (eotf != NULL) {
        (void)dlclose(eotf);
    }
    return nf_throw(environment, core, message);
}

JNIEXPORT jstring JNICALL
Java_com_neurofilm_srgbquantizer_QuantizerInstrumentation_nativeRun(
    JNIEnv *environment,
    jclass owner) {
    void *core;
    void *symbol;
    nf_apply_fn apply = NULL;
    nf_identity_fn identity = NULL;
    void *eotf = NULL;
    void *icc = NULL;
    nf_eotf_apply_fn eotf_apply = NULL;
    nf_identity_fn eotf_identity = NULL;
    nf_icc_size_fn icc_size = NULL;
    nf_identity_fn icc_identity = NULL;
    nf_icc_copy_fn icc_copy = NULL;
    float inputs[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    float invalid[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8_replay[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8_sentinel[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t q8_before[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint16_t q16[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint16_t q16_replay[NF_SRGB_QUANTIZER_VECTOR_COUNT];
    uint8_t icc_output[NF_SRGB_ICC_EXPECTED_SIZE];
    uint8_t icc_sentinel[NF_SRGB_ICC_EXPECTED_SIZE];
    uint8_t icc_before[NF_SRGB_ICC_EXPECTED_SIZE];
    float eotf_sentinel[4u];
    float eotf_before[4u];
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
    eotf = dlopen(NF_SRGB_EOTF_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    icc = dlopen(NF_SRGB_ICC_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (eotf == NULL || icc == NULL) {
        return nf_throw_all(
            environment, core, eotf, icc, "SDR rail dlopen failed");
    }
    symbol = dlsym(eotf, "nf_srgb_eotf_f32_apply_v1");
    memcpy(&eotf_apply, &symbol, sizeof(eotf_apply));
    symbol = dlsym(eotf, "nf_srgb_eotf_f32_lut_sha256_v1");
    memcpy(&eotf_identity, &symbol, sizeof(eotf_identity));
    symbol = dlsym(icc, "nf_srgb_icc_profile_size_v1");
    memcpy(&icc_size, &symbol, sizeof(icc_size));
    symbol = dlsym(icc, "nf_srgb_icc_profile_sha256_v1");
    memcpy(&icc_identity, &symbol, sizeof(icc_identity));
    symbol = dlsym(icc, "nf_srgb_icc_profile_copy_v1");
    memcpy(&icc_copy, &symbol, sizeof(icc_copy));
    if (
        eotf_apply == NULL
        || eotf_identity == NULL
        || icc_size == NULL
        || icc_identity == NULL
        || icc_copy == NULL
        || strcmp(
            eotf_identity(),
            "1b8f915b4ddf4dc2b4aa961934b05549d23f4593a7b65c2aacfdb7eaea80a1ca"
        ) != 0
        || icc_size() != NF_SRGB_ICC_EXPECTED_SIZE
        || strcmp(
            icc_identity(),
            "217fe48ec958c667f8eef725aa27198f465df95d7662593b90d0a1cc30114356"
        ) != 0
    ) {
        return nf_throw_all(
            environment, core, eotf, icc, "SDR rail identity mismatch");
    }
    for (index = 0u; index < 256u; ++index) {
        NF_CODES8[index] = (uint8_t)index;
    }
    for (index = 0u; index < 65536u; ++index) {
        NF_CODES16[index] = (uint16_t)index;
    }
    if (
        eotf_apply(NF_CODES8, 256u, 8u, NF_LINEAR8, 256u) != 1
        || apply(NF_LINEAR8, 256u, 8u, NF_ROUNDTRIP8, 256u) != 1
        || memcmp(NF_CODES8, NF_ROUNDTRIP8, 256u) != 0
        || eotf_apply(
            NF_CODES16, 65536u, 16u, NF_LINEAR16, 65536u) != 1
        || apply(
            NF_LINEAR16,
            65536u,
            16u,
            NF_ROUNDTRIP16,
            65536u) != 1
        || memcmp(
            NF_CODES16,
            NF_ROUNDTRIP16,
            sizeof(NF_CODES16)) != 0
    ) {
        return nf_throw_all(
            environment, core, eotf, icc, "EOTF roundtrip failed");
    }
    memset(eotf_sentinel, 0xa5, sizeof(eotf_sentinel));
    memcpy(eotf_before, eotf_sentinel, sizeof(eotf_sentinel));
    if (
        eotf_apply(NF_CODES8, 4u, 7u, eotf_sentinel, 4u) != 0
        || memcmp(
            eotf_sentinel, eotf_before, sizeof(eotf_sentinel)) != 0
        || eotf_apply(NF_CODES8, 4u, 8u, eotf_sentinel, 3u) != 0
        || memcmp(
            eotf_sentinel, eotf_before, sizeof(eotf_sentinel)) != 0
    ) {
        return nf_throw_all(
            environment, core, eotf, icc, "EOTF failure atomicity failed");
    }
    memset(icc_sentinel, 0xa5, sizeof(icc_sentinel));
    memcpy(icc_before, icc_sentinel, sizeof(icc_sentinel));
    if (
        icc_copy(icc_sentinel, NF_SRGB_ICC_EXPECTED_SIZE - 1u) != 0
        || memcmp(icc_sentinel, icc_before, sizeof(icc_sentinel)) != 0
        || icc_copy(icc_output, NF_SRGB_ICC_EXPECTED_SIZE) != 1
        || memcmp(
            icc_output,
            NF_SRGB_ICC_EXPECTED,
            NF_SRGB_ICC_EXPECTED_SIZE) != 0
    ) {
        return nf_throw_all(
            environment, core, eotf, icc, "ICC exact copy failed");
    }
    if (
        dlclose(icc) != 0
        || dlclose(eotf) != 0
        || dlclose(core) != 0
    ) {
        return nf_throw(environment, NULL, "SDR rail dlclose failed");
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
        "\"failure_atomic\":true,"
        "\"icc_exact\":true,"
        "\"eotf_q8_roundtrip_exact\":true,"
        "\"eotf_q16_roundtrip_exact\":true,"
        "\"eotf_failure_atomic\":true}",
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
