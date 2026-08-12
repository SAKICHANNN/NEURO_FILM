#define NF_DETERMINISTIC_LOG10_F32_BUILD
#include "nf_deterministic_log10_f32_v1.h"

#include <string.h>

static double nf_round_add(double a, double b) {
    volatile double value = a + b;
    return value;
}

static double nf_round_mul(double a, double b) {
    volatile double value = a * b;
    return value;
}

static int nf_valid(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) == 0u && (bits & 0x7f800000u) != 0u &&
        (bits & 0x7f800000u) != 0x7f800000u;
}

static float nf_neg_log10(float value) {
    uint32_t bits;
    int exponent;
    double mantissa;
    double z;
    double z2;
    double polynomial;
    double ln_value;
    memcpy(&bits, &value, sizeof(bits));
    exponent = (int)((bits >> 23u) & 0xffu) - 127;
    mantissa = 1.0 + (double)(bits & 0x007fffffu) / 8388608.0;
    if (mantissa > 1.4142135623730950488) {
        mantissa *= 0.5;
        ++exponent;
    }
    z = (mantissa - 1.0) / (mantissa + 1.0);
    z2 = nf_round_mul(z, z);
    polynomial = 1.0 / 15.0;
    polynomial = nf_round_add(1.0 / 13.0, nf_round_mul(z2, polynomial));
    polynomial = nf_round_add(1.0 / 11.0, nf_round_mul(z2, polynomial));
    polynomial = nf_round_add(1.0 / 9.0, nf_round_mul(z2, polynomial));
    polynomial = nf_round_add(1.0 / 7.0, nf_round_mul(z2, polynomial));
    polynomial = nf_round_add(1.0 / 5.0, nf_round_mul(z2, polynomial));
    polynomial = nf_round_add(1.0 / 3.0, nf_round_mul(z2, polynomial));
    polynomial = nf_round_add(1.0, nf_round_mul(z2, polynomial));
    ln_value = nf_round_add(
        nf_round_mul((double)exponent, 0.69314718055994530942),
        nf_round_mul(2.0 * z, polynomial));
    return (float)nf_round_mul(-ln_value, 0.43429448190325182765);
}

uint32_t nf_deterministic_log10_f32_abi_version_v1(void) {
    return NF_DETERMINISTIC_LOG10_F32_ABI_VERSION_V1;
}

nf_deterministic_log10_f32_status_v1 nf_deterministic_neg_log10_f32_apply_v1(
    const float* input, size_t count, float* output) {
    if (input == NULL || output == NULL || count == 0u || input == output) {
        return NF_DETERMINISTIC_LOG10_F32_INVALID_ARGUMENT_V1;
    }
    for (size_t index = 0u; index < count; ++index) {
        if (!nf_valid(input[index])) {
            return NF_DETERMINISTIC_LOG10_F32_DOMAIN_ERROR_V1;
        }
    }
    for (size_t index = 0u; index < count; ++index) {
        output[index] = nf_neg_log10(input[index]);
    }
    return NF_DETERMINISTIC_LOG10_F32_OK_V1;
}
