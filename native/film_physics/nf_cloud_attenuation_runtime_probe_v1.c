#include "nf_cloud_attenuation_f32_v1.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t hash_bytes(const unsigned char* bytes, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0u; index < count; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(void) {
    const size_t pixels = 4096u;
    const size_t total = pixels * 3u;
    const float gain[3] = {0.7874638824806821f, 0.7008792161934756f,
        0.6589818299954651f};
    float* expected = (float*)malloc(total * sizeof(float));
    float* base = (float*)malloc(total * sizeof(float));
    float* density = (float*)malloc(total * sizeof(float));
    float* output = (float*)malloc(total * sizeof(float));
    size_t index;
    int status;
    int bad_status;
    if (expected == NULL || base == NULL || density == NULL || output == NULL) {
        return 2;
    }
    for (index = 0u; index < total; ++index) {
        expected[index] = 0.2f + (float)((index * 17u) % 701u) / 1000.0f;
        base[index] = expected[index] -
            0.08f * (float)((index * 29u) % 997u) / 997.0f;
        density[index] = -7.0f;
        output[index] = -9.0f;
    }
    status = nf_cloud_attenuation_f32_apply_v1(
        expected, base, pixels, gain, density, output);
    base[total - 1u] = 0.0f;
    bad_status = nf_cloud_attenuation_f32_apply_v1(
        expected, base, pixels, gain, density, output);
    printf("status=%d density=%016" PRIx64 " transmittance=%016" PRIx64
           " invalid_status=%d unchanged=%d\n",
        status, hash_bytes((const unsigned char*)density, total * sizeof(float)),
        hash_bytes((const unsigned char*)output, total * sizeof(float)),
        bad_status, density[0] != -7.0f && output[0] != -9.0f);
    free(expected); free(base); free(density); free(output);
    return status == 0 && bad_status != 0 ? 0 : 3;
}
