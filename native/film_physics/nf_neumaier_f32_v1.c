#define NF_NEUMAIER_F32_BUILD
#include "nf_neumaier_f32_v1.h"

#include <math.h>

uint32_t nf_neumaier_f32_abi_version_v1(void) {
    return NF_NEUMAIER_F32_ABI_VERSION_V1;
}

int nf_neumaier_f32_accumulate_v1(
    const float* values,
    size_t value_count,
    double* total,
    double* compensation) {
    size_t index;
    double running_total;
    double running_compensation;
    if (values == NULL || value_count == 0u || total == NULL ||
        compensation == NULL || !isfinite(*total) ||
        !isfinite(*compensation)) {
        return 0;
    }
    for (index = 0; index < value_count; ++index) {
        if (!isfinite((double)values[index])) {
            return 0;
        }
    }
    running_total = *total;
    running_compensation = *compensation;
    for (index = 0; index < value_count; ++index) {
        const double sample = (double)values[index];
        const double updated = running_total + sample;
        if (fabs(running_total) >= fabs(sample)) {
            running_compensation += (running_total - updated) + sample;
        } else {
            running_compensation += (sample - updated) + running_total;
        }
        running_total = updated;
    }
    *total = running_total;
    *compensation = running_compensation;
    return 1;
}
