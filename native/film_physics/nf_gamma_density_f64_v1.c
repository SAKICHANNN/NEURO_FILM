#include "nf_gamma_density_f64_v1.h"

#include <float.h>
#include <math.h>

static double nf_gamma_p(double shape, double value) {
    const double log_gamma = lgamma(shape);
    if (value <= 0.0) {
        return 0.0;
    }
    if (value < shape + 1.0) {
        double ap = shape;
        double term = 1.0 / shape;
        double sum = term;
        uint32_t index;
        for (index = 1U; index <= 512U; ++index) {
            ap += 1.0;
            term *= value / ap;
            sum += term;
            if (fabs(term) <= fabs(sum) * (8.0 * DBL_EPSILON)) {
                break;
            }
        }
        return sum * exp(-value + shape * log(value) - log_gamma);
    }
    {
        const double floor_value = DBL_MIN / DBL_EPSILON;
        double b = value + 1.0 - shape;
        double c = 1.0 / floor_value;
        double d = 1.0 / b;
        double h = d;
        uint32_t index;
        for (index = 1U; index <= 512U; ++index) {
            const double fi = (double)index;
            const double an = -fi * (fi - shape);
            double factor;
            b += 2.0;
            d = an * d + b;
            if (fabs(d) < floor_value) {
                d = floor_value;
            }
            c = b + an / c;
            if (fabs(c) < floor_value) {
                c = floor_value;
            }
            d = 1.0 / d;
            factor = d * c;
            h *= factor;
            if (fabs(factor - 1.0) <= 8.0 * DBL_EPSILON) {
                break;
            }
        }
        return 1.0 - exp(-value + shape * log(value) - log_gamma) * h;
    }
}

static int nf_gamma_inverse(
    double probability,
    double shape,
    uint32_t iterations,
    double *result) {
    double low = 0.0;
    double high = shape > 1.0 ? shape : 1.0;
    uint32_t index;
    for (index = 0U; index < 128U && nf_gamma_p(shape, high) < probability; ++index) {
        high *= 2.0;
        if (!isfinite(high)) {
            return 0;
        }
    }
    if (nf_gamma_p(shape, high) < probability) {
        return 0;
    }
    for (index = 0U; index < iterations; ++index) {
        const double midpoint = low + 0.5 * (high - low);
        if (nf_gamma_p(shape, midpoint) < probability) {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    *result = low + 0.5 * (high - low);
    return isfinite(*result) && *result > 0.0;
}

uint32_t nf_gamma_density_f64_abi_version_v1(void) {
    return 1U;
}

int nf_gamma_density_f64_apply_v1(
    const double *uniforms,
    const double *shapes,
    const double *scales,
    size_t sample_count,
    uint32_t inverse_iterations,
    double *output,
    size_t output_count,
    nf_gamma_density_diagnostics_v1 *diagnostics) {
    size_t index;
    double minimum_output = DBL_MAX;
    double maximum_output = 0.0;
    if (uniforms == NULL || shapes == NULL || scales == NULL || output == NULL ||
        diagnostics == NULL || sample_count == 0U || output_count < sample_count ||
        inverse_iterations < 32U || inverse_iterations > 128U ||
        diagnostics->struct_size != sizeof(*diagnostics) ||
        diagnostics->abi_version != 1U) {
        return 1;
    }
    for (index = 0U; index < sample_count; ++index) {
        double standardized;
        double developed;
        if (!isfinite(uniforms[index]) || uniforms[index] <= 0.0 ||
            uniforms[index] >= 1.0 || !isfinite(shapes[index]) ||
            shapes[index] <= 0.0 || !isfinite(scales[index]) || scales[index] <= 0.0 ||
            !nf_gamma_inverse(
                uniforms[index], shapes[index], inverse_iterations, &standardized)) {
            return 2;
        }
        developed = standardized * scales[index];
        if (!isfinite(developed) || developed <= 0.0) {
            return 2;
        }
        if (developed < minimum_output) {
            minimum_output = developed;
        }
        if (developed > maximum_output) {
            maximum_output = developed;
        }
    }
    for (index = 0U; index < sample_count; ++index) {
        double standardized;
        if (!nf_gamma_inverse(
                uniforms[index], shapes[index], inverse_iterations, &standardized)) {
            return 3;
        }
        output[index] = standardized * scales[index];
    }
    diagnostics->inverse_iterations = inverse_iterations;
    diagnostics->sample_count = sample_count;
    diagnostics->minimum_output = minimum_output;
    diagnostics->maximum_output = maximum_output;
    return 0;
}
