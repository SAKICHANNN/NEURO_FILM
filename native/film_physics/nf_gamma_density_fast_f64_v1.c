#include "nf_gamma_density_hybrid_f64_v1.c"
#include "nf_gamma_density_fast_f64_v1.h"

static double nf_gamma_cf(double probability, double shape) {
    const double z = nf_acklam_inverse_normal(probability);
    const double root = sqrt(shape);
    const double z2 = z * z;
    const double z3 = z2 * z;
    const double z4 = z2 * z2;
    return shape + root * z + (z2 - 1.0) / 3.0 +
           (z3 - 7.0 * z) / (36.0 * root) -
           (3.0 * z4 + 7.0 * z2 - 16.0) / (810.0 * shape);
}

static int nf_gamma_newton(
    double probability, double shape, uint32_t iterations, double *result) {
    double value = nf_gamma_cf(probability, shape);
    double low = 0.0;
    double high = shape > value ? 2.0 * shape : 2.0 * value;
    uint32_t index;
    if (!isfinite(value) || value <= 0.0 || !isfinite(high) || high <= value) {
        return 0;
    }
    while (nf_gamma_p(shape, high) < probability) {
        high *= 2.0;
        if (!isfinite(high)) {
            return 0;
        }
    }
    for (index = 0U; index < iterations; ++index) {
        const double cdf = nf_gamma_p(shape, value);
        const double log_pdf = (shape - 1.0) * log(value) - value - lgamma(shape);
        const double pdf = exp(log_pdf);
        double proposed;
        if (fabs(cdf - probability) <= 32.0 * DBL_EPSILON) {
            break;
        }
        if (cdf < probability) {
            low = value;
        } else {
            high = value;
        }
        proposed = value - (cdf - probability) / pdf;
        if (!isfinite(proposed) || proposed <= low || proposed >= high) {
            proposed = low + 0.5 * (high - low);
        }
        value = proposed;
    }
    *result = value;
    return isfinite(value) && value > 0.0;
}

static int nf_gamma_inverse_fast(
    double probability,
    double shape,
    uint32_t direct_iterations,
    uint32_t newton_iterations,
    double direct_shape_upper,
    double newton_shape_upper,
    double *result) {
    if (shape < direct_shape_upper) {
        return nf_gamma_inverse(probability, shape, direct_iterations, result);
    }
    if (shape < newton_shape_upper) {
        return nf_gamma_newton(probability, shape, newton_iterations, result);
    }
    *result = nf_gamma_cf(probability, shape);
    return isfinite(*result) && *result > 0.0;
}

uint32_t nf_gamma_density_fast_f64_abi_version_v1(void) {
    return 1U;
}

int nf_gamma_density_fast_f64_apply_v1(
    const double *uniforms,
    const double *shapes,
    const double *scales,
    size_t sample_count,
    uint32_t direct_iterations,
    uint32_t newton_iterations,
    double direct_shape_upper,
    double newton_shape_upper,
    double *output,
    size_t output_count,
    nf_gamma_density_fast_diagnostics_v1 *diagnostics) {
    size_t index;
    size_t direct_count = 0U;
    size_t newton_count = 0U;
    size_t asymptotic_count = 0U;
    double minimum_output = DBL_MAX;
    double maximum_output = 0.0;
    if (uniforms == NULL || shapes == NULL || scales == NULL || output == NULL ||
        diagnostics == NULL || sample_count == 0U || output_count < sample_count ||
        direct_iterations < 32U || direct_iterations > 128U ||
        newton_iterations == 0U || newton_iterations > 16U ||
        !isfinite(direct_shape_upper) || !isfinite(newton_shape_upper) ||
        direct_shape_upper <= 1.0 || newton_shape_upper <= direct_shape_upper ||
        diagnostics->struct_size != sizeof(*diagnostics) || diagnostics->abi_version != 1U) {
        return 1;
    }
    for (index = 0U; index < sample_count; ++index) {
        double standardized;
        double developed;
        if (!isfinite(uniforms[index]) || uniforms[index] <= 0.0 || uniforms[index] >= 1.0 ||
            !isfinite(shapes[index]) || shapes[index] <= 0.0 ||
            !isfinite(scales[index]) || scales[index] <= 0.0 ||
            !nf_gamma_inverse_fast(uniforms[index], shapes[index], direct_iterations,
                                   newton_iterations, direct_shape_upper,
                                   newton_shape_upper, &standardized)) {
            return 2;
        }
        developed = standardized * scales[index];
        if (!isfinite(developed) || developed <= 0.0) {
            return 2;
        }
        direct_count += shapes[index] < direct_shape_upper ? 1U : 0U;
        newton_count += shapes[index] >= direct_shape_upper && shapes[index] < newton_shape_upper ? 1U : 0U;
        asymptotic_count += shapes[index] >= newton_shape_upper ? 1U : 0U;
        minimum_output = developed < minimum_output ? developed : minimum_output;
        maximum_output = developed > maximum_output ? developed : maximum_output;
    }
    for (index = 0U; index < sample_count; ++index) {
        double standardized;
        if (!nf_gamma_inverse_fast(uniforms[index], shapes[index], direct_iterations,
                                   newton_iterations, direct_shape_upper,
                                   newton_shape_upper, &standardized)) {
            return 3;
        }
        output[index] = standardized * scales[index];
    }
    diagnostics->direct_iterations = direct_iterations;
    diagnostics->newton_iterations = newton_iterations;
    diagnostics->sample_count = sample_count;
    diagnostics->direct_branch_count = direct_count;
    diagnostics->newton_branch_count = newton_count;
    diagnostics->asymptotic_branch_count = asymptotic_count;
    diagnostics->minimum_output = minimum_output;
    diagnostics->maximum_output = maximum_output;
    return 0;
}
