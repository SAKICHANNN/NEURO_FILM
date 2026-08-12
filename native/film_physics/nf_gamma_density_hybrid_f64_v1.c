#include "nf_gamma_density_f64_v1.c"
#include "nf_gamma_density_hybrid_f64_v1.h"

static double nf_acklam_inverse_normal(double probability) {
    static const double a[6] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};
    static const double c[6] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    double q;
    double r;
    if (probability < 0.02425) {
        q = sqrt(-2.0 * log(probability));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (probability > 0.97575) {
        q = sqrt(-2.0 * log(1.0 - probability));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    q = probability - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

static int nf_gamma_inverse_hybrid(
    double probability,
    double shape,
    uint32_t iterations,
    double threshold,
    double *result) {
    if (shape < threshold) {
        return nf_gamma_inverse(probability, shape, iterations, result);
    }
    {
        const double z = nf_acklam_inverse_normal(probability);
        const double root = sqrt(shape);
        const double z2 = z * z;
        const double z3 = z2 * z;
        const double z4 = z2 * z2;
        *result = shape + root * z + (z2 - 1.0) / 3.0 +
                  (z3 - 7.0 * z) / (36.0 * root) -
                  (3.0 * z4 + 7.0 * z2 - 16.0) / (810.0 * shape);
        return isfinite(*result) && *result > 0.0;
    }
}

uint32_t nf_gamma_density_hybrid_f64_abi_version_v1(void) {
    return 1U;
}

int nf_gamma_density_hybrid_f64_apply_v1(
    const double *uniforms,
    const double *shapes,
    const double *scales,
    size_t sample_count,
    uint32_t inverse_iterations,
    double high_shape_threshold,
    double *output,
    size_t output_count,
    nf_gamma_density_hybrid_diagnostics_v1 *diagnostics) {
    size_t index;
    size_t direct_count = 0U;
    size_t asymptotic_count = 0U;
    double minimum_output = DBL_MAX;
    double maximum_output = 0.0;
    if (uniforms == NULL || shapes == NULL || scales == NULL || output == NULL ||
        diagnostics == NULL || sample_count == 0U || output_count < sample_count ||
        inverse_iterations < 32U || inverse_iterations > 128U ||
        !isfinite(high_shape_threshold) || high_shape_threshold <= 1.0 ||
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
            !nf_gamma_inverse_hybrid(uniforms[index], shapes[index], inverse_iterations,
                                     high_shape_threshold, &standardized)) {
            return 2;
        }
        developed = standardized * scales[index];
        if (!isfinite(developed) || developed <= 0.0) {
            return 2;
        }
        direct_count += shapes[index] < high_shape_threshold ? 1U : 0U;
        asymptotic_count += shapes[index] < high_shape_threshold ? 0U : 1U;
        minimum_output = developed < minimum_output ? developed : minimum_output;
        maximum_output = developed > maximum_output ? developed : maximum_output;
    }
    for (index = 0U; index < sample_count; ++index) {
        double standardized;
        if (!nf_gamma_inverse_hybrid(uniforms[index], shapes[index], inverse_iterations,
                                     high_shape_threshold, &standardized)) {
            return 3;
        }
        output[index] = standardized * scales[index];
    }
    diagnostics->inverse_iterations = inverse_iterations;
    diagnostics->sample_count = sample_count;
    diagnostics->direct_branch_count = direct_count;
    diagnostics->asymptotic_branch_count = asymptotic_count;
    diagnostics->minimum_output = minimum_output;
    diagnostics->maximum_output = maximum_output;
    return 0;
}
