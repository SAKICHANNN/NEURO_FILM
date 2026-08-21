#define NF_DNG_CAMERA_TO_PCS_BUILD
#include "nf_dng_camera_to_pcs_v1.h"

#include <math.h>
#include <stdint.h>

static double nf_det3(const double matrix[9]) {
    return
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
        matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
        matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

static int nf_matrix_valid(const double matrix[9]) {
    size_t index;
    double determinant;
    for (index = 0; index < 9; ++index) {
        if (!isfinite(matrix[index])) {
            return 0;
        }
    }
    determinant = nf_det3(matrix);
    return isfinite(determinant) && fabs(determinant) >= 1.0e-12;
}

static int nf_ranges_overlap(
    const void* first,
    const void* second,
    size_t byte_count) {
    const uintptr_t first_begin = (uintptr_t)first;
    const uintptr_t second_begin = (uintptr_t)second;
    const uintptr_t first_end = first_begin + byte_count;
    const uintptr_t second_end = second_begin + byte_count;
    if (first_end < first_begin || second_end < second_begin) {
        return 1;
    }
    return first_begin < second_end && second_begin < first_end;
}

static void nf_transform(
    const double matrix[9],
    double x,
    double y,
    double z,
    double output[3]) {
    output[0] = (matrix[0] * x + matrix[1] * y) + matrix[2] * z;
    output[1] = (matrix[3] * x + matrix[4] * y) + matrix[5] * z;
    output[2] = (matrix[6] * x + matrix[7] * y) + matrix[8] * z;
}

uint32_t nf_dng_camera_to_pcs_abi_version_v1(void) {
    return NF_DNG_CAMERA_TO_PCS_ABI_VERSION_V1;
}

nf_dng_camera_to_pcs_status_v1 nf_dng_camera_to_pcs_apply_v1(
    const double matrix[9],
    const double* input_camera,
    size_t triplet_count,
    double* output_pcs,
    size_t output_value_count) {
    size_t value_count;
    size_t index;
    if (
        matrix == NULL || input_camera == NULL || output_pcs == NULL ||
        triplet_count == 0 || triplet_count > SIZE_MAX / 3
    ) {
        return NF_DNG_CAMERA_TO_PCS_INVALID_ARGUMENT_V1;
    }
    value_count = triplet_count * 3;
    if (output_value_count < value_count || value_count > SIZE_MAX / sizeof(double)) {
        return NF_DNG_CAMERA_TO_PCS_INVALID_ARGUMENT_V1;
    }
    if (!nf_matrix_valid(matrix)) {
        return NF_DNG_CAMERA_TO_PCS_INVALID_MATRIX_V1;
    }
    if (
        input_camera != output_pcs &&
        nf_ranges_overlap(input_camera, output_pcs, value_count * sizeof(double))
    ) {
        return NF_DNG_CAMERA_TO_PCS_OVERLAP_V1;
    }
    for (index = 0; index < triplet_count; ++index) {
        const double x = input_camera[index * 3];
        const double y = input_camera[index * 3 + 1];
        const double z = input_camera[index * 3 + 2];
        double transformed[3];
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
            return NF_DNG_CAMERA_TO_PCS_INVALID_INPUT_V1;
        }
        nf_transform(matrix, x, y, z, transformed);
        if (
            !isfinite(transformed[0]) || !isfinite(transformed[1]) ||
            !isfinite(transformed[2])
        ) {
            return NF_DNG_CAMERA_TO_PCS_NUMERIC_FAILURE_V1;
        }
    }
    for (index = 0; index < triplet_count; ++index) {
        const double x = input_camera[index * 3];
        const double y = input_camera[index * 3 + 1];
        const double z = input_camera[index * 3 + 2];
        double transformed[3];
        nf_transform(matrix, x, y, z, transformed);
        output_pcs[index * 3] = transformed[0];
        output_pcs[index * 3 + 1] = transformed[1];
        output_pcs[index * 3 + 2] = transformed[2];
    }
    return NF_DNG_CAMERA_TO_PCS_OK_V1;
}
