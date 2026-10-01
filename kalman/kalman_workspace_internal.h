#ifndef KFCORE_KALMAN_WORKSPACE_INTERNAL_H
#define KFCORE_KALMAN_WORKSPACE_INTERNAL_H

#include "kalman_status.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

static inline kfcore_kalman_status kfcore_kalman_check_dim(size_t value, int allow_zero)
{
    if ((!allow_zero && value == 0U) || value > (size_t)INT_MAX)
        return KFCORE_KALMAN_INVALID_ARGUMENT;
    return KFCORE_KALMAN_OK;
}

static inline kfcore_kalman_status kfcore_kalman_checked_add(size_t a, size_t b, size_t* out)
{
    if (!out) return KFCORE_KALMAN_INVALID_ARGUMENT;
    if (a > SIZE_MAX - b) return KFCORE_KALMAN_SIZE_OVERFLOW;
    *out = a + b;
    return KFCORE_KALMAN_OK;
}

static inline kfcore_kalman_status kfcore_kalman_checked_mul(size_t a, size_t b, size_t* out)
{
    if (!out) return KFCORE_KALMAN_INVALID_ARGUMENT;
    if (a != 0U && b > SIZE_MAX / a) return KFCORE_KALMAN_SIZE_OVERFLOW;
    *out = a * b;
    return KFCORE_KALMAN_OK;
}

static inline kfcore_kalman_status kfcore_kalman_require_workspace(
    float* workspace, size_t workspace_floats, size_t required)
{
    if (workspace_floats < required || (required != 0U && !workspace))
        return KFCORE_KALMAN_WORKSPACE_TOO_SMALL;
    return KFCORE_KALMAN_OK;
}

static inline kfcore_kalman_status kfcore_kalman_linalg_status(int status)
{
    return status == 0 ? KFCORE_KALMAN_OK : KFCORE_KALMAN_NUMERICAL_FAILURE;
}

#endif
