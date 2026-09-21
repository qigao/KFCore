#include "esn.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "linalg.h"

#define KFCORE_ESN_UNIT24_SCALE (1.0f / 16777216.0f)
#define KFCORE_ESN_POWER_SEED UINT64_C(0xD1B54A32D192ED03)

static int kfcore_esn_reservoir_matrix_count(int reservoir_size, size_t* count)
{
    if (reservoir_size <= 0 || !count)
    {
        return 0;
    }

    const size_t size = (size_t)reservoir_size;
    if (size > SIZE_MAX / size)
    {
        return 0;
    }

    *count = size * size;
    return 1;
}

static uint64_t kfcore_esn_splitmix64_next(uint64_t* state)
{
    *state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t value = *state;
    value = (value ^ (value >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31U);
}

static float kfcore_esn_unit_float(uint64_t value)
{
    return (float)(value >> 40U) * KFCORE_ESN_UNIT24_SCALE;
}

static float kfcore_esn_signed_float(uint64_t value)
{
    return 2.0f * kfcore_esn_unit_float(value) - 1.0f;
}

static int kfcore_esn_finite_matrix(const float* values, size_t count)
{
    if (!values || count == 0U)
    {
        return 0;
    }

    for (size_t i = 0; i < count; ++i)
    {
        if (!isfinite(values[i]))
        {
            return 0;
        }
    }
    return 1;
}

kfcore_esn_status kfcore_esn_init_reservoir_weights(float* reservoir_weights,
                                                     int reservoir_size, uint64_t seed,
                                                     float density)
{
    size_t matrix_count = 0U;
    if (!reservoir_weights || !kfcore_esn_reservoir_matrix_count(reservoir_size, &matrix_count) ||
        !isfinite(density) || density <= 0.0f || density > 1.0f)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    uint64_t state = seed;
    for (size_t i = 0; i < matrix_count; ++i)
    {
        const float mask_sample = kfcore_esn_unit_float(kfcore_esn_splitmix64_next(&state));
        const float candidate = kfcore_esn_signed_float(kfcore_esn_splitmix64_next(&state));
        reservoir_weights[i] = mask_sample < density ? candidate : 0.0f;
    }
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_estimate_spectral_radius(const float* reservoir_weights,
                                                       int reservoir_size, int iterations,
                                                       float* workspace,
                                                       float* spectral_radius)
{
    size_t matrix_count = 0U;
    if (!reservoir_weights || !workspace || !spectral_radius || iterations <= 0 ||
        !kfcore_esn_reservoir_matrix_count(reservoir_size, &matrix_count) ||
        !kfcore_esn_finite_matrix(reservoir_weights, matrix_count))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    float* vector = workspace;
    float* product = workspace + (size_t)reservoir_size;
    uint64_t state = KFCORE_ESN_POWER_SEED;
    for (int i = 0; i < reservoir_size; ++i)
    {
        vector[i] = kfcore_esn_signed_float(kfcore_esn_splitmix64_next(&state));
    }

    const float vector_norm = vecnorm(vector, reservoir_size);
    if (!isfinite(vector_norm) || vector_norm <= 0.0f)
    {
        return KFCORE_ESN_NUMERICAL_FAILURE;
    }

    const float initial_inverse = 1.0f / vector_norm;
    for (int i = 0; i < reservoir_size; ++i)
    {
        vector[i] *= initial_inverse;
    }

    double log_growth = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        matvec("N", reservoir_size, reservoir_size, 1.0f, reservoir_weights, vector, 0.0f,
               product);
        const float product_norm = vecnorm(product, reservoir_size);
        if (!isfinite(product_norm) || product_norm <= 0.0f)
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }

        log_growth += log((double)product_norm);
        const float inverse = 1.0f / product_norm;
        for (int i = 0; i < reservoir_size; ++i)
        {
            vector[i] = product[i] * inverse;
        }
    }

    const double estimate = exp(log_growth / (double)iterations);
    if (!isfinite(estimate) || estimate <= 0.0 || estimate > (double)FLT_MAX)
    {
        return KFCORE_ESN_NUMERICAL_FAILURE;
    }

    *spectral_radius = (float)estimate;
    return *spectral_radius > 0.0f && isfinite(*spectral_radius) ? KFCORE_ESN_OK
                                                                : KFCORE_ESN_NUMERICAL_FAILURE;
}

kfcore_esn_status kfcore_esn_scale_spectral_radius(float* reservoir_weights,
                                                    int reservoir_size, float target_radius,
                                                    int iterations, float* workspace)
{
    size_t matrix_count = 0U;
    if (!reservoir_weights || !workspace ||
        !kfcore_esn_reservoir_matrix_count(reservoir_size, &matrix_count) ||
        !isfinite(target_radius) || target_radius <= 0.0f || iterations <= 0)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    float estimated_radius = 0.0f;
    const kfcore_esn_status estimate_status = kfcore_esn_estimate_spectral_radius(
        reservoir_weights, reservoir_size, iterations, workspace, &estimated_radius);
    if (estimate_status != KFCORE_ESN_OK)
    {
        return estimate_status;
    }

    const double scale = (double)target_radius / (double)estimated_radius;
    if (!isfinite(scale) || scale <= 0.0)
    {
        return KFCORE_ESN_NUMERICAL_FAILURE;
    }

    for (size_t i = 0; i < matrix_count; ++i)
    {
        const double scaled = (double)reservoir_weights[i] * scale;
        if (!isfinite(scaled) || scaled > (double)FLT_MAX || scaled < -(double)FLT_MAX)
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }
    }
    for (size_t i = 0; i < matrix_count; ++i)
    {
        reservoir_weights[i] = (float)((double)reservoir_weights[i] * scale);
    }
    return KFCORE_ESN_OK;
}
