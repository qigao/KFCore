#include "esn.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "linalg.h"

static int kfcore_esn_rls_checked_product(int first, int second, size_t* product)
{
    if (first <= 0 || second <= 0 || !product)
    {
        return 0;
    }

    const size_t left = (size_t)first;
    const size_t right = (size_t)second;
    if (left > SIZE_MAX / right)
    {
        return 0;
    }

    *product = left * right;
    return 1;
}

static int kfcore_esn_rls_finite_array(const float* values, size_t count)
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

static int kfcore_esn_rls_fits_float(double value)
{
    return isfinite(value) && value <= (double)FLT_MAX && value >= -(double)FLT_MAX;
}

kfcore_esn_status kfcore_esn_rls_init(float* inverse_correlation, int reservoir_size,
                                      float delta)
{
    size_t matrix_count = 0U;
    if (!inverse_correlation || !kfcore_esn_rls_checked_product(reservoir_size, reservoir_size,
                                                                &matrix_count) ||
        matrix_count > SIZE_MAX / sizeof(float) || !isfinite(delta) || delta <= 0.0f)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    const double inverse_delta = 1.0 / (double)delta;
    if (!kfcore_esn_rls_fits_float(inverse_delta) || inverse_delta <= 0.0)
    {
        return KFCORE_ESN_NUMERICAL_FAILURE;
    }

    memset(inverse_correlation, 0, matrix_count * sizeof(float));
    const float diagonal = (float)inverse_delta;
    for (int i = 0; i < reservoir_size; ++i)
    {
        MAT_ELEM(inverse_correlation, i, i, reservoir_size, reservoir_size) = diagonal;
    }
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_rls_update(const float* state, const float* target,
                                        int reservoir_size, int output_size,
                                        float forgetting_factor, float* output_weights,
                                        float* inverse_correlation, float* workspace)
{
    size_t weight_count = 0U;
    size_t correlation_count = 0U;
    if (!state || !target || !output_weights || !inverse_correlation || !workspace ||
        !kfcore_esn_rls_checked_product(output_size, reservoir_size, &weight_count) ||
        !kfcore_esn_rls_checked_product(reservoir_size, reservoir_size, &correlation_count) ||
        !isfinite(forgetting_factor) || forgetting_factor <= 0.0f || forgetting_factor > 1.0f)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    if (!kfcore_esn_rls_finite_array(state, (size_t)reservoir_size) ||
        !kfcore_esn_rls_finite_array(target, (size_t)output_size) ||
        !kfcore_esn_rls_finite_array(output_weights, weight_count) ||
        !kfcore_esn_rls_finite_array(inverse_correlation, correlation_count))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    float* gain = workspace;
    float* error = workspace + (size_t)reservoir_size;

    matvec("N", reservoir_size, reservoir_size, 1.0f, inverse_correlation, state, 0.0f, gain);

    const double denominator =
        (double)forgetting_factor + (double)vecdot(state, gain, reservoir_size);
    if (!isfinite(denominator) || denominator <= 0.0 || denominator > (double)FLT_MAX)
    {
        return KFCORE_ESN_NUMERICAL_FAILURE;
    }

    for (int i = 0; i < reservoir_size; ++i)
    {
        const double value = (double)gain[i] / denominator;
        if (!kfcore_esn_rls_fits_float(value))
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }
        gain[i] = (float)value;
    }

    matvec("N", output_size, reservoir_size, 1.0f, output_weights, state, 0.0f, error);
    for (int i = 0; i < output_size; ++i)
    {
        const double residual = (double)target[i] - (double)error[i];
        if (!kfcore_esn_rls_fits_float(residual))
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }
        error[i] = (float)residual;
    }

    for (int col = 0; col < reservoir_size; ++col)
    {
        for (int row = 0; row < output_size; ++row)
        {
            const size_t index = (size_t)row + (size_t)col * (size_t)output_size;
            const double updated =
                (double)output_weights[index] + (double)error[row] * (double)gain[col];
            if (!kfcore_esn_rls_fits_float(updated))
            {
                return KFCORE_ESN_NUMERICAL_FAILURE;
            }
        }
    }

    for (int col = 0; col < reservoir_size; ++col)
    {
        for (int row = 0; row < reservoir_size; ++row)
        {
            const size_t index = (size_t)row + (size_t)col * (size_t)reservoir_size;
            const double numerator = (double)inverse_correlation[index] -
                                     denominator * (double)gain[row] * (double)gain[col];
            const double updated = numerator / (double)forgetting_factor;
            if (!kfcore_esn_rls_fits_float(numerator) || !kfcore_esn_rls_fits_float(updated))
            {
                return KFCORE_ESN_NUMERICAL_FAILURE;
            }
        }
    }

    rank1update(output_weights, error, gain, output_size, reservoir_size, 1.0f);
    rank1update(inverse_correlation, gain, gain, reservoir_size, reservoir_size,
                -(float)denominator);

    if (forgetting_factor != 1.0f)
    {
        for (size_t i = 0; i < correlation_count; ++i)
        {
            inverse_correlation[i] =
                (float)((double)inverse_correlation[i] / (double)forgetting_factor);
        }
    }

    return KFCORE_ESN_OK;
}
