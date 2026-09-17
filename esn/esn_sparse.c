#include "esn_sparse.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "linalg.h"

static int kfcore_esn_sparse_matrix_count(int reservoir_size, size_t* count)
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

static kfcore_esn_status kfcore_esn_sparse_count_dense(const float* reservoir_weights,
                                                        int reservoir_size,
                                                        int* nonzero_count)
{
    size_t matrix_count = 0U;
    if (!reservoir_weights || !nonzero_count ||
        !kfcore_esn_sparse_matrix_count(reservoir_size, &matrix_count))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    size_t count = 0U;
    for (size_t i = 0; i < matrix_count; ++i)
    {
        if (!isfinite(reservoir_weights[i]))
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }
        if (reservoir_weights[i] != 0.0f)
        {
            ++count;
            if (count > (size_t)INT_MAX)
            {
                return KFCORE_ESN_INVALID_ARGUMENT;
            }
        }
    }

    *nonzero_count = (int)count;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_sparse_count_nonzero(const float* reservoir_weights,
                                                  int reservoir_size,
                                                  int* nonzero_count)
{
    int count = 0;
    const kfcore_esn_status status =
        kfcore_esn_sparse_count_dense(reservoir_weights, reservoir_size, &count);
    if (status != KFCORE_ESN_OK)
    {
        return status;
    }

    *nonzero_count = count;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_sparse_from_dense(const float* reservoir_weights,
                                               int reservoir_size, int capacity,
                                               int* row_offsets, int* column_indices,
                                               float* values, int* nonzero_count)
{
    if (!row_offsets || !nonzero_count || capacity < 0)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int required = 0;
    const kfcore_esn_status count_status =
        kfcore_esn_sparse_count_dense(reservoir_weights, reservoir_size, &required);
    if (count_status != KFCORE_ESN_OK)
    {
        return count_status;
    }

    if (required > capacity || (capacity > 0 && (!column_indices || !values)))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int cursor = 0;
    row_offsets[0] = 0;
    for (int row = 0; row < reservoir_size; ++row)
    {
        for (int column = 0; column < reservoir_size; ++column)
        {
            const float value =
                reservoir_weights[(size_t)row + (size_t)column * (size_t)reservoir_size];
            if (value != 0.0f)
            {
                column_indices[cursor] = column;
                values[cursor] = value;
                ++cursor;
            }
        }
        row_offsets[row + 1] = cursor;
    }

    *nonzero_count = required;
    return KFCORE_ESN_OK;
}

static kfcore_esn_status
kfcore_esn_validate_sparse_reservoir(const kfcore_esn_sparse_reservoir* sparse,
                                     int reservoir_size)
{
    if (!sparse || reservoir_size <= 0 || sparse->reservoir_size != reservoir_size ||
        sparse->nonzero_count < 0 || !sparse->row_offsets)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    const int nonzero_count = sparse->nonzero_count;
    if (sparse->row_offsets[0] != 0 || sparse->row_offsets[reservoir_size] != nonzero_count)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    if (nonzero_count > 0 && (!sparse->column_indices || !sparse->values))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    for (int row = 0; row < reservoir_size; ++row)
    {
        const int begin = sparse->row_offsets[row];
        const int end = sparse->row_offsets[row + 1];
        if (begin < 0 || end < begin || end > nonzero_count)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }

        int previous_column = -1;
        for (int entry = begin; entry < end; ++entry)
        {
            const int column = sparse->column_indices[entry];
            if (column < 0 || column >= reservoir_size || column <= previous_column ||
                !isfinite(sparse->values[entry]))
            {
                return KFCORE_ESN_INVALID_ARGUMENT;
            }
            previous_column = column;
        }
    }

    return KFCORE_ESN_OK;
}

static kfcore_esn_status kfcore_esn_validate_sparse_model(const kfcore_esn_model* model)
{
    if (!model || model->input_size <= 0 || model->reservoir_size <= 0 ||
        !isfinite(model->leak_rate) || model->leak_rate <= 0.0f || model->leak_rate > 1.0f ||
        !model->input_weights || !model->reservoir_bias)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_step_sparse(const kfcore_esn_model* model,
                                         const kfcore_esn_sparse_reservoir* sparse,
                                         const float* input, float* state,
                                         float* workspace)
{
    if (kfcore_esn_validate_sparse_model(model) != KFCORE_ESN_OK || !input || !state ||
        !workspace)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    const kfcore_esn_status sparse_status =
        kfcore_esn_validate_sparse_reservoir(sparse, model->reservoir_size);
    if (sparse_status != KFCORE_ESN_OK)
    {
        return sparse_status;
    }

    matvec("N", model->reservoir_size, model->input_size, 1.0f, model->input_weights, input,
           0.0f, workspace);

    for (int row = 0; row < model->reservoir_size; ++row)
    {
        double recurrent = 0.0;
        const int begin = sparse->row_offsets[row];
        const int end = sparse->row_offsets[row + 1];
        for (int entry = begin; entry < end; ++entry)
        {
            recurrent += (double)sparse->values[entry] *
                         (double)state[sparse->column_indices[entry]];
        }

        const double combined = (double)workspace[row] + recurrent;
        if (!isfinite(combined) || combined > (double)FLT_MAX || combined < -(double)FLT_MAX)
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }
        workspace[row] = (float)combined;
    }

    const float keep = 1.0f - model->leak_rate;
    for (int row = 0; row < model->reservoir_size; ++row)
    {
        const float activated = tanhf(workspace[row] + model->reservoir_bias[row]);
        const float next_state = keep * state[row] + model->leak_rate * activated;
        if (!isfinite(next_state))
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }
    }

    for (int row = 0; row < model->reservoir_size; ++row)
    {
        const float activated = tanhf(workspace[row] + model->reservoir_bias[row]);
        state[row] = keep * state[row] + model->leak_rate * activated;
    }

    return KFCORE_ESN_OK;
}
