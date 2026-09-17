#include "esn.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "linalg.h"

static int kfcore_esn_positive_dimensions(int first, int second, int third)
{
    return first > 0 && second > 0 && third > 0;
}

static int kfcore_esn_checked_product(int first, int second, size_t* product)
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

static int kfcore_esn_finite_array(const float* values, size_t count)
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

static kfcore_esn_status kfcore_esn_validate_reservoir(const kfcore_esn_model* model)
{
    if (!model || model->input_size <= 0 || model->reservoir_size <= 0 ||
        !isfinite(model->leak_rate) || model->leak_rate <= 0.0f || model->leak_rate > 1.0f ||
        !model->input_weights || !model->reservoir_weights || !model->reservoir_bias)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }
    return KFCORE_ESN_OK;
}

static kfcore_esn_status kfcore_esn_validate_readout(const kfcore_esn_model* model)
{
    if (!model || model->reservoir_size <= 0 || model->output_size <= 0 || !model->output_weights ||
        !model->output_bias)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_step(const kfcore_esn_model* model, const float* input, float* state,
                                  float* workspace)
{
    if (kfcore_esn_validate_reservoir(model) != KFCORE_ESN_OK || !input || !state || !workspace)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    matvec("N", model->reservoir_size, model->input_size, 1.0f, model->input_weights, input, 0.0f,
           workspace);
    matvec("N", model->reservoir_size, model->reservoir_size, 1.0f, model->reservoir_weights,
           state, 1.0f, workspace);

    const float keep = 1.0f - model->leak_rate;
    for (int i = 0; i < model->reservoir_size; ++i)
    {
        const float activated = tanhf(workspace[i] + model->reservoir_bias[i]);
        state[i] = keep * state[i] + model->leak_rate * activated;
    }

    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_predict(const kfcore_esn_model* model, const float* state,
                                     float* output)
{
    if (kfcore_esn_validate_readout(model) != KFCORE_ESN_OK || !state || !output)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    memcpy(output, model->output_bias, sizeof(float) * (size_t)model->output_size);
    matvec("N", model->output_size, model->reservoir_size, 1.0f, model->output_weights, state,
           1.0f, output);
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_step_predict(const kfcore_esn_model* model, const float* input,
                                          float* state, float* workspace, float* output)
{
    const kfcore_esn_status step_status = kfcore_esn_step(model, input, state, workspace);
    if (step_status != KFCORE_ESN_OK)
    {
        return step_status;
    }
    return kfcore_esn_predict(model, state, output);
}

kfcore_esn_status kfcore_esn_fit_ridge(const float* states, const float* targets,
                                       int reservoir_size, int output_size, int sample_count,
                                       float lambda, float* output_weights, float* gram_workspace)
{
    size_t state_count = 0U;
    size_t target_count = 0U;
    size_t weight_count = 0U;

    if (!states || !targets || !output_weights || !gram_workspace ||
        !kfcore_esn_positive_dimensions(reservoir_size, output_size, sample_count) ||
        !isfinite(lambda) || lambda <= 0.0f ||
        !kfcore_esn_checked_product(reservoir_size, sample_count, &state_count) ||
        !kfcore_esn_checked_product(output_size, sample_count, &target_count) ||
        !kfcore_esn_checked_product(output_size, reservoir_size, &weight_count))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    if (!kfcore_esn_finite_array(states, state_count) ||
        !kfcore_esn_finite_array(targets, target_count))
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    matmul("N", "T", reservoir_size, reservoir_size, sample_count, 1.0f, states, states, 0.0f,
           gram_workspace);
    for (int i = 0; i < reservoir_size; ++i)
    {
        MAT_ELEM(gram_workspace, i, i, reservoir_size, reservoir_size) += lambda;
    }

    matmul("N", "T", output_size, reservoir_size, sample_count, 1.0f, targets, states, 0.0f,
           output_weights);

    if (cholesky(gram_workspace, reservoir_size, 0) != 0)
    {
        return KFCORE_ESN_NUMERICAL_FAILURE;
    }

    trisolveright(gram_workspace, output_weights, reservoir_size, output_size, "T");
    trisolveright(gram_workspace, output_weights, reservoir_size, output_size, "N");

    for (size_t i = 0; i < weight_count; ++i)
    {
        if (!isfinite(output_weights[i]))
        {
            return KFCORE_ESN_NUMERICAL_FAILURE;
        }
    }

    return KFCORE_ESN_OK;
}
