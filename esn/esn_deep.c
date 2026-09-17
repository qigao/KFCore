#include "esn_deep.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static kfcore_esn_status kfcore_esn_deep_measure(const kfcore_esn_deep_model* deep,
                                                 int* external_input_size,
                                                 int* total_state_size,
                                                 int* max_reservoir_size)
{
    if (!deep || deep->layer_count <= 0 || !deep->layers || !external_input_size ||
        !total_state_size || !max_reservoir_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    const int input_size = deep->layers[0].input_size;
    if (input_size <= 0)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    size_t total = 0U;
    int max_layer = 0;
    for (int i = 0; i < deep->layer_count; ++i)
    {
        const kfcore_esn_model* layer = &deep->layers[i];
        if (layer->input_size <= 0 || layer->reservoir_size <= 0 ||
            !isfinite(layer->leak_rate) || layer->leak_rate <= 0.0f ||
            layer->leak_rate > 1.0f || !layer->input_weights ||
            !layer->reservoir_weights || !layer->reservoir_bias)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }

        if (i > 0 && layer->input_size != deep->layers[i - 1].reservoir_size)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }

        const size_t reservoir_size = (size_t)layer->reservoir_size;
        if (reservoir_size > (size_t)INT_MAX - total)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }
        total += reservoir_size;

        if (layer->reservoir_size > max_layer)
        {
            max_layer = layer->reservoir_size;
        }
    }

    *external_input_size = input_size;
    *total_state_size = (int)total;
    *max_reservoir_size = max_layer;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_deep_layout(const kfcore_esn_deep_model* deep,
                                         int* state_size, int* workspace_size)
{
    if (!state_size || !workspace_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int input_size = 0;
    int total = 0;
    int max_layer = 0;
    const kfcore_esn_status status =
        kfcore_esn_deep_measure(deep, &input_size, &total, &max_layer);
    if (status != KFCORE_ESN_OK)
    {
        return status;
    }

    if (total > INT_MAX - max_layer)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    *state_size = total;
    *workspace_size = total + max_layer;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_deep_step(const kfcore_esn_deep_model* deep,
                                       const float* input, float* state,
                                       float* workspace)
{
    if (!input || !state || !workspace)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int input_size = 0;
    int total_state_size = 0;
    int max_layer = 0;
    const kfcore_esn_status measure_status =
        kfcore_esn_deep_measure(deep, &input_size, &total_state_size, &max_layer);
    if (measure_status != KFCORE_ESN_OK)
    {
        return measure_status;
    }

    if (total_state_size > INT_MAX - max_layer)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    float* candidate = workspace;
    float* scratch = workspace + total_state_size;
    memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);

    const float* layer_input = input;
    int offset = 0;
    for (int i = 0; i < deep->layer_count; ++i)
    {
        const kfcore_esn_model* layer = &deep->layers[i];
        float* layer_state = candidate + offset;
        const kfcore_esn_status status =
            kfcore_esn_step(layer, layer_input, layer_state, scratch);
        if (status != KFCORE_ESN_OK)
        {
            return status;
        }

        layer_input = layer_state;
        offset += layer->reservoir_size;
    }

    memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
    return KFCORE_ESN_OK;
}
