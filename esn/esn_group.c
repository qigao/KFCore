#include "esn_group.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static kfcore_esn_status kfcore_esn_grouped_measure(
    const kfcore_esn_grouped_model* grouped,
    int* common_input_size,
    int* total_state_size,
    int* max_reservoir_size)
{
    if (!grouped || grouped->group_count <= 0 || !grouped->groups || !common_input_size ||
        !total_state_size || !max_reservoir_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    const int expected_input_size = grouped->groups[0].input_size;
    if (expected_input_size <= 0)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    size_t total = 0U;
    int max_group = 0;
    for (int i = 0; i < grouped->group_count; ++i)
    {
        const kfcore_esn_model* group = &grouped->groups[i];
        if (group->input_size != expected_input_size || group->reservoir_size <= 0 ||
            !isfinite(group->leak_rate) || group->leak_rate <= 0.0f || group->leak_rate > 1.0f ||
            !group->input_weights || !group->reservoir_weights || !group->reservoir_bias)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }

        const size_t reservoir_size = (size_t)group->reservoir_size;
        if (reservoir_size > (size_t)INT_MAX - total)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }
        total += reservoir_size;

        if (group->reservoir_size > max_group)
        {
            max_group = group->reservoir_size;
        }
    }

    *common_input_size = expected_input_size;
    *total_state_size = (int)total;
    *max_reservoir_size = max_group;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_grouped_layout(const kfcore_esn_grouped_model* grouped,
                                            int* state_size, int* workspace_size)
{
    if (!state_size || !workspace_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int common_input_size = 0;
    int total = 0;
    int max_group = 0;
    const kfcore_esn_status status =
        kfcore_esn_grouped_measure(grouped, &common_input_size, &total, &max_group);
    if (status != KFCORE_ESN_OK)
    {
        return status;
    }

    if (total > INT_MAX - max_group)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    *state_size = total;
    *workspace_size = total + max_group;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_grouped_step(const kfcore_esn_grouped_model* grouped,
                                          const float* input, float* state,
                                          float* workspace)
{
    if (!input || !state || !workspace)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int common_input_size = 0;
    int total_state_size = 0;
    int max_reservoir_size = 0;
    const kfcore_esn_status measure_status =
        kfcore_esn_grouped_measure(grouped, &common_input_size, &total_state_size,
                                   &max_reservoir_size);
    if (measure_status != KFCORE_ESN_OK)
    {
        return measure_status;
    }

    if (total_state_size > INT_MAX - max_reservoir_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    float* candidate = workspace;
    float* scratch = workspace + total_state_size;
    memcpy(candidate, state, sizeof(float) * (size_t)total_state_size);

    int offset = 0;
    for (int i = 0; i < grouped->group_count; ++i)
    {
        const kfcore_esn_model* group = &grouped->groups[i];
        const kfcore_esn_status step_status =
            kfcore_esn_step(group, input, candidate + offset, scratch);
        if (step_status != KFCORE_ESN_OK)
        {
            return step_status;
        }
        offset += group->reservoir_size;
    }

    memcpy(state, candidate, sizeof(float) * (size_t)total_state_size);
    return KFCORE_ESN_OK;
}
