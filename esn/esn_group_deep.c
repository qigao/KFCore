#include "esn_group_deep.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static kfcore_esn_status kfcore_esn_grouped_deep_measure(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* common_input_size,
    int* total_state_size,
    int* max_group_workspace_size)
{
    if (!grouped_deep || grouped_deep->group_count <= 0 || !grouped_deep->groups ||
        !common_input_size || !total_state_size || !max_group_workspace_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int expected_input_size = 0;
    int max_workspace = 0;
    size_t total = 0U;

    for (int i = 0; i < grouped_deep->group_count; ++i)
    {
        int group_state_size = 0;
        int group_workspace_size = 0;
        const kfcore_esn_status status =
            kfcore_esn_deep_layout(&grouped_deep->groups[i],
                                   &group_state_size,
                                   &group_workspace_size);
        if (status != KFCORE_ESN_OK)
        {
            return status;
        }

        const int group_input_size = grouped_deep->groups[i].layers[0].input_size;
        if (i == 0)
        {
            expected_input_size = group_input_size;
        }
        else if (group_input_size != expected_input_size)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }

        const size_t state_size = (size_t)group_state_size;
        if (state_size > (size_t)INT_MAX - total)
        {
            return KFCORE_ESN_INVALID_ARGUMENT;
        }
        total += state_size;

        if (group_workspace_size > max_workspace)
        {
            max_workspace = group_workspace_size;
        }
    }

    *common_input_size = expected_input_size;
    *total_state_size = (int)total;
    *max_group_workspace_size = max_workspace;
    return KFCORE_ESN_OK;
}

kfcore_esn_status kfcore_esn_grouped_deep_layout(
    const kfcore_esn_grouped_deep_model* grouped_deep,
    int* state_size,
    int* workspace_size)
{
    if (!state_size || !workspace_size)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    int input_size = 0;
    int total = 0;
    int max_group_workspace = 0;
    const kfcore_esn_status status =
        kfcore_esn_grouped_deep_measure(grouped_deep, &input_size,
                                        &total, &max_group_workspace);
    if (status != KFCORE_ESN_OK)
    {
        return status;
    }

    if (total > INT_MAX - max_group_workspace)
    {
        return KFCORE_ESN_INVALID_ARGUMENT;
    }

    *state_size = total;
    *workspace_size = total + max_group_workspace;
    return KFCORE_ESN_OK;
}
